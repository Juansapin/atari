import random
from collections import deque
from pathlib import Path
from typing import Self

import ale_py
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

gym.register_envs(ale_py)


# ── Helper: construir entorno preprocesado ────────────────────────────


def make_env(env_id: str, render_mode=None):
    env = gym.make(env_id, render_mode=render_mode, frameskip=1)
    env = AtariPreprocessing(
        env,
        screen_size=84,
        grayscale_obs=True,
        frame_skip=4,
        grayscale_newaxis=False,
    )
    env = FrameStackObservation(env, stack_size=4)
    return env


# ── Neural network (CNN estilo DeepMind) ──────────────────────────────


class QNetwork(nn.Module):
    """
    Arquitectura convolucional para observaciones de imagen (4, 84, 84).
    Basada en Mnih et al. 2015 (Nature DQN).

    Capas convolucionales:
      Conv1: 4 canales → 32 filtros, kernel 8x8, stride 4  → (32, 20, 20)
      Conv2: 32 → 64 filtros, kernel 4x4, stride 2         → (64, 9, 9)
      Conv3: 64 → 64 filtros, kernel 3x3, stride 1         → (64, 7, 7)

    Capas fully connected:
      FC1: 3136 → 512
      FC2: 512  → n_actions
    """

    def __init__(self, n_actions: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalizar píxeles de 0-255 a 0-1
        x = x.float() / 255.0
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        return self.fc(x)


# ── Replay Buffer ────────────────────────────────────────────────────


class ReplayBuffer:
    """
    Buffer de experiencias para DQN.
    Almacena transiciones (s, a, r, s', done) y permite
    muestreo aleatorio para romper correlaciones temporales.
    """

    def __init__(self, capacity: int = 100_000) -> None:
        self.buffer: deque[tuple] = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> list[tuple]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


# ── Agent ─────────────────────────────────────────────────────────────


class DQNAgent:
    """
    Agente DQN para Space Invaders con observaciones de imagen.

    Implementa:
    - Red neuronal convolucional (QNetwork) para aproximar Q(s, a)
    - Target network para estabilizar el entrenamiento
    - Experience replay para romper correlaciones temporales
    - Política epsilon-greedy con decaimiento
    """

    def __init__(
        self,
        env_id: str,
        *,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.9995,
        batch_size: int = 32,
        buffer_capacity: int = 100_000,
        target_update_freq: int = 1000,
        learning_starts: int = 10_000,
    ) -> None:
        self.env_id = env_id
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learning_starts = learning_starts
        self.training_episodes = 0
        self.total_steps = 0

        # Obtener número de acciones del entorno
        env = make_env(env_id)
        self.action_dim = int(env.action_space.n)
        env.close()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando dispositivo: {self.device}")

        # Redes: online (se entrena) y target (se congela periódicamente)
        self.q_net = QNetwork(self.action_dim).to(self.device)
        self.target_net = QNetwork(self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss, más estable que MSE
        self.buffer = ReplayBuffer(buffer_capacity)

    # ── Selección de acción ───────────────────────────────────────────

    def select_action(self, state: np.ndarray, *, deterministic: bool = False) -> int:
        if not deterministic and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            t = torch.from_numpy(np.array(state)).unsqueeze(0).to(self.device)
            return int(self.q_net(t).argmax(dim=1).item())

    def predict(
        self, obs: np.ndarray, *, deterministic: bool = True
    ) -> tuple[int, None]:
        return self.select_action(obs, deterministic=deterministic), None

    # ── Paso de aprendizaje ───────────────────────────────────────────

    def _learn(self) -> float:
        if len(self.buffer) < self.batch_size:
            return 0.0

        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.from_numpy(np.array(states)).to(self.device)
        next_states_t = torch.from_numpy(np.array(next_states)).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Q(s, a) actual
        current_q = self.q_net(states_t).gather(1, actions_t)

        # Target: r + γ * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1, keepdim=True).values

        target_q = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        return loss.item()

    # ── Loop de entrenamiento ─────────────────────────────────────────

    def train(self, total_episodes: int = 1000, log_interval: int = 20) -> list[float]:
        env = make_env(self.env_id)
        rewards_history: list[float] = []
        best_avg_reward = -float("inf")
        solved_threshold = 500  # Space Invaders: ~500 puntos es un buen objetivo

        print(f"Acciones disponibles: {self.action_dim}")
        print(f"Aprendizaje comienza en paso: {self.learning_starts}")

        for episode in range(1, total_episodes + 1):
            obs, _ = env.reset()
            total_reward, done = 0.0, False

            while not done:
                action = self.select_action(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                self.buffer.push(obs, action, float(reward), next_obs, done)
                self.total_steps += 1

                if self.total_steps >= self.learning_starts:
                    self._learn()

                    if self.total_steps % self.target_update_freq == 0:
                        self.target_net.load_state_dict(self.q_net.state_dict())

                obs = next_obs
                total_reward += reward

            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            self.training_episodes += 1
            rewards_history.append(total_reward)

            if episode % log_interval == 0:
                avg = np.mean(rewards_history[-log_interval:])
                print(
                    f"Ep {episode:4d}/{total_episodes} | "
                    f"Avg: {avg:7.2f} | "
                    f"Eps: {self.epsilon:.4f} | "
                    f"Steps: {self.total_steps:,} | "
                    f"Buffer: {len(self.buffer):,}"
                )

                if avg > best_avg_reward:
                    best_avg_reward = avg
                    self.save(Path("saves/dqn_si_best.pt"))
                    print(f"  Nuevo mejor modelo guardado (avg={avg:.2f})")

                if avg >= solved_threshold:
                    print(f"\nResuelto con avg {avg:.2f} >= {solved_threshold}!")
                    self.save(Path("saves/dqn_si_final.pt"))
                    break

        env.close()
        self.save(Path("saves/dqn_si_best.pt"))
        print("Modelo guardado al finalizar entrenamiento.")
        return rewards_history

    # ── Persistencia ──────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "q_net_state": self.q_net.state_dict(),
                "epsilon": self.epsilon,
                "total_steps": self.total_steps,
                "training_episodes": self.training_episodes,
                "env_id": self.env_id,
                "lr": self.lr,
                "gamma": self.gamma,
                "batch_size": self.batch_size,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> Self:
        data = torch.load(path, weights_only=False)
        agent = cls(
            data["env_id"],
            lr=data["lr"],
            gamma=data["gamma"],
            batch_size=data["batch_size"],
        )
        agent.q_net.load_state_dict(data["q_net_state"])
        agent.target_net.load_state_dict(data["q_net_state"])
        agent.epsilon = data["epsilon"]
        agent.total_steps = data.get("total_steps", 0)
        agent.training_episodes = data.get("training_episodes", 0)
        return agent
