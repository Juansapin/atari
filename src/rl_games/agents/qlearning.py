import pickle
from collections import defaultdict
from pathlib import Path


import gymnasium as gym
import numpy as np

_OBS_BOUNDS = np.array(
    [
        [-1.5, 1.5],
        [-0.5, 1.5],
        [-5.0, 5.0],
        [-5.0, 5.0],
        [-3.14, 3.14],
        [-5.0, 5.0],
    ]
)

_N_ACTIONS = 4


class QLearningAgent:
    def __init__(
        self,
        env_id: str,
        *,
        n_bins: int = 14,  # más granularidad
        lr: float = 0.1,  # más agresivo al inicio
        lr_min: float = 0.01,  # frena al madurar
        lr_decay: float = 0.9999,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.9995,  # más rápido
    ) -> None:
        self.env_id = env_id
        self.n_bins = n_bins
        self.lr = lr
        self.lr_min = lr_min
        self.lr_decay = lr_decay
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.training_episodes = 0

        # Bins no uniformes: más densidad cerca del suelo y del centro
        self._bins = self._make_bins()
        self.q_table: dict[tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(_N_ACTIONS)
        )

    def _make_bins(self) -> list[np.ndarray]:
        """Bins con distribución no uniforme para dimensiones críticas."""
        lo, hi = _OBS_BOUNDS[:, 0], _OBS_BOUNDS[:, 1]
        bins = []
        for i in range(6):
            if i in (0, 1):
                # x, y: más denso cerca del centro/suelo
                edges = np.sign(np.linspace(-1, 1, self.n_bins + 1))
                edges = lo[i] + (edges + 1) / 2 * (hi[i] - lo[i])
            else:
                edges = np.linspace(lo[i], hi[i], self.n_bins + 1)
            bins.append(edges[1:-1])
        return bins

    def discretize(self, obs: np.ndarray) -> tuple:
        continuous = np.clip(obs[:6], _OBS_BOUNDS[:, 0], _OBS_BOUNDS[:, 1])
        indices = [int(np.digitize(continuous[i], self._bins[i])) for i in range(6)]
        indices.append(int(obs[6]))
        indices.append(int(obs[7]))
        return tuple(indices)

    def select_action(self, state: tuple, *, deterministic: bool = False) -> int:
        if not deterministic and np.random.random() < self.epsilon:
            return np.random.randint(_N_ACTIONS)
        return int(np.argmax(self.q_table[state]))

    def predict(
        self, obs: np.ndarray, *, deterministic: bool = True
    ) -> tuple[int, None]:
        state = self.discretize(obs)
        return self.select_action(state, deterministic=deterministic), None

    def _update(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
        done: bool,
    ) -> None:
        best_next = 0.0 if done else float(np.max(self.q_table[next_state]))
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.lr * td_error

    def train(
        self, total_episodes: int = 30_000, log_interval: int = 500
    ) -> list[float]:
        env = gym.make(self.env_id)
        rewards_history: list[float] = []
        best_avg = -np.inf

        for episode in range(1, total_episodes + 1):
            obs, _ = env.reset()
            state = self.discretize(obs)
            total_reward = 0.0
            done = False

            while not done:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                next_state = self.discretize(next_obs)
                self._update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

            # Decaimiento de epsilon y lr
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            self.lr = max(self.lr_min, self.lr * self.lr_decay)
            self.training_episodes += 1
            rewards_history.append(total_reward)

            if episode % log_interval == 0:
                avg = np.mean(rewards_history[-log_interval:])
                tag = " ★" if avg > best_avg else ""
                if avg > best_avg:
                    best_avg = avg
                print(
                    f"Ep {episode:>6}/{total_episodes} | "
                    f"Avg: {avg:>8.2f} | "
                    f"ε: {self.epsilon:.4f} | "
                    f"lr: {self.lr:.4f} | "
                    f"States: {len(self.q_table):>7}{tag}"
                )

        env.close()
        return rewards_history

    # ── persistence (igual que antes) ─────────────────────────────────────────

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "q_table": dict(self.q_table),
            "epsilon": self.epsilon,
            "training_episodes": self.training_episodes,
            "env_id": self.env_id,
            "n_bins": self.n_bins,
            "lr": self.lr,
            "lr_min": self.lr_min,
            "lr_decay": self.lr_decay,
            "gamma": self.gamma,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"Guardado en {path}")

    @classmethod
    def load(cls, path: Path) -> "QLearningAgent":
        with open(path, "rb") as f:
            data = pickle.load(f)  # noqa: S301
        agent = cls(
            env_id=data["env_id"],
            n_bins=data["n_bins"],
            lr=data["lr"],
            lr_min=data.get("lr_min", 0.01),
            lr_decay=data.get("lr_decay", 0.9999),
            gamma=data["gamma"],
            epsilon_start=data["epsilon"],
            epsilon_end=data["epsilon_end"],
            epsilon_decay=data["epsilon_decay"],
        )
        agent.q_table = defaultdict(lambda: np.zeros(_N_ACTIONS), data["q_table"])
        agent.training_episodes = data["training_episodes"]
        return agent

    def info(self) -> str:
        return (
            f"Q-Learning agent for {self.env_id}\n"
            f"  Episodes trained : {self.training_episodes}\n"
            f"  States visited   : {len(self.q_table)}\n"
            f"  Epsilon          : {self.epsilon:.4f}\n"
            f"  LR               : {self.lr:.4f}\n"
            f"  Gamma            : {self.gamma}"
        )
