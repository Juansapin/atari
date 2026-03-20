import argparse
import ale_py                          # ← agregar
import gymnasium as gym
import numpy as np
from pathlib import Path
from importlib.metadata import version
from gymnasium.wrappers import FrameStackObservation, AtariPreprocessing

gym.register_envs(ale_py)             # ← agregar, después de los imports

# ── Configuración Principal ──────────────────────────────────────────
ENV_ID = "ALE/DonkeyKong-v5"
SAVE_DIR = Path("saves")
AGENT_CHOICES = ("qlearning", "dqn")
VERSION = version("rl_games")

# Acciones estándar de Atari
ACTION_NAMES = {
    0: "NOOP",      1: "FIRE",       2: "UP",
    3: "RIGHT",     4: "LEFT",       5: "DOWN",
    6: "UPRIGHT",   7: "UPLEFT",     8: "DOWNRIGHT",
    9: "DOWNLEFT",  10: "UPFIRE",    11: "RIGHTFIRE",
    12: "LEFTFIRE", 13: "DOWNFIRE",  14: "UPRIGHTFIRE",
    15: "UPLEFTFIRE", 16: "DOWNRIGHTFIRE", 17: "DOWNLEFTFIRE",
}


def _fmt_action(action: int) -> str:
    name = ACTION_NAMES.get(action, f"Acción {action}")
    return f"{action} ({name})"

def make_env(env_id, render_mode=None):
    env = gym.make(env_id, render_mode=render_mode, frameskip=1)  # ← agregar frameskip=1

    env = AtariPreprocessing(
        env,
        screen_size=84,
        grayscale_obs=True,
        frame_skip=4,
        grayscale_newaxis=False
    )

    env = FrameStackObservation(env, stack_size=4)
    return env

def _save_path(agent_type: str) -> Path:
    SAVE_DIR.mkdir(exist_ok=True)
    suffix = "dk.pt" if agent_type == "dqn" else "dk.pkl"
    return SAVE_DIR / f"{agent_type}_{suffix}"

def _load_agent(agent_type: str):
    path = _save_path(agent_type)

    if agent_type == "qlearning":
        from rl_games.agents.qlearning import QLearningAgent
        return QLearningAgent.load(path)

    from rl_games.agents.dqn import DQNAgent
    return DQNAgent.load(path)

# ── Comandos ──────────────────────────────────────────────────────────

def cmd_inspect(args: argparse.Namespace) -> None:
    env_id = getattr(args, "env", None) or ENV_ID  # Fix 2a: args.env puede no existir
    env = make_env(env_id)

    print("--- Inspeccionando Donkey Kong ---")
    print(f"Espacio de Observación: {env.observation_space.shape}")
    print(f"Número de Acciones: {env.action_space.n}")

    obs, _ = env.reset()
    print(f"Estado inicial procesado (shape): {obs.shape}")

    env.close()


def cmd_train(args: argparse.Namespace) -> None:
    path = _save_path(args.agent)

    if args.agent == "dqn":
        from rl_games.agents.dqn import DQNAgent

        agent = DQNAgent.load(path) if path.exists() else DQNAgent(ENV_ID)

        print(f"Entrenando DQN en Donkey Kong por {args.episodes} episodios...")
        agent.train(total_episodes=args.episodes)
        agent.save(path)

    else:
        # Fix 3: Q-Learning habilitado (útil con espacio de obs discretizado externamente)
        from rl_games.agents.qlearning import QLearningAgent

        agent = QLearningAgent.load(path) if path.exists() else QLearningAgent(ENV_ID)

        print(f"Entrenando Q-Learning en Donkey Kong por {args.episodes} episodios...")
        agent.train(total_episodes=args.episodes)
        agent.save(path)


def cmd_sim(args: argparse.Namespace) -> None:
    path = _save_path(args.agent)

    if not path.exists():
        print(f"No hay modelo en {path}")
        return

    agent = _load_agent(args.agent)
    env = make_env(ENV_ID)

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        step = 0

        while not done:
            step += 1
            # usar deterministic=False para ver variedad de acciones
            action, _ = agent.predict(obs, deterministic=False)  # ← cambio

            obs, reward, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
            total_reward += reward

            if args.steps is None or step <= args.steps:
                print(f"Step {step} | Action: {_fmt_action(int(action))} | Reward: {reward:.2f} | Total: {total_reward:.2f}")

        print(f"Episodio {ep} finalizado. Reward: {total_reward}\n")

    env.close()


def cmd_render(args: argparse.Namespace) -> None:
    path = _save_path(args.agent)

    if not path.exists():
        print(f"No hay modelo en {path}")
        return

    agent = _load_agent(args.agent)
    env = make_env(ENV_ID, render_mode="human")

    for _ in range(args.episodes):
        obs, _ = env.reset()
        done = False

        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated

    env.close()


# ── Parser ──────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rlgames",
        description="Atari Donkey Kong Solver"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # Fix 2b: se agrega --env opcional al subparser inspect
    inspect = sub.add_parser("inspect")
    inspect.add_argument("--env", type=str, default=None, help="ID del entorno (por defecto: ALE/DonkeyKong-v5)")
    inspect.set_defaults(func=cmd_inspect)

    train = sub.add_parser("train")
    train.add_argument("agent", choices=AGENT_CHOICES)
    train.add_argument("--episodes", type=int, default=1000)
    train.set_defaults(func=cmd_train)

    sim = sub.add_parser("sim")
    sim.add_argument("agent", choices=AGENT_CHOICES)
    sim.add_argument("--episodes", type=int, default=1)
    sim.add_argument("--steps", type=int, default=100)
    sim.set_defaults(func=cmd_sim)

    render = sub.add_parser("render")
    render.add_argument("agent", choices=AGENT_CHOICES)
    render.add_argument("--episodes", type=int, default=1)
    render.set_defaults(func=cmd_render)

    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()