import argparse
from pathlib import Path
from importlib.metadata import version

import ale_py
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

gym.register_envs(ale_py)

# ── Configuración Principal ──────────────────────────────────────────
ENV_ID = "ALE/SpaceInvaders-v5"
SAVE_DIR = Path("saves")
AGENT_CHOICES = ("dqn",)
VERSION = version("rl_games")

# Acciones de Space Invaders
ACTION_NAMES = {
    0: "NOOP",
    1: "FIRE",
    2: "RIGHT",
    3: "LEFT",
    4: "RIGHTFIRE",
    5: "LEFTFIRE",
}


def _fmt_action(action: int) -> str:
    name = ACTION_NAMES.get(action, f"Acción {action}")
    return f"{action} ({name})"


def make_env(env_id: str, render_mode=None):
    env = gym.make(
        env_id, render_mode=render_mode, frameskip=1
    )  # ← frameskip=1 está bien
    env = AtariPreprocessing(
        env,
        screen_size=84,
        grayscale_obs=True,
        frame_skip=4,  # ← AtariPreprocessing maneja el frameskip
        grayscale_newaxis=False,
    )
    env = FrameStackObservation(env, stack_size=4)
    return env


def _save_path(agent_type: str) -> Path:
    SAVE_DIR.mkdir(exist_ok=True)
    return SAVE_DIR / f"{agent_type}_si_best.pt"


def _load_agent(agent_type: str):
    path = _save_path(agent_type)
    from rl_games.agents.dqn import DQNAgent

    return DQNAgent.load(path)


# ── Comandos ──────────────────────────────────────────────────────────


def cmd_inspect(args: argparse.Namespace) -> None:
    env_id = getattr(args, "env", None) or ENV_ID
    env = make_env(env_id)

    print("--- Inspeccionando Space Invaders ---")
    print(f"Entorno: {env_id}")
    print(f"Espacio de Observación: {env.observation_space.shape}")
    print(f"Número de Acciones: {env.action_space.n}")
    print("Acciones disponibles:")
    for k, v in ACTION_NAMES.items():
        print(f"  {k}: {v}")

    obs, _ = env.reset()
    print(f"\nEstado inicial procesado (shape): {obs.shape}")
    env.close()


def cmd_train(args: argparse.Namespace) -> None:
    path = _save_path(args.agent)
    from rl_games.agents.dqn import DQNAgent

    agent = DQNAgent.load(path) if path.exists() else DQNAgent(ENV_ID)
    print(f"Entrenando DQN en Space Invaders por {args.episodes} episodios...")
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
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
            total_reward += reward

            if args.steps is None or step <= args.steps:
                print(
                    f"Step {step:3d} | "
                    f"Action: {_fmt_action(int(action))} | "
                    f"Reward: {reward:.2f} | "
                    f"Total: {total_reward:.2f}"
                )

        print(f"Episodio {ep} finalizado. Reward total: {total_reward}\n")

    env.close()


def cmd_render(args: argparse.Namespace) -> None:
    path = _save_path(args.agent)

    if not path.exists():
        print(f"No hay modelo en {path}")
        return

    agent = _load_agent(args.agent)
    env = make_env(ENV_ID, render_mode="human")

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
            total_reward += reward

        print(f"Episodio {ep + 1} finalizado. Reward total: {total_reward}")

    env.close()


def cmd_list(args: argparse.Namespace) -> None:
    print(f"Entorno activo: {ENV_ID}")
    for agent in AGENT_CHOICES:
        path = _save_path(agent)
        status = f"guardado en {path}" if path.exists() else "sin guardar"
        print(f"  {agent}: {status}")


def cmd_delete(args: argparse.Namespace) -> None:
    path = _save_path(args.agent)
    if path.exists():
        path.unlink()
        print(f"Modelo {path} eliminado.")
    else:
        print(f"No hay modelo en {path}")


def cmd_version(args: argparse.Namespace) -> None:
    print(f"rl_games v{VERSION}")


# ── Parser ──────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rlgames", description="Space Invaders DQN Solver"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # version
    ver = sub.add_parser("version")
    ver.set_defaults(func=cmd_version)

    # list
    lst = sub.add_parser("list")
    lst.set_defaults(func=cmd_list)

    # inspect
    inspect = sub.add_parser("inspect")
    inspect.add_argument("--env", type=str, default=None)
    inspect.set_defaults(func=cmd_inspect)

    # train
    train = sub.add_parser("train")
    train.add_argument("agent", choices=AGENT_CHOICES)
    train.add_argument("--episodes", type=int, default=3000)
    train.set_defaults(func=cmd_train)

    # sim
    sim = sub.add_parser("sim")
    sim.add_argument("agent", choices=AGENT_CHOICES)
    sim.add_argument("--episodes", type=int, default=1)
    sim.add_argument("--steps", type=int, default=None)
    sim.set_defaults(func=cmd_sim)

    # render
    render = sub.add_parser("render")
    render.add_argument("agent", choices=AGENT_CHOICES)
    render.add_argument("--episodes", type=int, default=1)
    render.set_defaults(func=cmd_render)

    # delete
    delete = sub.add_parser("delete")
    delete.add_argument("agent", choices=AGENT_CHOICES)
    delete.set_defaults(func=cmd_delete)

    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
