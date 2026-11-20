"""Entrena el robot limpiador con DQN.

Uso (desde la carpeta del proyecto):
    python . --episodes 600            # entrena y guarda el modelo
    python . --watch                   # mira jugar al modelo guardado
"""

import argparse

import numpy as np

from agent import DQNModel
from enviroment import Environment
from actions.actions import Action

MODEL_PATH = "dqn_robot.pt"


def train_rl_agent(agent, env, n_episodes, log_every=20):
    history = []

    for episode in range(1, n_episodes + 1):
        obs = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = agent.decide_action(obs)
            next_obs, reward, done = env.step(action)

            # Guardamos (s, a, r, s', done) y entrenamos con Bellman
            agent.receive_reward(obs, action, reward, next_obs, done)

            obs = next_obs
            total_reward += reward

        eps = agent.epsilon
        agent.end_episode()

        dirt_left = int((env.room == env.DIRT).sum())
        history.append((total_reward, env.dirt_count - dirt_left, env.steps))

        if episode % log_every == 0:
            recent = np.array(history[-log_every:])
            print(
                f"Episodio {episode:4d} | "
                f"recompensa media {recent[:, 0].mean():7.2f} | "
                f"limpió {recent[:, 1].mean():4.1f}/{env.dirt_count} | "
                f"pasos {recent[:, 2].mean():5.1f} | "
                f"epsilon {eps:.3f}"
            )

    return history


def watch_agent(agent, env):
    """Corre un episodio sin exploración y lo muestra paso a paso."""

    obs = env.reset()
    env.display()
    total_reward = 0.0
    done = False

    while not done:
        input("Enter para el siguiente paso...")
        action_id = agent.decide_action(obs, greedy=True)
        print(f"\nAcción elegida: {Action(action_id).name}")
        obs, reward, done = env.step(action_id)
        total_reward += reward
        env.display()

    print(f"Pasos: {env.steps} | Recompensa total: {total_reward:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--watch", action="store_true",
                        help="cargar el modelo guardado y mostrar un episodio")
    args = parser.parse_args()

    env = Environment()
    agent = DQNModel(room_size=env.size)

    if args.watch:
        agent.load(MODEL_PATH)
        watch_agent(agent, env)
    else:
        train_rl_agent(agent, env, args.episodes)
        agent.save(MODEL_PATH)
        print(f"Modelo guardado en {MODEL_PATH}")
