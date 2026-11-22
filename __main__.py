"""Entrena el robot limpiador con DQN.

Uso (desde la carpeta del proyecto):
    python . --rooms 2 --episodes-per-room 20 # entrena y guarda el modelo
    python . --watch                          # mira jugar al modelo guardado
"""

import argparse

import numpy as np

from agent import DQNModel
from environment import Environment
from actions.actions import Action
from room import get_room_generator, CellType

MODEL_PATH = "dqn_robot.pt"


def train_rl_agent(agent, env, *, n_rooms, n_episodes_per_room, log_every=20, review_times=0):
    history = []

    episode = 1
    for _ in range(n_rooms):
        obs = env.reset()
        for ep in range(n_episodes_per_room):
            if ep != 0:
                obs = env.reset_to_cache()
            
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

                dirt_left = int((env.room == int(CellType.DIRT)).sum())
                history.append((total_reward, env.dirt_count - dirt_left, env.steps))
            
            for _ in range(review_times):
                agent.learn()

            if episode % log_every == 0:
                recent = np.array(history[-log_every:])
                print(
                    f"Episodio {episode:4d} | "
                    f"recompensa media {recent[:, 0].mean():7.2f} | "
                    f"limpió {recent[:, 1].mean():4.1f}/{env.dirt_count} | "
                    f"pasos {recent[:, 2].mean():5.1f} | "
                    f"epsilon {eps:.3f}"
                )

            episode += 1

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
    parser.add_argument("--room-src", type=str, default=None,
                        help="fuente de la habitación (ejemplo: '.../.../...')")
    parser.add_argument("--obstacles-src", type=str, default=None,
                        help="fuente de los obstáculos (ejemplo: 'Bed,1,2/Table,3,4')")
    parser.add_argument("--rooms", type=int, default=2)
    parser.add_argument("--episodes-per-room", type=int, default=20)
    parser.add_argument("--watch", action="store_true",
                        help="cargar el modelo guardado y mostrar un episodio")
    parser.add_argument("--review-times", type=int, default=0,
                        help="número de veces que el agente repasa la memoria al final de cada episodio")
    args = parser.parse_args()

    room_generator = get_room_generator(args)

    env = Environment(room_generator=room_generator)
    agent = DQNModel(room_size=env.size)

    if args.watch:
        agent.load(MODEL_PATH)
        watch_agent(agent, env)
    else:
        train_rl_agent(
            agent, env,
            n_rooms=args.rooms,
            n_episodes_per_room=args.episodes_per_room,
            review_times=args.review_times
        )
        agent.save(MODEL_PATH)
        print(f"Modelo guardado en {MODEL_PATH}")
