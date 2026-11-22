
import numpy as np
import random

from obstacles.bed import Bed
from obstacles.table import Table
from actions.actions import Action, MOVES
from robot.robot import Robot
from room import CellType


class Environment:
    # Recompensas (se pueden ir ajustando según cómo aprenda el agente)
    STEP_PENALTY = -0.01      # cada paso cuesta un poco, para que no pierda tiempo
    BUMP_PENALTY = -0.5       # chocar contra una pared o un mueble
    CLEAN_REWARD = 1          # limpiar una celda sucia
    CLEAN_PENALTY = -0.5      # limpiar una celda limpia
    ALL_CLEAN_REWARD = 10     # dejar toda la habitación limpia
    NO_BATTERY_PENALTY = -10  # quedarse sin batería
    CONT_NO_CLEAN_PENALTY = 0  # castigo por no limpiar nada en un periodo largo de tiempo (para que no se quede dando vueltas)
    CONT_NO_CLEAN_MIN_STEPS = 10  # pasos mínimos para empezar a aplicar el castigo por no limpiar nada

    def __init__(self, *, max_steps=200, room_generator):

        self.max_steps = max_steps
        self.room_generator = room_generator
        self.size = room_generator.size
        self.dirt_count = room_generator.dirt_count

        # Creamos la primera habitación
        self.reset()

    # ------------------------------------------------------------------
    # Interfaz para el agente de RL
    # ------------------------------------------------------------------

    def reset(self):
        """Empieza un episodio nuevo: habitación nueva y robot en el inicio."""

        self.robot = Robot(self.room_generator.start_position)
        self.steps = 0
        self.generate_room()
        self.no_clean_steps = 0

        return self.get_observation()
    
    def reset_to_cache(self):
        """Reinicia la habitación a su estado inicial (para repasar la memoria)."""

        self.robot = Robot(self.room_generator.start_position)
        self.steps = 0
        self.room = self.room_cache.copy()
        self.no_clean_steps = 0

        return self.get_observation()

    def step(self, action):
        """Ejecuta una acción y devuelve (observación, recompensa, terminado)."""
        cleaned = False

        action = Action(action)
        reward = self.STEP_PENALTY
        self.steps += 1

        if action in MOVES:
            dx, dy = MOVES[action]
            x, y = self.robot.position
            nx, ny = x + dx, y + dy

            if self.is_free(nx, ny):
                self.robot.move((nx, ny))
            else:
                # Chocó: se queda donde está, pero gasta batería igual
                reward += self.BUMP_PENALTY
                self.robot.spend_battery()

        elif action == Action.CLEAN:
            self.robot.clean()

            if self.room[self.robot.position] == int(CellType.DIRT):
                self.room[self.robot.position] = int(CellType.EMPTY)
                reward += self.CLEAN_REWARD
                cleaned = True
            else:
                reward += self.CLEAN_PENALTY

        # Si está en el cargador, recarga
        if self.robot.position == self.room_generator.charger_position:
            self.robot.recharge()

        done = False

        if not (self.room == int(CellType.DIRT)).any():
            reward += self.ALL_CLEAN_REWARD
            done = True
        elif self.robot.battery <= 0:
            reward += self.NO_BATTERY_PENALTY
            done = True
        elif self.steps >= self.max_steps:
            # Cortamos el episodio para que no dure para siempre
            done = True
        
        if not cleaned:
            self.no_clean_steps += 1
            if self.no_clean_steps >= self.CONT_NO_CLEAN_MIN_STEPS:
                reward += self.CONT_NO_CLEAN_PENALTY * (self.no_clean_steps - self.CONT_NO_CLEAN_MIN_STEPS)

        return self.get_observation(), reward, done

    def get_observation(self):
        """Lo que el agente "ve" para decidir qué hacer.

        Devolvemos una copia de la habitación: si devolviéramos room,
        la observación guardada cambiaría cuando el robot limpia después.
        """

        return (
            self.robot.position,
            self.robot.battery,
            self.room.copy(),
        )

    def is_free(self, x, y):
        """¿El robot se puede mover a (x, y)?"""

        return (
            0 <= x < self.size
            and 0 <= y < self.size
            and self.room[x, y] != int(CellType.OBSTACLE)
        )

    # ------------------------------------------------------------------
    # Generación de la habitación
    # ------------------------------------------------------------------

    def generate_room(self, max_room_attempts=100):

        room, obstacles = self.room_generator.generate_room()
        self.room = room
        self.obstacles = obstacles

        # Guardamos una copia de la habitación para poder reiniciarla después
        self.room_cache = room.copy()

    # ------------------------------------------------------------------
    # Visualización
    # ------------------------------------------------------------------

    def display(self):

        print("\nHabitación:")

        for x in range(self.size):

            row = []

            for y in range(self.size):

                if (x, y) == self.robot.position:
                    row.append(" R ")
                elif self.room[x, y] == int(CellType.EMPTY):
                    row.append(" . ")
                elif self.room[x, y] == int(CellType.DIRT):
                    row.append(" * ")
                elif self.room[x, y] == int(CellType.OBSTACLE):
                    row.append(" # ")
                elif self.room[x, y] == int(CellType.CHARGER):
                    row.append(" C ")

            print("".join(row))

        print(f"\nBatería: {self.robot.battery}%")


# Prueba rápida: un episodio con acciones al azar
if __name__ == "__main__":

    env = Environment()
    env.display()

    done = False
    total_reward = 0

    while not done:
        print()
        input("Press Enter to continue...")
        action = random.choice(list(Action))
        observation, reward, done = env.step(action)
        total_reward += reward
        env.display()

    print(f"Pasos: {env.steps} | Recompensa total: {total_reward:.2f}")
