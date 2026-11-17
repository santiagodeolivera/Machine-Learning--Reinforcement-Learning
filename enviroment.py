
import numpy as np
import random
from collections import deque

from obstacles.bed import Bed
from obstacles.table import Table
from actions.actions import Action, MOVES
from robot.robot import Robot


class Environment:

    EMPTY = 0
    DIRT = 1
    OBSTACLE = 2
    CHARGER = 3
    OBSTACLE_TYPES = [Bed, Table]

    # Recompensas (se pueden ir ajustando según cómo aprenda el agente)
    STEP_PENALTY = -0.01      # cada paso cuesta un poco, para que no pierda tiempo
    BUMP_PENALTY = -0.5       # chocar contra una pared o un mueble
    CLEAN_REWARD = 1          # limpiar una celda sucia
    ALL_CLEAN_REWARD = 10     # dejar toda la habitación limpia
    NO_BATTERY_PENALTY = -10  # quedarse sin batería

    def __init__(self, size=8, dirt_count=8, obstacle_count=3, max_steps=200):

        self.size = size
        self.dirt_count = dirt_count
        self.obstacle_count = obstacle_count
        self.max_steps = max_steps

        # Posiciones fijas
        self.start_position = (0, 0)
        self.charger_position = (size - 1, size - 1)

        # Creamos la primera habitación
        self.reset()

    # ------------------------------------------------------------------
    # Interfaz para el agente de RL
    # ------------------------------------------------------------------

    def reset(self):
        """Empieza un episodio nuevo: habitación nueva y robot en el inicio."""

        self.robot = Robot(self.start_position)
        self.steps = 0
        self.generate_room()

        return self.get_observation()

    def step(self, action):
        """Ejecuta una acción y devuelve (observación, recompensa, terminado)."""

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

            if self.room[self.robot.position] == self.DIRT:
                self.room[self.robot.position] = self.EMPTY
                reward += self.CLEAN_REWARD

        # Si está en el cargador, recarga
        if self.robot.position == self.charger_position:
            self.robot.recharge()

        done = False

        if not (self.room == self.DIRT).any():
            reward += self.ALL_CLEAN_REWARD
            done = True
        elif self.robot.battery <= 0:
            reward += self.NO_BATTERY_PENALTY
            done = True
        elif self.steps >= self.max_steps:
            # Cortamos el episodio para que no dure para siempre
            done = True

        return self.get_observation(), reward, done

    def get_observation(self):
        """Lo que el agente "ve" para decidir qué hacer.

        Es una tupla para que se pueda usar como clave en la tabla de Q-learning.
        """

        return (
            self.robot.position,
            self.robot.battery,
            tuple(self.room.flatten()),
        )

    def is_free(self, x, y):
        """¿El robot se puede mover a (x, y)?"""

        return (
            0 <= x < self.size
            and 0 <= y < self.size
            and self.room[x, y] != self.OBSTACLE
        )

    # ------------------------------------------------------------------
    # Generación de la habitación
    # ------------------------------------------------------------------

    def generate_room(self, max_room_attempts=100):

        for _ in range(max_room_attempts):

            # Reiniciamos la habitación
            self.room = np.zeros((self.size, self.size), dtype=int)

            # Volvemos a colocar el cargador
            self.room[self.charger_position] = self.CHARGER

            # Elegimos aleatoriamente los muebles
            obstacles = self.generate_obstacles()

            # Intentamos ubicarlos y comprobamos accesibilidad
            if self.place_obstacles(obstacles) and self.all_cells_reachable():
                break

        else:
            raise RuntimeError("No se pudo generar una habitación válida")

        # Una vez colocados los obstáculos, generamos la suciedad
        self.place_dirt()

    def generate_obstacles(self):

        obstacles = []

        # Elegimos los muebles aleatoriamente
        for _ in range(self.obstacle_count):
            obstacle_type = random.choice(self.OBSTACLE_TYPES)
            obstacle = obstacle_type()
            obstacles.append(obstacle)

        return obstacles

    # Ubico los obstáculos dentro de la grilla
    def place_obstacles(self, obstacles, max_attempts=100):

        self.obstacles = []

        for obstacle in obstacles:

            placed = False

            for _ in range(max_attempts):

                x = random.randint(0, self.size - obstacle.height)
                y = random.randint(0, self.size - obstacle.width)

                cells = obstacle.get_occupied_positions(x, y)

                # Comprobamos que todas las casillas estén libres
                if all(
                    self.room[c] == self.EMPTY
                    and c != self.start_position
                    for c in cells
                ):

                    # Ocupamos las casillas
                    for c in cells:
                        self.room[c] = self.OBSTACLE

                    self.obstacles.append((obstacle, (x, y)))
                    placed = True
                    break

            if not placed:
                return False

        return True

    def all_cells_reachable(self):

        visited = {self.start_position}
        queue = deque([self.start_position])

        while queue:

            x, y = queue.popleft()

            for dx, dy in MOVES.values():

                nx = x + dx
                ny = y + dy

                if self.is_free(nx, ny) and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))

        free_cells = np.sum(self.room != self.OBSTACLE)

        return len(visited) == free_cells

    def place_dirt(self):

        # Celdas libres que quedaron después de poner los muebles
        available_positions = [
            (x, y)
            for x in range(self.size)
            for y in range(self.size)
            if self.room[x, y] == self.EMPTY
            and (x, y) != self.start_position
        ]

        # Si no hay lugar para toda la suciedad, ponemos la que entre
        dirt = random.sample(
            available_positions,
            min(self.dirt_count, len(available_positions))
        )

        for position in dirt:
            self.room[position] = self.DIRT

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
                elif self.room[x, y] == self.EMPTY:
                    row.append(" . ")
                elif self.room[x, y] == self.DIRT:
                    row.append(" * ")
                elif self.room[x, y] == self.OBSTACLE:
                    row.append(" # ")
                elif self.room[x, y] == self.CHARGER:
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
        action = random.choice(list(Action))
        observation, reward, done = env.step(action)
        total_reward += reward

    env.display()
    print(f"Pasos: {env.steps} | Recompensa total: {total_reward:.2f}")
