
import numpy as np
import random


class Environment:

    EMPTY = 0
    DIRT = 1
    OBSTACLE = 2
    CHARGER = 3

    ACTIONS = {
        0: (-1, 0),  # Arriba
        1: (1, 0),   # Abajo
        2: (0, -1),  # Izquierda
        3: (0, 1)    # Derecha
    }

    def __init__(self, size=6, dirt_count=8, obstacle_count=5):

        self.size = size
        self.dirt_count = dirt_count
        self.obstacle_count = obstacle_count

        # Creamos una habitación vacía
        self.room = np.full((size, size), self.EMPTY, dtype=int)

        # Posición fija del cargador
        self.charger_position = (size - 1, size - 1)

        # Colocamos el cargador
        self.room[self.charger_position] = self.CHARGER

        # Posición del robot
        self.robot_position = (0, 0)

        # Batería inicial
        self.battery = 100

        # Generamos obstáculos y suciedad
        self.generate_room()

    def generate_room(self):

        available_positions = [
            (x, y)
            for x in range(self.size)
            for y in range(self.size)
            if (x, y) != self.charger_position
            and (x, y) != self.robot_position
        ]

        # Elegimos posiciones aleatorias para los obstáculos
        obstacles = random.sample(
            available_positions,
            self.obstacle_count
        )

        for position in obstacles:
            self.room[position] = self.OBSTACLE

        # Quitamos las posiciones ocupadas
        available_positions = [
            position for position in available_positions
            if self.room[position] == self.EMPTY
        ]

        # Elegimos posiciones aleatorias para la suciedad
        dirt = random.sample(
            available_positions,
            self.dirt_count
        )

        for position in dirt:
            self.room[position] = self.DIRT

    def display(self):

        print("\nHabitación:")

        for x in range(self.size):

            row = []

            for y in range(self.size):

                if (x, y) == self.robot_position:
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

        print(f"\nBatería: {self.battery}%")
