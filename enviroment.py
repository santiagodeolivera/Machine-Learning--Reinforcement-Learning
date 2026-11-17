
import numpy as np
import random
from obstacles.bed import Bed
from obstacles.table import Table


class Environment:

    EMPTY = 0
    DIRT = 1
    OBSTACLE = 2
    CHARGER = 3
    OBSTACLE_TYPES = [Bed, Table]

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
        self.room = np.zeros((size, size), dtype=int)

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

    def generate_obstacles(self):

        obstacles = []

        # Elegimos los muebles aleatoriamente
        for _ in range(self.obstacle_count):
            obstacle_type = random.choice(self.OBSTACLE_TYPES)
            obstacle = obstacle_type()
            obstacles.append(obstacle)

        return obstacles

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
                    and c != self.robot_position
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
