import random
from collections import deque

import numpy as np

from actions.actions import MOVES
from obstacles.bed import Bed
from obstacles.table import Table

class CellType:
    EMPTY = 0
    DIRT = 1
    OBSTACLE = 2
    CHARGER = 3

OBSTACLE_TYPES = [Bed, Table]

class FixedRoomGenerator:
    def __init__(self, size, src, obstacles_src, start_position=(0, 0), charger_position=None):
        self.size = size
        self.dirt_count = src.count("*")

        self.src = src
        self.obstacles_src = obstacles_src

        self.start_position = start_position
        self.charger_position = charger_position if charger_position is not None else (size - 1, size - 1)

    def generate_room(self):
        room = np.full((self.size, self.size), CellType.EMPTY, dtype=int)

        for i, row in enumerate(self.src.split("/")):
            for j, cell in enumerate(row):
                if cell == "#":
                    room[i, j] = CellType.OBSTACLE
                elif cell == "*":
                    room[i, j] = CellType.DIRT
                elif cell == "C":
                    room[i, j] = CellType.CHARGER
                elif cell == ".":
                    room[i, j] = CellType.EMPTY
                else:
                    raise ValueError(f"Caracter inválido en la habitación: {repr(cell)}")
                
        
        obstacles = []
        if self.obstacles_src != "X":
            for ob_src in self.obstacles_src.split("/"):
                ob_type, i_str, j_str = ob_src.split(",")
                i, j = int(i_str), int(j_str)
                match ob_type:
                    case "Bed":
                        obstacles.append((Bed(), (i, j)))
                    case "Table":
                        obstacles.append((Table(), (i, j)))
                    case _:
                        raise ValueError(f"Tipo de obstáculo inválido: {repr(ob_type)}")

        return room, obstacles


class RandomRoomGenerator:
    def __init__(self, size, max_room_attempts=100, max_place_obstacles_attempts=100, obstacle_count=5, dirt_count=10, start_position=None, charger_position=None, seed=42):
        self.size = size
        self.dirt_count = dirt_count

        self.max_room_attempts = max_room_attempts
        self.max_place_obstacles_attempts = max_place_obstacles_attempts
        self.obstacle_count = obstacle_count

        self.start_position = start_position if start_position is not None else (0, 0)
        self.charger_position = charger_position if charger_position is not None else (size-1, size-1)

        self.seed = seed

    def generate_room(self):
        random_prev_state = random.getstate()
        random.seed(self.seed)

        for _ in range(self.max_room_attempts):

            # Reiniciamos la habitación
            room = np.full((self.size, self.size), CellType.EMPTY, dtype=int)

            # Volvemos a colocar el cargador
            room[self.charger_position] = CellType.CHARGER

            # Elegimos aleatoriamente los muebles
            obstacles = self.generate_obstacles()
            placed_obstacles = self.place_obstacles(room, obstacles)

            # Intentamos ubicarlos y comprobamos accesibilidad
            if placed_obstacles is not None and self.all_cells_reachable(room):
                break

        else:
            raise RuntimeError("No se pudo generar una habitación válida")
        
        # Una vez colocados los obstáculos, generamos la suciedad
        self.place_dirt(room)

        random.setstate(random_prev_state)
        return room, placed_obstacles

    def generate_obstacles(self):
        obstacles = []

        # Elegimos los muebles aleatoriamente
        for _ in range(self.obstacle_count):
            obstacle_type = random.choice(OBSTACLE_TYPES)
            obstacle = obstacle_type()
            obstacles.append(obstacle)

        return obstacles

    # Ubico los obstáculos dentro de la grilla
    def place_obstacles(self, room, obstacles):

        result_obstacles = []

        for obstacle in obstacles:

            placed = False

            for _ in range(self.max_place_obstacles_attempts):

                x = random.randint(0, self.size - obstacle.height)
                y = random.randint(0, self.size - obstacle.width)

                cells = obstacle.get_occupied_positions(x, y)

                # Comprobamos que todas las casillas estén libres
                if all(
                    room[c] == CellType.EMPTY
                    and c != self.start_position
                    for c in cells
                ):

                    # Ocupamos las casillas
                    for c in cells:
                        room[c] = CellType.OBSTACLE

                    result_obstacles.append((obstacle, (x, y)))
                    placed = True
                    break

            if not placed:
                return None

        return result_obstacles

    def all_cells_reachable(self, room):

        visited = {self.start_position}
        queue = deque([self.start_position])

        while queue:

            x, y = queue.popleft()

            for dx, dy in MOVES.values():

                nx = x + dx
                ny = y + dy

                if self.is_free(room, nx, ny) and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))

        free_cells = np.sum(room != CellType.OBSTACLE)

        return len(visited) == free_cells

    def place_dirt(self, room):

        # Celdas libres que quedaron después de poner los muebles
        available_positions = [
            (x, y)
            for x in range(self.size)
            for y in range(self.size)
            if room[x, y] == CellType.EMPTY
            and (x, y) != self.start_position
        ]

        # Si no hay lugar para toda la suciedad, ponemos la que entre
        dirt = random.sample(
            available_positions,
            min(self.dirt_count, len(available_positions))
        )

        for position in dirt:
            room[position] = CellType.DIRT
    
    def is_free(self, room, x, y):
        return (
            0 <= x < self.size
            and 0 <= y < self.size
            and room[x, y] != CellType.OBSTACLE
        )

def get_room_generator(args):
    if args.room_src and args.obstacles_src:
        return FixedRoomGenerator(
            size=8,
            src=args.room_src,
            obstacles_src=args.obstacles_src,
            start_position=(0, 0),
            charger_position=(7, 7)
        )
    else:
        return RandomRoomGenerator(
            size=8,
            max_room_attempts=100,
            max_place_obstacles_attempts=100,
            obstacle_count=3,
            dirt_count=8,
            start_position=(0, 0),
            charger_position=(7, 7),
            seed=42
        )
