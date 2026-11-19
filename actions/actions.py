from enum import IntEnum


# Las acciones van numeradas desde 0 porque el agente de RL
# elige acciones como números enteros (0, 1, 2, 3, 4)
class Action(IntEnum):
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    CLEAN = 4


# Cuánto se mueve el robot (fila, columna) con cada acción de movimiento
MOVES = {
    Action.MOVE_UP: (-1, 0),
    Action.MOVE_DOWN: (1, 0),
    Action.MOVE_LEFT: (0, -1),
    Action.MOVE_RIGHT: (0, 1),
}
