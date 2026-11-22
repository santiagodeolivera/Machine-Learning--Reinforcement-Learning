"""Agente Deep Q-Learning (DQN) para el robot limpiador.

Basado en: https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

Idea general:
  - Una red neuronal (MLPModel: CNN + densa) estima Q(s, a): cuánta recompensa
    futura esperamos si en el estado s hacemos la acción a.
  - DQNModel tiene dos copias de esa red:
        policy_nn: la que entrenamos y usamos para elegir acciones.
        target_nn: copia que se actualiza cada `update_steps` pasos y se usa
                   solo para calcular el Q objetivo (así el objetivo no se
                   mueve en cada paso y el entrenamiento es estable).
  - El robot elige acciones con epsilon-greedy (a veces explora al azar).
  - Cada transición (s, a, r, s', done) se guarda en una memoria (replay buffer).
  - Para entrenar usamos la ecuación de Bellman:
        Q_objetivo = r + gamma * max_a' Q_target(s', a')    (si no terminó)
        Q_objetivo = r                                     (si terminó)
    y minimizamos la diferencia entre Q_policy(s, a) y Q_objetivo.
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from actions.actions import Action
from robot.robot import Robot


seed = 42

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

N_ACTIONS = len(Action)

# Canales de la "imagen" que ve la CNN:
#   0: vacío, 1: suciedad, 2: obstáculo, 3: cargador (one-hot de la grilla)
#   4: posición del robot
N_CHANNELS = 5

# Datos extra que van directo a la red densa: batería, fila y columna del robot
N_EXTRA = 3

# Determina si el modelo toma en cuenta una sección de alta calidad alrededor del robot,
# como una vista a corta distancia.
INCLUDE_HQ_OBSERVATION = True

# Tamaño de la sección de alta calidad (1 -> 3x3, 2 -> 5x5, etc.)
HQ_OBSERVATION_SIZE = 2


if INCLUDE_HQ_OBSERVATION:
    EXTRA_SIZE = N_EXTRA + N_CHANNELS * (HQ_OBSERVATION_SIZE * 2 + 1) ** 2
else:
    EXTRA_SIZE = N_EXTRA

def encode_observation(obs):
    """Convierte la observación del entorno en arrays para la red.

    obs = (posición, batería, habitación) -> (grid [N_CHANNELS, N, N], extra [EXTRA_SIZE])
    """

    position, battery, room = obs
    size = room.shape[0]

    grid = np.zeros((N_CHANNELS, size, size), dtype=np.float32)
    for cell_type in range(4):
        grid[cell_type] = room == cell_type
    grid[4][position] = 1.0

    if INCLUDE_HQ_OBSERVATION:
        hq_grid = np.zeros((N_CHANNELS, HQ_OBSERVATION_SIZE * 2 + 1, HQ_OBSERVATION_SIZE * 2 + 1), dtype=np.float32)
        hq_grid[2] = 1.0  # Asumimos que los bordes son obstáculos
        # TODO: Confirmar si está bien que la sección de alta calidad sea 3x3
        # TODO: Confirmar si está bien que los bordes sean obstáculos

        min_x = max(0, position[0] - HQ_OBSERVATION_SIZE)
        max_x = min(size, position[0] + HQ_OBSERVATION_SIZE + 1)
        min_y = max(0, position[1] - HQ_OBSERVATION_SIZE)
        max_y = min(size, position[1] + HQ_OBSERVATION_SIZE + 1)
        hq_grid_data = grid[:, min_x:max_x, min_y:max_y]
        
        # Posición del robot dentro de la sección de alta calidad
        relative_position = divmod(hq_grid_data[4].argmax(), hq_grid_data.shape[2])

        min_x = HQ_OBSERVATION_SIZE - relative_position[0]
        min_y = HQ_OBSERVATION_SIZE - relative_position[1]
        max_x = min_x + hq_grid_data.shape[1]
        max_y = min_y + hq_grid_data.shape[2]
        hq_grid[:, min_x:max_x, min_y:max_y] = hq_grid_data

    extra = np.array(
        [
            *( hq_grid.flatten() if INCLUDE_HQ_OBSERVATION else [] ),
            battery / Robot.MAX_BATTERY,
            position[0] / (size - 1),
            position[1] / (size - 1),
        ],
        dtype=np.float32,
    )

    return grid, extra


class MLPModel(nn.Module):
    """Red Q: CNN para la habitación + red densa que da un valor por acción."""

    def __init__(self, room_size, n_actions=N_ACTIONS):
        super().__init__()

        # padding=1 mantiene el tamaño de la grilla (N x N) en cada capa
        self.cnn = nn.Sequential(
            nn.Conv2d(N_CHANNELS, 32, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        cnn_output_size = 64 * (room_size - 2)**2

        self.dense = nn.Sequential(
            nn.Linear(cnn_output_size + EXTRA_SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, grid, extra):
        # grid: [batch, N_CHANNELS, N, N]   extra: [batch, EXTRA_SIZE]   ->   [batch, N_ACTIONS]

        features = self.cnn(grid)

        return self.dense(torch.cat((features, extra), dim=1))


class ReplayBuffer:
    """Memoria de transiciones pasadas para entrenar con muestras al azar."""

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, grid, extra, action, reward, next_grid, next_extra, done):
        self.memory.append((grid, extra, action, reward, next_grid, next_extra, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        grid, extra, action, reward, next_grid, next_extra, done = zip(*batch)

        def to_tensor(values, dtype=torch.float32):
            return torch.as_tensor(np.array(values), dtype=dtype, device=device)

        return (
            to_tensor(grid),
            to_tensor(extra),
            to_tensor(action, torch.int64),
            to_tensor(reward),
            to_tensor(next_grid),
            to_tensor(next_extra),
            to_tensor(done),
        )

    def __len__(self):
        return len(self.memory)


class DQNModel:

    def __init__(
        self,
        room_size,
        discount_factor=0.99,
        learning_rate=1e-4,
        batch_size=64,
        buffer_capacity=50_000,
        min_buffer_size=1_000,
        update_steps=1_000,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
    ):
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.update_steps = update_steps

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Red principal (la que entrenamos y usamos para actuar)
        self.policy_nn = MLPModel(room_size).to(device)

        # Red objetivo: copia de la principal, solo se usa para calcular Q_objetivo
        self.target_nn = MLPModel(room_size).to(device)
        self.target_nn.load_state_dict(self.policy_nn.state_dict())
        self.target_nn.eval()

        self.optimizer = optim.Adam(self.policy_nn.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()  # Huber: más estable que MSE en DQN
        self.memory = ReplayBuffer(buffer_capacity)
        self.learn_steps = 0

    # ------------------------------------------------------------------
    # Actuar
    # ------------------------------------------------------------------

    def decide_action(self, obs, greedy=False):
        """Epsilon-greedy: con prob. epsilon explora, si no elige argmax Q."""

        if not greedy and random.random() < self.epsilon:
            return random.randrange(N_ACTIONS)

        grid, extra = encode_observation(obs)
        with torch.no_grad():
            q_values = self.policy_nn(
                torch.as_tensor(grid, device=device).unsqueeze(0),
                torch.as_tensor(extra, device=device).unsqueeze(0),
            )
        return int(q_values.argmax(dim=1).item())

    # ------------------------------------------------------------------
    # Aprender
    # ------------------------------------------------------------------

    def receive_reward(self, obs, action, reward, next_obs, done):
        """Guarda la transición y hace un paso de entrenamiento."""

        grid, extra = encode_observation(obs)
        next_grid, next_extra = encode_observation(next_obs)
        self.memory.push(grid, extra, action, reward, next_grid, next_extra, float(done))

        return self.learn()

    def learn(self):
        if len(self.memory) < self.min_buffer_size:
            return None

        grid, extra, action, reward, next_grid, next_extra, done = \
            self.memory.sample(self.batch_size)

        # Q(s, a) que predice la red principal para la acción que se tomó
        predicted_q = self.policy_nn(grid, extra).gather(1, action.unsqueeze(1)).squeeze(1)

        # Ecuación de Bellman con la red objetivo
        with torch.no_grad():
            next_q = self.target_nn(next_grid, next_extra).max(dim=1).values
            target_q = reward + self.discount_factor * next_q * (1 - done)

        loss = self.loss_fn(predicted_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_nn.parameters(), 10)
        self.optimizer.step()

        # Cada `update_steps` pasos copiamos los pesos de policy_nn a target_nn
        self.learn_steps += 1
        if self.learn_steps % self.update_steps == 0:
            self.target_nn.load_state_dict(self.policy_nn.state_dict())

        return loss.item()

    def end_episode(self):
        """Al terminar cada episodio exploramos un poco menos."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Guardar / cargar
    # ------------------------------------------------------------------

    def save(self, path):
        torch.save(self.policy_nn.state_dict(), path)

    def load(self, path):
        state = torch.load(path, map_location=device)
        self.policy_nn.load_state_dict(state)
        self.target_nn.load_state_dict(state)
