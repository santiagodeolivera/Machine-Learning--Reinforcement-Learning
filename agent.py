import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# TODO: Determine whether source URL must be included
# https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
torch.set_default_device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

seed = 42

# TODO: Determine what to do with this
# Deep Q Learning Agent
class DQNAgent:
    def __init__(self, room_size, middle_layers):
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 256, (3, 3)),
            nn.ReLU(),
            nn.Flatten()
        )
        
        cnn_output_estimate = 256 * (room_size - 4) * (room_size - 4)
        
        self.dense = nn.Sequential(
            nn.Linear(cnn_output_estimate + 3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.Softmax()
        )

    def decide_action(obs):
        input_
        input_data = np.concat((obs[0], (obs[1],), obs[2]))
        output_data = (self.module)(input_data)
        return np.argmax(output-data)

# TODO: Determine what to do with this
"""
# Deep Q Learning Agent
class DQNAgent:
    def __init__(self, room_size, middle_layers):
        input_size = room_size * room_size * 4 + 3
        output_size = 4
        
        layers = []
        current_size = input_size
        for size in middle_layers:
            layers.append( nn.Linear(current_size, size) )
            layers.append( nn.ReLU() )
            current_size = size
        layers.append( nn.Linear(current_size, output_size) )
        layers.append( nn.Softmax() )
        
        self.module = nn.Sequential(*layers)

    def decide_action(obs):
        input_data = np.concat((obs[0], (obs[1],), obs[2]))
        output_data = (self.module)(input_data)
        return np.argmax(output-data)
    
    def receive_reward()
"""