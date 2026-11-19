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

# Deep Q Learning Agent
class DQNAgent:
    def __init__(self, room_size, middle_layers):
        input_size = room_size * room_size + 3
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

    def

# TODO: Determine what to do with this
"""
# Deep Q Learning Agent
class DQNAgent:
    def __init__(self, room_size, middle_layers):
        input_size = room_size * room_size + 3
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

    def
"""
