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
class Model(nn.Module):
    def __init__(self, room_size, middle_layers):
        super(Model, self).__init__()

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
        )
    
    def forward(self, obs):
        cnn_input = obs[2]
        cnn_output = (self.cnn)(cnn_input)
        dense_input = np.concat((obs[0], (obs[1],), cnn_output))
        dense_output = (self.dense)(dense_input)
        return dense_input


def train_target_model(model, env, loss_fn, discount_factor, learning_rate, epochs):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        env.reset()

        done = False
        current_state = env.get_observation()
        while not done:
            optimizer.zero_grad()

            predicted_q = model(current_state)

            best_action = np.argmax(predicted_q)
            next_state, reward, done = env.step(best_action)
            target_q = reward + discount_factor * np.max(model(next_state))

            loss = loss_fn(predicted_q, target_q)
            loss.backward()
            optimizer.step()

            current_state = next_state

def train_policy_model(model, target_model, loss_fn, discount_factor, learning_rate, epochs):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        env.reset()

        done = False
        current_state = env.get_observation()
        while not done:
            optimizer.zero_grad()

            predicted_action_weights = model(current_state)

            target_action = np.argmax( target_model(current_state) )
            target_action_weights = np.full_like(predicted_action_weights, -1e9)
            target_action_weights[target_action] = 1e9

            loss = loss_fn(predicted_action_weights, target_action_weights)
            loss.backward()
            optimizer.step()

            current_state = next_state

"""
def train_model(model, optimizer, criterion, train_loader, epochs):
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
"""