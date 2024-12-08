import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

#Keeping the actor and critic functions seperate to ensure we can optimize both independently of one another.

class Actor(nn.Module):
    def __init__(self, num_states, num_actions):
        super(Actor,self).__init__()

        self.first_layer = nn.Linear(num_states, 64)
        self.second_layer = nn.Linear(64, 64)
        self.third_layer = nn.Linear(64, num_actions)


    def forward(self, input):

        #Converting to tensor if required
        input = torch.as_tensor(input, dtype=torch.float)
        
        #Forward pass
        input_layer = F.relu(self.first_layer(input))
        middle_layer = F.relu(self.second_layer(input_layer))
        output_layer = self.third_layer(middle_layer)

        return output_layer


class Critic(nn.Module):
    def __init__(self, num_states):
        super(Critic,self).__init__()

        self.first_layer = nn.Linear(num_states, 64)
        self.second_layer = nn.Linear(64, 64)
        self.third_layer = nn.Linear(64, 1)


    def forward(self, input):

        #Converting to tensor if required
        input = torch.as_tensor(input, dtype=torch.float)
        
        #Forward pass
        input_layer = F.relu(self.first_layer(input))
        middle_layer = F.relu(self.second_layer(input_layer))
        output_layer = self.third_layer(middle_layer)

        return output_layer