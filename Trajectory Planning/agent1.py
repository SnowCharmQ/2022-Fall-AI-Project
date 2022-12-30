import os
import copy
import math
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from src import *

input_size = 256
output_size = 10
CHOICES = 16


class Net(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.hidden2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.out = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.out(x)
        return x


def real_score(traj: torch.Tensor,
               target_pos: torch.Tensor,
               target_scores: torch.Tensor,
               radius: float) -> torch.Tensor:
    cdist = torch.cdist(target_pos, traj)
    d = cdist.min(-1).values
    hit = (d < radius)
    value = torch.sum(hit * target_scores, dim=-1)
    return value


def loss_fn(sol: torch.Tensor,
            target_pos: torch.Tensor,
            target_scores: torch.Tensor,
            radius: float) -> torch.Tensor:
    values = torch.zeros(CHOICES)
    for i in range(CHOICES):
        temp = sol[i]
        traj = compute_traj(temp)
        cdist = torch.cdist(target_pos, traj)
        d = cdist.min(-1).values
        hits = d <= radius
        d[hits] = 1
        d[~hits] = radius / d[~hits]
        result = d * target_scores
        result /= 10
        value = torch.sum(result, dim=-1)
        value = -value
        values[i] = value
    return values


class Agent1:

    def __init__(self):
        self.classifier = Net(input_size, int(input_size * 1.5), int(input_size * 1.5), output_size)
        self.classifier.load_state_dict(torch.load("model.pkl"))

    def get_action(self,
                   target_pos: torch.Tensor,
                   target_features: torch.Tensor,
                   class_scores: torch.Tensor) -> torch.Tensor:
        s = time.time()
        target_class_pred = self.classifier(target_features)
        target_class_pred = torch.max(target_class_pred, 1)[1]
        target_class_pred = target_class_pred.data.numpy()
        ctps_inter = torch.rand((N_CTPS - 2, 2)) * torch.tensor([N_CTPS - 2, 2.]) + torch.tensor([1., -1.])
        best_score = real_score(compute_traj(ctps_inter), target_pos, class_scores[target_class_pred], RADIUS)
        ctps_inter.requires_grad = True
        sol = torch.zeros((CHOICES, 3, 2))
        for i in range(CHOICES):
            temp = torch.rand((N_CTPS - 2, 2)) * torch.tensor([N_CTPS - 2, 2.]) + torch.tensor([1., -1.])
            sol[i] = temp
        sol.requires_grad = True
        optimizer = torch.optim.RMSprop([sol], lr=0.1)
        while time.time() - s < 0.25:
            loss = loss_fn(sol, target_pos, class_scores[target_class_pred], RADIUS)
            optimizer.zero_grad()
            loss.backward(torch.ones_like(loss))
            optimizer.step()
        for i in range(CHOICES):
            score = real_score(compute_traj(sol[i]), target_pos, class_scores[target_class_pred], RADIUS)
            if score > best_score:
                best_score = score
                ctps_inter = sol[i]
        return ctps_inter
