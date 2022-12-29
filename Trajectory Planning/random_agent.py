import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from src import *

input_size = 256
output_size = 10


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


class RandomAgent:

    def __init__(self):
        self.classifier = Net(input_size, int(input_size * 1.5), int(input_size * 1.5), output_size)
        self.classifier.load_state_dict(torch.load("model.pkl"))

    def evaluate(self, traj: torch.Tensor,
                 target_pos: torch.Tensor,
                 target_scores: torch.Tensor,
                 radius: float) -> torch.Tensor:
        cdist = torch.cdist(target_pos, traj)
        d = cdist.min(-1).values
        hits = d <= radius
        d[hits] = 1
        d[~hits] = radius / d[~hits]
        value = torch.sum(d * target_scores, dim=-1)
        return value

    def get_action(self,
                   target_pos: torch.Tensor,
                   target_features: torch.Tensor,
                   class_scores: torch.Tensor) -> torch.Tensor:
        s = time.time()
        target_class_pred = self.classifier(target_features)
        target_class_pred = torch.max(target_class_pred, 1)[1]
        target_class_pred = target_class_pred.data.numpy()

        ctps_inter = torch.rand((N_CTPS - 2, 2)) * torch.tensor([N_CTPS - 2, 2.]) + torch.tensor([1., -1.])
        ctps_inter.requires_grad = True
        best_score = -100
        while not time.time() - s > 0.29:
            temp = torch.rand((N_CTPS - 2, 2)) * torch.tensor([N_CTPS - 2, 2.]) + torch.tensor([1., -1.])
            score = evaluate(compute_traj(temp), target_pos, class_scores[target_class_pred], RADIUS)
            if score > best_score:
                best_score = score
                ctps_inter = temp
        if best_score < 0:
            ctps_inter = torch.Tensor([[-100, -100], [-100, -100], [-100, 100]])
        return ctps_inter
