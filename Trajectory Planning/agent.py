import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

P = 3
N_CTPS = 5
RADIUS = 0.3
N_CLASSES = 10
FEATURE_DIM = 256
input_size = 256
output_size = 10
CHOICES = 200


def compute_traj(ctps_inter: torch.Tensor):
    t = torch.linspace(0, N_CTPS - P, 100, device=ctps_inter.device)
    knots = torch.cat([
        torch.zeros(P, device=ctps_inter.device),
        torch.arange(N_CTPS + 1 - P, device=ctps_inter.device),
        torch.full((P,), N_CTPS - P, device=ctps_inter.device),
    ])
    a = [0., 0.] * CHOICES
    a = torch.tensor(a, device=ctps_inter.device)
    a = a.reshape(CHOICES, 1, 2)
    b = [N_CTPS, 0.] * CHOICES
    b = torch.tensor(b, device=ctps_inter.device)
    b = b.reshape(CHOICES, 1, 2)
    ctps = torch.cat([a, ctps_inter, b], dim=1)
    return splev(t, knots, ctps, P)


def splev(
        x: torch.Tensor,
        knots: torch.Tensor,
        ctps: torch.Tensor,
        degree: int,
) -> torch.Tensor:
    return splev_torch(x, knots, ctps, degree)


def splev_torch(x: torch.Tensor, t: torch.Tensor, c: torch.Tensor, k: int):
    x = torch.atleast_1d(x)
    assert t.size(0) == c.size(1) + k + 1, f"{len(t)} != {len(c)} + {k} + {1}"  # m = n + k + 1
    assert x.dim() == 1 and t.dim() == 1 and c.dim() == 3, f"{x.shape}, {t.shape}, {c.shape}"
    n = c.size(1)
    u = (torch.searchsorted(t, x) - 1).clip(k, n - 1).unsqueeze(-1)
    x = x.unsqueeze(-1)
    a = u - k + torch.arange(k + 1, device=c.device)
    d = c[:, a].contiguous()
    for r in range(1, k + 1):
        j = torch.arange(r - 1, k, device=c.device) + 1
        t0 = t[j + u - k]
        t1 = t[j + u + 1 - r]
        alpha = ((x - t0) / (t1 - t0)).unsqueeze(-1)
        d[:, :, j] = (1 - alpha) * d[:, :, j - 1] + alpha * d[:, :, j]
    return d[:, :, k]


def loss_fn(sol: torch.Tensor,
            target_pos: torch.Tensor,
            target_scores: torch.Tensor,
            radius: float) -> torch.Tensor:
    traj = compute_traj(sol)
    cdist = torch.cdist(target_pos, traj)
    d = cdist.min(-1).values
    hits = d <= radius
    d[hits] = 1
    d[~hits] = radius / d[~hits]
    result = d * target_scores
    result /= 10
    values = torch.sum(result, dim=-1)
    values = -values
    return values


def real_score(sol: torch.Tensor,
               target_pos: torch.Tensor,
               target_scores: torch.Tensor,
               radius: float):
    traj = compute_traj(sol)
    cdist = torch.cdist(target_pos, traj)
    d = cdist.min(-1).values
    hit = (d < radius)
    values = torch.sum(hit * target_scores, dim=-1)
    idx = torch.argmax(values)
    return values.view(-1)[idx], sol[idx]


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


class Agent:

    def __init__(self):
        self.classifier = Net(input_size, int(input_size * 1.5), int(input_size * 1.5), output_size)
        file_name = os.path.join(os.path.dirname(__file__), 'model.pkl')
        self.classifier.load_state_dict(torch.load(file_name))

    def get_action(self,
                   target_pos: torch.Tensor,
                   target_features: torch.Tensor,
                   class_scores: torch.Tensor) -> torch.Tensor:
        s = time.time()
        target_class_pred = self.classifier(target_features)
        target_class_pred = torch.max(target_class_pred, 1)[1]
        target_class_pred = target_class_pred.data.numpy()
        sol = torch.rand((CHOICES, P, 2)) * torch.tensor([P, 2.]) + torch.tensor([1., -1.])
        sol.requires_grad = True
        optimizer = torch.optim.RMSprop([sol], lr=0.1, alpha=0.95)
        e = time.time()
        while e - s < 0.25:
            loss = loss_fn(sol, target_pos, class_scores[target_class_pred], RADIUS)
            optimizer.zero_grad()
            loss.backward(torch.ones_like(loss))
            optimizer.step()
            e = time.time()
        score, ctps = real_score(sol, target_pos, class_scores[target_class_pred], RADIUS)
        if score > torch.tensor(0):
            return ctps
        else:
            return torch.Tensor([[-100, -100], [-100, -100], [-100, 100]])
