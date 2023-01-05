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
CHOICES = 100


def splev(
        x: torch.Tensor,
        knots: torch.Tensor,
        ctps: torch.Tensor,
        degree: int,
        der: int,
        new: bool
) -> torch.Tensor:
    if der == 0:
        if new:
            return splev_torch_new(x, knots, ctps, degree)
        else:
            return splev_torch_impl(x, knots, ctps, degree)
    else:
        assert der <= degree, "The order of derivative to compute must be less than or equal to k."
        n = ctps.size(-2)
        ctps = (ctps[..., 1:, :] - ctps[..., :-1, :]) / (knots[degree + 1:degree + n] - knots[1:n]).unsqueeze(-1)
        return degree * splev(x, knots[..., 1:-1], ctps, degree - 1, der - 1, new)


def splev_torch_impl(x: torch.Tensor, t: torch.Tensor, c: torch.Tensor, k: int):
    assert t.size(0) == c.size(0) + k + 1, f"{len(t)} != {len(c)} + {k} + {1}"  # m = n + k + 1
    x = torch.atleast_1d(x)
    assert x.dim() == 1 and t.dim() == 1 and c.dim() == 2, f"{x.shape}, {t.shape}, {c.shape}"
    n = c.size(0)
    u = (torch.searchsorted(t, x) - 1).clip(k, n - 1).unsqueeze(-1)
    x = x.unsqueeze(-1)
    d = c[u - k + torch.arange(k + 1, device=c.device)].contiguous()
    for r in range(1, k + 1):
        j = torch.arange(r - 1, k, device=c.device) + 1
        t0 = t[j + u - k]
        t1 = t[j + u + 1 - r]
        alpha = ((x - t0) / (t1 - t0)).unsqueeze(-1)
        d[:, j] = (1 - alpha) * d[:, j - 1] + alpha * d[:, j]
    return d[:, k]


def splev_torch_new(x: torch.Tensor, t: torch.Tensor, c: torch.Tensor, k: int):
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


def compute_traj(ctps_inter: torch.Tensor):
    t = torch.linspace(0, N_CTPS - P, 100, device=ctps_inter.device)
    knots = torch.cat([
        torch.zeros(P, device=ctps_inter.device),
        torch.arange(N_CTPS + 1 - P, device=ctps_inter.device),
        torch.full((P,), N_CTPS - P, device=ctps_inter.device),
    ])
    ctps = torch.cat([
        torch.tensor([[0., 0.]], device=ctps_inter.device),
        ctps_inter,
        torch.tensor([[N_CTPS, 0.]], device=ctps_inter.device)
    ])
    return splev(t, knots, ctps, P, 0, False)


def compute_traj_new(ctps_inter: torch.Tensor):
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
    return splev(t, knots, ctps, P, 0, True)


def loss_fn(sol: torch.Tensor,
            target_pos: torch.Tensor,
            target_scores: torch.Tensor,
            radius: float) -> torch.Tensor:
    traj = compute_traj_new(sol)
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


def real_score(traj: torch.Tensor,
               target_pos: torch.Tensor,
               target_scores: torch.Tensor,
               radius: float) -> torch.Tensor:
    cdist = torch.cdist(target_pos, traj)
    d = cdist.min(-1).values
    hit = (d < radius)
    value = torch.sum(hit * target_scores, dim=-1)
    return value


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
        ctps_inter = torch.rand((P, 2)) * torch.tensor([P, 2.]) + torch.tensor([1., -1.])
        best_score = real_score(compute_traj(ctps_inter), target_pos, class_scores[target_class_pred], RADIUS)
        sol = torch.rand((CHOICES, P, 2)) * torch.tensor([P, 2.]) + torch.tensor([1., -1.])
        sol.requires_grad = True
        optimizer = torch.optim.RMSprop([sol], lr=0.1, alpha=0.95)
        e = time.time()
        while e - s < 0.27:
            loss = loss_fn(sol, target_pos, class_scores[target_class_pred], RADIUS)
            optimizer.zero_grad()
            loss.backward(torch.ones_like(loss))
            optimizer.step()
            e = time.time()
        for i in range(CHOICES):
            score = real_score(compute_traj(sol[i]), target_pos, class_scores[target_class_pred], RADIUS)
            if score > best_score:
                best_score = score
                ctps_inter = sol[i]
        return ctps_inter
