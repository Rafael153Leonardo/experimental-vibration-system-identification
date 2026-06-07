from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def pinn_available() -> bool:
    """Return True when PyTorch is installed."""

    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError as exc:
        raise ImportError("Install torch to use the PINN module.") from exc
    return torch, nn, optim


@dataclass(frozen=True)
class PinnTrainingConfig:
    linear_threshold: float = 15.0
    hidden_dim: int = 64
    hidden_layers: int = 2
    stage1_epochs: int = 500
    stage2_epochs: int = 500
    learning_rate_stage1: float = 1e-3
    learning_rate_stage2: float = 5e-4
    physics_weight: float = 1e-4
    max_points: int = 1200
    seed: int = 7
    device: str | None = None
    initial_mu: float = 0.1
    initial_alpha: float = 1e-5
    initial_stiffness_over_mass: float | None = None


@dataclass(frozen=True)
class PinnResult:
    mu: float
    alpha: float
    stiffness_over_mass: float
    omega_n: float
    frequency_hz: float
    device: str
    history: list[dict[str, float]]
    model: object


def _inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-12)
    return float(np.log(np.expm1(value)))


def _subsample(t: np.ndarray, x: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or len(t) <= max_points:
        return t, x
    idx = np.linspace(0, len(t) - 1, max_points).astype(int)
    return t[idx], x[idx]


def build_pinn_model(
    *,
    t_mean: float,
    t_scale: float,
    hidden_dim: int = 64,
    hidden_layers: int = 2,
    initial_mu: float = 0.1,
    initial_alpha: float = 1e-5,
    initial_stiffness_over_mass: float = 1000.0,
    device: str | None = None,
):
    """Create a neural model with trainable oscillator parameters."""

    torch, nn, _ = _require_torch()
    selected_device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    class DampedOscillatorPINN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[nn.Module] = [nn.Linear(1, hidden_dim), nn.Tanh()]
            for _ in range(max(0, hidden_layers - 1)):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
            layers.append(nn.Linear(hidden_dim, 1))
            self.net = nn.Sequential(*layers)
            self.register_buffer("t_mean", torch.tensor([t_mean], dtype=torch.float32))
            self.register_buffer("t_scale", torch.tensor([max(t_scale, 1e-9)], dtype=torch.float32))
            self.raw_mu = nn.Parameter(torch.tensor([_inverse_softplus(initial_mu)], dtype=torch.float32))
            self.raw_alpha = nn.Parameter(torch.tensor([_inverse_softplus(initial_alpha)], dtype=torch.float32))
            self.raw_k_m = nn.Parameter(
                torch.tensor([_inverse_softplus(initial_stiffness_over_mass)], dtype=torch.float32)
            )

        @property
        def mu(self):
            return torch.nn.functional.softplus(self.raw_mu)

        @property
        def alpha(self):
            return torch.nn.functional.softplus(self.raw_alpha)

        @property
        def stiffness_over_mass(self):
            return torch.nn.functional.softplus(self.raw_k_m)

        def forward(self, t):
            tau = (t - self.t_mean) / self.t_scale
            return self.net(tau)

    return DampedOscillatorPINN().to(selected_device)


def train_two_stage_pinn(t: np.ndarray, x: np.ndarray, config: PinnTrainingConfig | None = None) -> PinnResult:
    """Train a two-stage PINN for a damped oscillator with quadratic drag."""

    torch, _, optim = _require_torch()
    config = config or PinnTrainingConfig()
    torch.manual_seed(config.seed)

    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(t) & np.isfinite(x)
    t = t[mask]
    x = x[mask]
    order = np.argsort(t)
    t = t[order]
    x = x[order]
    if len(t) < 20:
        raise ValueError("At least 20 samples are required for PINN training.")

    t = t - t[0]
    t, x = _subsample(t, x, config.max_points)
    dt = float(np.median(np.diff(t)))
    initial_k = config.initial_stiffness_over_mass
    if initial_k is None:
        freqs = np.fft.rfftfreq(len(x), d=dt)
        amp = np.abs(np.fft.rfft(x - np.mean(x)))
        freq = freqs[int(np.argmax(amp[1:]) + 1)] if len(freqs) > 1 else 1.0
        initial_k = float((2.0 * np.pi * max(freq, 1e-6)) ** 2)

    model = build_pinn_model(
        t_mean=float(np.mean(t)),
        t_scale=float(np.std(t) or 1.0),
        hidden_dim=config.hidden_dim,
        hidden_layers=config.hidden_layers,
        initial_mu=config.initial_mu,
        initial_alpha=config.initial_alpha,
        initial_stiffness_over_mass=initial_k,
        device=config.device,
    )
    device = next(model.parameters()).device
    mse = torch.nn.MSELoss()

    t_all = torch.tensor(t.reshape(-1, 1), dtype=torch.float32, device=device).requires_grad_(True)
    x_all = torch.tensor(x.reshape(-1, 1), dtype=torch.float32, device=device)

    linear_mask = np.abs(x) <= config.linear_threshold
    if np.count_nonzero(linear_mask) < 20:
        linear_mask = np.ones_like(x, dtype=bool)
    t_lin = torch.tensor(t[linear_mask].reshape(-1, 1), dtype=torch.float32, device=device).requires_grad_(True)
    x_lin = torch.tensor(x[linear_mask].reshape(-1, 1), dtype=torch.float32, device=device)

    def residual(x_pred, t_tensor, *, include_quadratic: bool):
        dxdt = torch.autograd.grad(x_pred, t_tensor, torch.ones_like(x_pred), create_graph=True)[0]
        d2xdt2 = torch.autograd.grad(dxdt, t_tensor, torch.ones_like(dxdt), create_graph=True)[0]
        r = d2xdt2 + model.mu * dxdt + model.stiffness_over_mass * x_pred
        if include_quadratic:
            r = r + model.alpha * dxdt * torch.abs(dxdt)
        return r

    history: list[dict[str, float]] = []

    model.raw_alpha.requires_grad_(False)
    optimizer1 = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=config.learning_rate_stage1)
    for epoch in range(config.stage1_epochs):
        optimizer1.zero_grad()
        x_pred = model(t_lin)
        r = residual(x_pred, t_lin, include_quadratic=False)
        data_loss = mse(x_pred, x_lin)
        physics_loss = torch.mean(r**2)
        loss = data_loss + config.physics_weight * physics_loss
        loss.backward()
        optimizer1.step()
        if epoch == config.stage1_epochs - 1:
            history.append(
                {
                    "stage": 1.0,
                    "epoch": float(epoch + 1),
                    "loss": float(loss.detach().cpu()),
                    "data_loss": float(data_loss.detach().cpu()),
                    "physics_loss": float(physics_loss.detach().cpu()),
                }
            )

    model.raw_alpha.requires_grad_(True)
    optimizer2 = optim.Adam(model.parameters(), lr=config.learning_rate_stage2)
    for epoch in range(config.stage2_epochs):
        optimizer2.zero_grad()
        x_pred = model(t_all)
        r = residual(x_pred, t_all, include_quadratic=True)
        data_loss = mse(x_pred, x_all)
        physics_loss = torch.mean(r**2)
        loss = data_loss + config.physics_weight * physics_loss
        loss.backward()
        optimizer2.step()
        if epoch == config.stage2_epochs - 1:
            history.append(
                {
                    "stage": 2.0,
                    "epoch": float(epoch + 1),
                    "loss": float(loss.detach().cpu()),
                    "data_loss": float(data_loss.detach().cpu()),
                    "physics_loss": float(physics_loss.detach().cpu()),
                }
            )

    k_m = float(model.stiffness_over_mass.detach().cpu())
    mu = float(model.mu.detach().cpu())
    alpha = float(model.alpha.detach().cpu())
    omega_n = float(np.sqrt(k_m))
    return PinnResult(
        mu=mu,
        alpha=alpha,
        stiffness_over_mass=k_m,
        omega_n=omega_n,
        frequency_hz=omega_n / (2.0 * np.pi),
        device=str(device),
        history=history,
        model=model,
    )


def predict(model: object, t: np.ndarray) -> np.ndarray:
    """Evaluate a trained PINN model."""

    torch, _, _ = _require_torch()
    device = next(model.parameters()).device
    t_tensor = torch.tensor(np.asarray(t, dtype=float).reshape(-1, 1), dtype=torch.float32, device=device)
    with torch.no_grad():
        y = model(t_tensor).detach().cpu().numpy().ravel()
    return y
