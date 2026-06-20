from __future__ import annotations

import numpy as np


def hankel_matrix(x: np.ndarray, delays: int) -> np.ndarray:
    """Build a Hankel delay matrix from a one-dimensional signal."""

    x = np.asarray(x, dtype=float)
    if delays >= len(x) // 2:
        raise ValueError("delays should be well below half the number of samples.")
    cols = len(x) - delays + 1
    return np.vstack([x[i : i + cols] for i in range(delays)])


def havok_svd(x: np.ndarray, *, delays: int = 100):
    """Compute the SVD basis used by HAVOK."""

    H = hankel_matrix(x, delays)
    U, S, Vt = np.linalg.svd(H, full_matrices=False)
    return H, U, S, Vt


def identify_havok(t: np.ndarray, S: np.ndarray, Vt: np.ndarray, *, rank: int = 6):
    """Identify z_dot = A z + B u in HAVOK coordinates."""

    t = np.asarray(t, dtype=float)
    Z = np.diag(S[:rank]) @ Vt[:rank, :]
    z = Z[:-1, :]
    u = Z[-1, :]
    dt = float(np.median(np.diff(t)))

    z_dot = (z[:, 1:] - z[:, :-1]) / dt
    z_state = z[:, :-1]
    u_state = u[:-1][None, :]
    reg = np.vstack([z_state, u_state])
    AB = z_dot @ np.linalg.pinv(reg)

    A = AB[:, :-1]
    B = AB[:, -1][:, None]
    return A, B, z_state, u_state
