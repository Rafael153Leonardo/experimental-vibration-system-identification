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

    # Central differences: a forward difference lags the state by half a
    # sample, which the regression absorbs as artificial damping (the
    # simulated modes then decay orders of magnitude too fast).
    z_dot = np.gradient(z, dt, axis=1, edge_order=2)
    z_state = z
    u_state = u[None, :]
    reg = np.vstack([z_state, u_state])
    AB = z_dot @ np.linalg.pinv(reg)

    A = AB[:, :-1]
    B = AB[:, -1][:, None]
    return A, B, z_state, u_state


def simulate_havok(
    A: np.ndarray,
    B: np.ndarray,
    u: np.ndarray,
    t: np.ndarray,
    z0: np.ndarray,
) -> np.ndarray:
    """Integrate ``z' = A z + B u`` with the recorded forcing ``u(t)``.

    Returns the simulated latent trajectory with shape ``(len(z0), len(t))``,
    for comparing the identified HAVOK model against the extracted modes.
    """

    from scipy.integrate import solve_ivp

    t = np.asarray(t, dtype=float)
    u = np.asarray(u, dtype=float).ravel()
    b = np.asarray(B, dtype=float).ravel()

    def rhs(ti: float, z: np.ndarray) -> np.ndarray:
        return A @ z + b * np.interp(ti, t, u)

    sol = solve_ivp(rhs, (float(t[0]), float(t[-1])), np.asarray(z0, dtype=float), t_eval=t, rtol=1e-6, atol=1e-9)
    return sol.y
