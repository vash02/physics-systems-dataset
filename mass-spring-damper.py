# requirements: numpy, scipy, matplotlib

import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple, Dict, Optional
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


@dataclass
class MSDParams:
    """Mass–spring–damper parameters."""
    m: float  # mass
    c: float  # damping
    k: float  # stiffness


def msd_rhs(t: float, y: np.ndarray, p: MSDParams, f: Callable[[float], float]) -> np.ndarray:
    """
    State-space ODE for mass–spring–damper:
      y = [x, v]
      x' = v
      v' = (1/m) * (f(t) - c*v - k*x)
    """
    x, v = y
    return np.array([v, (f(t) - p.c * v - p.k * x) / p.m], dtype=float)


def simulate_msd(
    params: MSDParams,
    t_span: Tuple[float, float] = (0.0, 10.0),
    dt: float = 0.001,
    y0: Tuple[float, float] = (0.0, 0.0),
    force: Optional[Callable[[float], float]] = None,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    method: str = "RK45",
) -> Dict[str, np.ndarray]:
    """
    Simulate the MSD system.

    Args:
        params: MSDParams(m, c, k)
        t_span: (t0, tf)
        dt: output sampling step
        y0: (x0, v0)
        force: function f(t) -> scalar. If None, uses zero force.
        rtol, atol: solver tolerances
        method: solve_ivp method ('RK45', 'Radau', 'BDF', ...)

    Returns:
        dict with keys: 't', 'x', 'v', and 'a' (acceleration)
    """
    if force is None:
        force = lambda _t: 0.0

    t_eval = np.arange(t_span[0], t_span[1] + dt, dt)

    sol = solve_ivp(
        fun=lambda t, y: msd_rhs(t, y, params, force),
        t_span=t_span,
        y0=np.asarray(y0, dtype=float),
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        method=method,
        vectorized=False,
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    # States
    x = sol.y[0]
    v = sol.y[1]

    # Acceleration a = (f(t) - c*v - k*x)/m
    f_vals = np.array([force(ti) for ti in sol.t], dtype=float)
    a = (f_vals - params.c * v - params.k * x) / params.m

    return {"t": sol.t, "x": x, "v": v, "a": a}


# ------------------------- Examples ---------------------------------
if __name__ == "__main__":
    # Example 1: Free decay from initial displacement (no external force)
    p = MSDParams(m=1.0, c=0.3, k=4.0)
    out_free = simulate_msd(
        params=p,
        t_span=(0.0, 10.0),
        dt=0.002,
        y0=(1.0, 0.0),         # x(0)=1, v(0)=0
        force=None             # zero input
    )

    # Example 2: Sinusoidal forcing f(t) = F0 * sin(ω t)
    F0 = 1.0
    w = 2.0  # rad/s
    force_sin = lambda t: F0 * np.sin(w * t)

    out_forced = simulate_msd(
        params=p,
        t_span=(0.0, 10.0),
        dt=0.002,
        y0=(0.0, 0.0),
        force=force_sin
    )

    # Quick plots
    fig1, ax1 = plt.subplots()
    ax1.plot(out_free["t"], out_free["x"], label="x(t) free")
    ax1.plot(out_free["t"], out_free["v"], label="v(t) free", linestyle="--")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("State")
    ax1.set_title("MSD Free Response")
    ax1.legend()
    ax1.grid(True)

    fig2, ax2 = plt.subplots()
    ax2.plot(out_forced["t"], out_forced["x"], label="x(t) forced")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Displacement [m]")
    ax2.set_title("MSD Forced Response (sin input)")
    ax2.legend()
    ax2.grid(True)

    plt.show()
#GPT-5
