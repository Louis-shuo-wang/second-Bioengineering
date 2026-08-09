"""
Explicit forward-Euler finite-volume stepper with upwind chemotaxis.
Positivity holds under the CFL condition
    dt * [ 2*dim*max_i d_i / h^2  +  xi*max|grad w|/h  +  reaction_bound ] <= 1.
"""
import numpy as np
from model import Params


def rhs_2d(u, v, w, p, hx, hy):
    def lap(c, D):
        out = np.zeros_like(c)
        Fx = np.zeros((c.shape[0], c.shape[1] + 1))
        Fx[:, 1:-1] = -D * (c[:, 1:] - c[:, :-1]) / hx      # no-flux: boundary faces = 0
        out += -(Fx[:, 1:] - Fx[:, :-1]) / hx
        Fy = np.zeros((c.shape[0] + 1, c.shape[1]))
        Fy[1:-1, :] = -D * (c[1:, :] - c[:-1, :]) / hy
        out += -(Fy[1:, :] - Fy[:-1, :]) / hy
        return out

    du = lap(u, p.d1) + u * (1 - u - v)
    dw = lap(w, p.d3) + p.alpha * u + p.gamma * u * v - p.ell * w

    # v: diffusion + upwind chemotaxis
    Fvx = np.zeros((v.shape[0], v.shape[1] + 1))
    Fvx[:, 1:-1] = -p.d2 * (v[:, 1:] - v[:, :-1]) / hx
    ax = p.xi * (w[:, 1:] - w[:, :-1]) / hx
    Fvx[:, 1:-1] += ax * np.where(ax >= 0, v[:, :-1], v[:, 1:])
    Fvy = np.zeros((v.shape[0] + 1, v.shape[1]))
    Fvy[1:-1, :] = -p.d2 * (v[1:, :] - v[:-1, :]) / hy
    ay = p.xi * (w[1:, :] - w[:-1, :]) / hy
    Fvy[1:-1, :] += ay * np.where(ay >= 0, v[:-1, :], v[1:, :])
    dv = -(Fvx[:, 1:] - Fvx[:, :-1]) / hx - (Fvy[1:, :] - Fvy[:-1, :]) / hy
    dv += p.s0 + p.s1 * w / (1 + w) - p.delta * v - p.beta * u * v
    return du, dv, dw


def cfl_dt(u, v, w, p, hx, hy, safety=0.4):
    dmax = max(p.d1, p.d2, p.d3)
    diff = 2 * 2 * dmax / min(hx, hy) ** 2         # 2*dim*dmax/h^2
    gx = np.abs(w[:, 1:] - w[:, :-1]).max() / hx if w.size else 0.0
    gy = np.abs(w[1:, :] - w[:-1, :]).max() / hy if w.size else 0.0
    chemo = p.xi * max(gx, gy) / min(hx, hy)
    react = 2.0 + p.delta + p.beta + p.ell         # Lipschitz-type reaction bound
    denom = diff + chemo + react
    return safety / denom if denom > 0 else 1e-3


def march_2d(Nx, Ny, Lx, Ly, p, u, v, w, T, safety=0.4, record=None):
    hx, hy = Lx / Nx, Ly / Ny
    t = 0.0
    gmin = min(u.min(), v.min(), w.min())
    while t < T:
        dt = cfl_dt(u, v, w, p, hx, hy, safety)
        if t + dt > T:
            dt = T - t
        du, dv, dw = rhs_2d(u, v, w, p, hx, hy)
        u = u + dt * du; v = v + dt * dv; w = w + dt * dw
        t += dt
        gmin = min(gmin, u.min(), v.min(), w.min())
    return u, v, w, t, gmin
