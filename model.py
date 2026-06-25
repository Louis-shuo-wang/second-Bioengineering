"""
Tumour-immune-chemokine model: equilibria + linear stability (matrix M_k).
Nondimensional system:
  u_t = d1 Lap u + u(1-u-v)
  v_t = d2 Lap v - xi div(v grad w) + s0 + s1 w/(1+w) - delta v - beta u v
  w_t = d3 Lap w + alpha u + gamma u v - ell w
"""
import numpy as np
from dataclasses import dataclass, asdict
from scipy.optimize import brentq

@dataclass
class Params:
    d1: float = 0.01
    d2: float = 0.10
    d3: float = 1.00
    xi: float = 0.0
    s0: float = 0.10     # sigma_0
    s1: float = 0.50     # sigma_1
    delta: float = 0.30
    beta: float = 0.40
    alpha: float = 0.50
    gamma: float = 0.50
    ell: float = 1.00

# ---------- equilibria ----------
def w_of_u(u, p):
    return u*(p.alpha + p.gamma*(1.0-u))/p.ell

def F(u, p):
    w = w_of_u(u, p)
    return p.s0 + p.s1*w/(1.0+w) - (p.delta + p.beta*u)*(1.0-u)

def coexistence(p, ngrid=400):
    """Return all interior roots u* in (0,1) of F, with (u*,v*,w*)."""
    us = np.linspace(1e-6, 1-1e-6, ngrid)
    Fs = np.array([F(u, p) for u in us])
    roots = []
    for i in range(len(us)-1):
        if Fs[i] == 0.0:
            roots.append(us[i])
        elif Fs[i]*Fs[i+1] < 0:
            r = brentq(F, us[i], us[i+1], args=(p,))
            roots.append(r)
    out = []
    for u in roots:
        v = 1.0 - u
        w = w_of_u(u, p)
        out.append((u, v, w))
    return out

def tumour_free(p):
    return (0.0, p.s0/p.delta, 0.0)

# ---------- linear stability ----------
def reaction_jacobian(eq, p):
    """J = d(f1,f2,f3)/d(u,v,w) at equilibrium eq=(u,v,w)."""
    u, v, w = eq
    J = np.array([
        [1.0 - 2*u - v,        -u,                      0.0],
        [-p.beta*v,            -(p.delta + p.beta*u),   p.s1/(1.0+w)**2],
        [p.alpha + p.gamma*v,   p.gamma*u,             -p.ell],
    ])
    return J

def M_k(eq, p, mu):
    """Mode-mu stability matrix.  mu = Neumann eigenvalue (= (k pi/L)^2 in 1D).
       Diffusion contributes -d_i mu on the diagonal.
       Chemotaxis -xi div(v grad w) linearises to -xi v* Lap(w_hat),
       i.e. +xi v* mu in the (v,w) entry.
    """
    u, v, w = eq
    J = reaction_jacobian(eq, p)
    D = np.diag([p.d1, p.d2, p.d3])
    Mk = J - mu*D
    Mk[1, 2] += p.xi * v * mu          # chemotaxis coupling
    return Mk

def max_re_eig(eq, p, mus):
    """max over given mu's and over eigenvalues of Re(lambda)."""
    best = -np.inf
    argk = None
    for m in mus:
        ev = np.linalg.eigvals(M_k(eq, p, m))
        r = ev.real.max()
        if r > best:
            best = r; argk = m
    return best, argk

if __name__ == "__main__":
    p = Params()
    print("=== base parameters ===")
    print(asdict(p))

    print("\n=== tumour-free equilibrium ===")
    E0 = tumour_free(p)
    print("E0 =", E0, " (v*=s0/delta=%.4f)" % (p.s0/p.delta))
    J0 = reaction_jacobian(E0, p)
    print("reaction Jacobian at E0:\n", np.round(J0,4))
    print("eigs of reaction Jacobian at E0:", np.round(np.linalg.eigvals(J0),4))
    print("predicted tumour mode 1 - s0/delta =", 1 - p.s0/p.delta)

    print("\n=== coexistence equilibria ===")
    eqs = coexistence(p)
    for eq in eqs:
        print("  (u*,v*,w*) = (%.6f, %.6f, %.6f)  F=%.2e" %
              (eq[0], eq[1], eq[2], F(eq[0], p)))
    eqc = eqs[0]
    print("F(0)=%.4f (should be s0-delta=%.4f)" % (F(1e-9,p), p.s0-p.delta))
    print("F(1)=%.4f" % F(1-1e-9, p))
    print("reaction Jacobian at coexistence:\n", np.round(reaction_jacobian(eqc,p),4))
    print("eigs (mu=0):", np.round(np.linalg.eigvals(M_k(eqc,p,0.0)),4))
