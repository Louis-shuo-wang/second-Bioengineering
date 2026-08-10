"""Conservative finite-volume solver for the 1D tumour-immune-chemokine system.
Diffusion: centered fluxes.  Chemotaxis: upwind immune flux.  No-flux boundaries.
Time integration: implicit BDF (scipy solve_ivp, method='BDF')."""
import numpy as np
from scipy.integrate import solve_ivp
from model import Params

def rhs_factory(N, L, p: Params):
    h = L/N
    def Rh(t, U):
        u = U[:N]; v = U[N:2*N]; w = U[2*N:]
        def lap(c, D):
            F = np.zeros(N+1)                 # faces 0..N ; boundary faces stay 0
            F[1:N] = -D*(c[1:]-c[:-1])/h      # interior diffusive fluxes
            return -(F[1:]-F[:-1])/h
        # u
        du = lap(u, p.d1) + u*(1.0-u-v)
        # v : diffusion + upwind chemotaxis
        Fv = np.zeros(N+1)
        Fv[1:N] += -p.d2*(v[1:]-v[:-1])/h
        a = p.xi*(w[1:]-w[:-1])/h             # face chemotactic velocity (length N-1)
        vup = np.where(a >= 0.0, v[:-1], v[1:])
        Fv[1:N] += a*vup
        dv = -(Fv[1:]-Fv[:-1])/h + p.s0 + p.s1*w/(1.0+w) - p.delta*v - p.beta*u*v
        # w
        dw = lap(w, p.d3) + p.alpha*u + p.gamma*u*v - p.ell*w
        return np.concatenate([du, dv, dw])
    return Rh, h

def solve(N, L, p, U0, tspan, t_eval, rtol=1e-7, atol=1e-9):
    Rh, h = rhs_factory(N, L, p)
    sol = solve_ivp(Rh, tspan, U0, method='BDF', t_eval=t_eval,
                    rtol=rtol, atol=atol, dense_output=False)
    return sol, h

def split(U, N):
    return U[:N], U[N:2*N], U[2*N:]

def cell_centers(N, L):
    h = L/N
    return (np.arange(N)+0.5)*h
