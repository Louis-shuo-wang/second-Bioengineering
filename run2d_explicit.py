import time, json, numpy as np
from model import Params, coexistence
from explicit_fv import march_2d
from solver2d import centers

p0 = Params(); eq = coexistence(p0)[0]; xic = 12.4738
Nx = Ny = 64; Lx = Ly = 10.0; xi = 1.4 * xic
p = Params(**{**p0.__dict__, 'xi': xi})
x = centers(Nx, Lx); y = centers(Ny, Ly); XX, YY = np.meshgrid(x, y)
rng = np.random.default_rng(3)
pert = 3e-2 * np.cos(6*np.pi*XX/Lx) * np.cos(5*np.pi*YY/Ly) + 2e-3 * rng.standard_normal((Ny, Nx))
u = eq[0] + pert; v = eq[1] + pert; w = eq[2] + pert
t0 = time.time()
u, v, w, tf, gmin = march_2d(Nx, Ny, Lx, Ly, p, u, v, w, T=80.0, safety=0.45)
print(json.dumps(dict(wall_s=round(time.time()-t0, 1), t_final=round(tf, 2),
    u_range=[float(u.min()), float(u.max())], v_range=[float(v.min()), float(v.max())],
    w_range=[float(w.min()), float(w.max())], global_min=float(gmin))))
np.savez('run2d_fields.npz', x=x, y=y, u=u, v=v, w=w, xi=xi, xic=xic, eq=np.array(eq))
print("saved")
