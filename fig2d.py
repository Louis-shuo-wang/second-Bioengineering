import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
d = np.load('run2d_fields.npz'); x = d['x']; y = d['y']; u = d['u']; v = d['v']; w = d['w']
xi = float(d['xi']); xic = float(d['xic']); eq = d['eq']
fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.1))
ext = [x.min(), x.max(), y.min(), y.max()]
for a, (Fld, name, cmap) in zip(ax, [(u, r'tumour $u$', 'viridis'),
                                     (v, r'immune $v$', 'magma'),
                                     (w, r'chemokine $w$', 'cividis')]):
    im = a.imshow(Fld, origin='lower', extent=ext, aspect='equal', cmap=cmap)
    a.set(xlabel='x', ylabel='y', title=name)
    plt.colorbar(im, ax=a, fraction=0.046, pad=0.04)
fig.suptitle(r'2D chemotaxis-driven pattern at $\xi=1.4\,\xi_c$ '
             f'($\\xi_c\\approx{xic:.1f}$): immune cells aggregate into foci where '
             'the tumour is locally depleted', fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig('fig_exp7.pdf'); fig.savefig('fig_2d_check.png', dpi=95)
print('saved; u', u.min(), u.max(), 'v', v.min(), v.max())
