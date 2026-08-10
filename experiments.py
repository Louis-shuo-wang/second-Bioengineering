import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import Params, coexistence, tumour_free, M_k, reaction_jacobian
from solver import solve, split, cell_centers

plt.rcParams.update({'font.size': 11, 'figure.dpi': 130, 'axes.grid': True,
                     'grid.alpha': 0.3, 'axes.axisbelow': True})
RES = {}
FIGDIR = '.'
rng = np.random.default_rng(0)
Lbase = 10.0

# ============================================================
# EXPERIMENT 1: tumour-free threshold
# ============================================================
def exp1():
    L=Lbase; N=200; x=cell_centers(N,L)
    out={}
    fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    disp={'decay (s0>delta)':r'decay ($\sigma_0>\delta$)',
          'invade (s0<delta)':r'invade ($\sigma_0<\delta$)'}
    for tag,s0,col in [('decay (s0>delta)',0.50,'C0'),('invade (s0<delta)',0.10,'C3')]:
        lab=disp[tag]
        p=Params(xi=0.0, s0=s0, delta=0.30)
        v0=s0/p.delta
        # small tumour seed on tumour-free background
        u0=1e-3*(1.0+0.3*np.cos(2*np.pi*x/L))
        U0=np.concatenate([u0, np.full(N,v0), np.zeros(N)])
        t_eval=np.linspace(0,40,400)
        sol,h=solve(N,L,p,U0,(0,40),t_eval,rtol=1e-10,atol=1e-13)
        uinf=[]; umass=[]
        for j in range(sol.y.shape[1]):
            u,v,w=split(sol.y[:,j],N); uinf.append(u.max()); umass.append(h*u.sum())
        uinf=np.array(uinf); umass=np.array(umass)
        rate_pred=1.0-s0/p.delta
        # measure exponential rate in accurately-resolved linear window
        mask=(uinf>1e-6)&(uinf<3e-3)&(t_eval>0.5)
        tt=t_eval[mask]; ll=np.log(uinf[mask])
        rate_meas=np.polyfit(tt,ll,1)[0] if mask.sum()>5 else np.nan
        out[tag]=dict(s0=s0,rate_pred=rate_pred,rate_meas=float(rate_meas),
                      uinf_final=float(uinf[-1]))
        axes[0].semilogy(t_eval,uinf,col,label=lab)
        axes[1].plot(t_eval,umass,col,label=lab)
        # growth-rate panel: compare slope to prediction
        axes[2].plot(rate_pred,rate_meas,'o',color=col,ms=9,label=lab)
    axes[0].set(xlabel='t',ylabel=r'$\|u(t)\|_\infty$',title='(a) tumour amplitude')
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel='t',ylabel=r'$\int_\Omega u\,dx$',title='(b) tumour mass')
    axes[1].legend(fontsize=8)
    lim=[-0.8,0.8]; axes[2].plot(lim,lim,'k--',lw=1,alpha=0.6)
    axes[2].set(xlabel=r'predicted $1-\sigma_0/\delta$',ylabel='measured rate',
                title='(c) growth rate',xlim=lim,ylim=lim); axes[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f'{FIGDIR}/fig_exp1.pdf'); plt.close(fig)
    RES['exp1']=out
    print('EXP1',json.dumps(out,indent=0))

# ============================================================
# EXPERIMENT 2: coexistence equilibrium verification
# ============================================================
def exp2():
    L=Lbase; N=200; x=cell_centers(N,L)
    p=Params(xi=0.0)  # below any instability; converge to homogeneous coexistence
    eq=coexistence(p)[0]
    # random positive IC
    u0=0.3+0.2*rng.random(N); v0=0.3+0.2*rng.random(N); w0=0.2+0.2*rng.random(N)
    U0=np.concatenate([u0,v0,w0])
    t_eval=np.linspace(0,200,400)
    sol,h=solve(N,L,p,U0,(0,200),t_eval)
    err=[]
    for j in range(sol.y.shape[1]):
        u,v,w=split(sol.y[:,j],N)
        e=np.sqrt(h*((u-eq[0])**2+(v-eq[1])**2+(w-eq[2])**2).sum())
        err.append(e)
    u,v,w=split(sol.y[:,-1],N)
    final=(float(u.mean()),float(v.mean()),float(w.mean()))
    out=dict(analytic=[float(c) for c in eq], numeric=list(final),
             abs_err=[abs(final[i]-eq[i]) for i in range(3)],
             final_L2_err=float(err[-1]))
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    axes[0].semilogy(t_eval,err,'C2')
    axes[0].set(xlabel='t',ylabel=r'$\|(u,v,w)-(u^*,v^*,w^*)\|_{L^2}$',
                title='(a) convergence to coexistence')
    labels=['u','v','w']; cols=['C0','C1','C3']
    for c,lab,col,eqc in zip([u,v,w],labels,cols,eq):
        axes[1].plot(x,c,col,label=f'{lab} (num)')
        axes[1].axhline(eqc,ls='--',color=col,alpha=0.7)
    axes[1].set(xlabel='x',title='(b) final field vs analytic (dashed)')
    axes[1].legend(fontsize=8,ncol=3)
    fig.tight_layout(); fig.savefig(f'{FIGDIR}/fig_exp2.pdf'); plt.close(fig)
    RES['exp2']=out
    print('EXP2',json.dumps(out,indent=0))

# ============================================================
# EXPERIMENT 3: dispersion relation + simulation Fourier spectrum
# ============================================================
def exp3():
    L=Lbase; N=256; x=cell_centers(N,L)
    p0=Params()
    eq=coexistence(p0)[0]
    us,vs,ws=eq

    # ----------------------------------------------------------------
    # (0) ONSET DIAGNOSTICS over the admissible INTEGER Neumann modes.
    #     Verifies every number in Remark rem:hopf-onset-quant:
    #       xi_c, k*, omega*, a1a2-a3 = 0, a3 > 0, and A_k t - p s < 0.
    # ----------------------------------------------------------------
    from scipy.optimize import brentq
    def mu_of_k(k): return (k*np.pi/L)**2
    def maxRe(k,xi):
        pp=Params(**{**p0.__dict__,'xi':xi})
        return np.linalg.eigvals(M_k(eq,pp,mu_of_k(k))).real.max()
    def rh_coeffs(k,xi):
        mu=mu_of_k(k)
        A=us+p0.d1*mu; B=p0.delta+p0.beta*us+p0.d2*mu; C=p0.ell+p0.d3*mu
        p_=us; q_=p0.beta*vs; r_=p0.s1/(1+ws)**2+xi*vs*mu
        s_=p0.alpha+p0.gamma*vs; t_=p0.gamma*us
        a1=A+B+C
        a2=A*B+A*C+B*C-r_*t_-p_*q_
        a3=A*B*C-A*r_*t_-p_*q_*C+p_*r_*s_
        return a1,a2,a3,(A,t_,p_,s_)
    # per-mode onset sensitivity: smallest xi making mode k marginal
    onset={}
    for k in range(1,26):
        f=lambda xi: maxRe(k,xi)
        if f(0.0)>=0:
            onset[k]=0.0; continue
        hi=1.0
        while f(hi)<0 and hi<1e4: hi*=1.6
        if f(hi)>=0: onset[k]=brentq(f,0.0,hi,xtol=1e-10,rtol=1e-12)
    kstar_onset=min(onset,key=onset.get)
    xic=onset[kstar_onset]                       # integer-mode critical sensitivity
    # eigenvalues at the onset -> marginal frequency omega*
    ppc=Params(**{**p0.__dict__,'xi':xic})
    evc=np.linalg.eigvals(M_k(eq,ppc,mu_of_k(kstar_onset)))
    pair=evc[np.abs(evc.real)<1e-4]
    omega_star=float(np.abs(pair.imag).max()) if pair.size else float(np.abs(evc.imag).max())
    marg_re=float(evc.real.max())
    # Routh-Hurwitz at onset
    a1,a2,a3,_=rh_coeffs(kstar_onset,xic)
    hopf_resid=float(a1*a2-a3)
    # sign of A_k t - p s across the band
    Aktps=[float(rh_coeffs(k,xic)[3][0]*rh_coeffs(k,xic)[3][1]
                 -rh_coeffs(k,xic)[3][2]*rh_coeffs(k,xic)[3][3]) for k in range(1,16)]
    onset_diag=dict(
        xi_c=float(xic), k_star_onset=int(kstar_onset),
        marginal_Re=marg_re, omega_star=omega_star,
        a1=float(a1), a2=float(a2), a3=float(a3),
        hopf_resid_a1a2_minus_a3=hopf_resid,
        sqrt_a2=float(np.sqrt(a2)), a3_positive=bool(a3>0),
        Akt_minus_ps_all_negative=bool(all(v<0 for v in Aktps)))
    # assertions: every claimed number must hold
    assert abs(xic-12.47)<0.05,           f"xi_c={xic}"
    assert kstar_onset==5,                 f"k*={kstar_onset}"
    assert abs(omega_star-0.99)<0.02,      f"omega*={omega_star}"
    assert abs(hopf_resid)<1e-6,           f"a1a2-a3={hopf_resid}"
    assert abs(a3-4.68)<0.02 and a3>0,     f"a3={a3}"
    assert onset_diag['Akt_minus_ps_all_negative']
    print("EXP3 onset:  xi_c=%.4f  k*=%d  omega*=%.4f  a1a2-a3=%.2e  a3=%.4f (>0:%s)"
          %(xic,kstar_onset,omega_star,hopf_resid,a3,a3>0))

    # ----------------------------------------------------------------
    # (a) dispersion relation (continuous mu + admissible integer modes)
    # ----------------------------------------------------------------
    mu_cont=np.linspace(0,6,800)
    kgrid=np.arange(0,26)
    mu_k=(kgrid*np.pi/L)**2
    fig,axes=plt.subplots(1,2,figsize=(11,4.0))
    for xi,col in [(0.0,'C0'),(0.6*xic,'C1'),(1.6*xic,'C3')]:
        pp=Params(**{**p0.__dict__,'xi':xi})
        gc=[np.linalg.eigvals(M_k(eq,pp,m)).real.max() for m in mu_cont]
        gd=[np.linalg.eigvals(M_k(eq,pp,m)).real.max() for m in mu_k]
        axes[0].plot(np.sqrt(mu_cont)*L/np.pi,gc,col,lw=1.6,label=fr'$\xi={xi:.1f}$')
        axes[0].plot(kgrid,gd,'o',color=col,ms=3)
    axes[0].axhline(0,color='k',lw=0.8)
    axes[0].axvline(kstar_onset,color='0.5',ls=':',lw=1)
    axes[0].set(xlabel='wavenumber index k',ylabel=r'$\max_j\,\mathrm{Re}\,\lambda_j(M_k)$',
                title='(a) dispersion relation',xlim=(0,18))
    axes[0].legend(fontsize=8)

    # ----------------------------------------------------------------
    # (b) simulation above onset: modal growth via DCT (Neumann cosine basis)
    # ----------------------------------------------------------------
    from scipy.fft import dct
    xi=1.6*xic; pp=Params(**{**p0.__dict__,'xi':xi})
    U0=np.concatenate([eq[0]+1e-4*rng.standard_normal(N),
                       eq[1]+1e-4*rng.standard_normal(N),
                       eq[2]+1e-4*rng.standard_normal(N)])
    t_eval=np.linspace(0,18,19)
    sol,h=solve(N,L,pp,U0,(0,18),t_eval,rtol=1e-10,atol=1e-12)
    amps=[]
    for j in range(sol.y.shape[1]):
        u,v,w=split(sol.y[:,j],N)
        ck=np.abs(dct(v-v.mean(), type=2, norm='ortho'))
        amps.append(ck)
    amps=np.array(amps)
    meas=[]
    for kk in range(1,18):
        a=amps[:,kk]; good=a>1e-7
        if good.sum()>=4:
            r=np.polyfit(t_eval[good],np.log(a[good]),1)[0]
            meas.append((kk,r))
    meas=np.array(meas)
    pred=np.array([np.linalg.eigvals(M_k(eq,pp,(k*np.pi/L)**2)).real.max() for k in meas[:,0]])
    band=pred>-0.05
    axes[1].plot(meas[:,0],pred,'C3-o',ms=4,label='linear theory  max Re$\\lambda(M_k)$')
    axes[1].plot(meas[band,0],meas[band,1],'ks',ms=6,mfc='none',label='simulation (DCT, active band)')
    axes[1].plot(meas[~band,0],meas[~band,1],'x',color='0.6',ms=6,label='damped modes (noise floor)')
    axes[1].axhline(0,color='k',lw=0.8)
    axes[1].set(xlabel='mode k',ylabel='growth rate',title=fr'(b) modal growth rates, $\xi={xi:.1f}$')
    axes[1].legend(fontsize=7.5)
    fig.tight_layout(); fig.savefig(f'{FIGDIR}/fig_exp3.pdf'); plt.close(fig)

    kstar_dom=int(meas[np.argmax(pred),0])       # dominant mode at the SUPERCRITICAL xi
    band_corr=float(np.corrcoef(pred[band],meas[band,1])[0,1])
    RES['exp3']=dict(onset=onset_diag,
                     k_star_dominant_supercritical=kstar_dom,
                     band_corr=band_corr,
                     pred_kstar=float(pred.max()),
                     meas_kstar=float(meas[np.argmax(pred),1]))
    print('EXP3 supercritical: dominant k*=%d  band_corr=%.4f'%(kstar_dom,band_corr))


# ============================================================
# EXPERIMENT 4: chemotaxis-driven pattern formation (vary xi)
# ============================================================
def exp4():
    L=Lbase; N=128; x=cell_centers(N,L)
    p0=Params(); eq=coexistence(p0)[0]
    xic=float(np.load(f'{FIGDIR}/xic.npy')[0])
    cases=[('subcritical',0.7*xic),('near onset',1.02*xic),('supercritical',1.8*xic)]
    fig,axes=plt.subplots(2,3,figsize=(13,6.4))
    out={}
    seed=np.cos(6*np.pi*x/L)
    U0base=np.concatenate([eq[0]+1e-2*seed+1e-3*rng.standard_normal(N),
                           eq[1]+1e-2*seed+1e-3*rng.standard_normal(N),
                           eq[2]+1e-2*seed+1e-3*rng.standard_normal(N)])
    for col,(tag,xi) in enumerate(cases):
        pp=Params(**{**p0.__dict__,'xi':xi})
        t_eval=np.linspace(0,120,60)
        sol,h=solve(N,L,pp,U0base.copy(),(0,120),t_eval,rtol=1e-6,atol=1e-9)
        u,v,w=split(sol.y[:,-1],N)
        amp=[(split(sol.y[:,j],N)[0].max()-split(sol.y[:,j],N)[0].min())
             for j in range(sol.y.shape[1])]
        a0=axes[0,col]; a1=axes[1,col]
        a0.plot(x,u,'C0',label='u'); a0.plot(x,v,'C3',label='v')
        a0.set(xlabel='x',title=f'{tag}: '+fr'$\xi={xi:.1f}$')
        a0.legend(fontsize=8)
        a1.semilogy(t_eval,np.maximum(amp,1e-12),'C2')
        a1.set(xlabel='t',ylabel=r'$\max u-\min u$',title='amplitude')
        out[tag]=dict(xi=xi, final_u_amp=float(u.max()-u.min()),
                      final_v_amp=float(v.max()-v.min()),
                      minvals=[float(u.min()),float(v.min()),float(w.min())])
    fig.suptitle(fr'Chemotaxis-driven patterning ($\xi_c={xic:.1f}$): top = final fields, bottom = amplitude growth',
                 fontsize=10)
    fig.tight_layout(); fig.savefig(f'{FIGDIR}/fig_exp4.pdf'); plt.close(fig)
    RES['exp4']=out
    print('EXP4',json.dumps(out,indent=0))

# ============================================================
# EXPERIMENT 5: two-parameter sensitivity (stability) maps
# ============================================================
def exp5():
    p0=Params(); L=Lbase
    mu_k=((np.arange(1,30))*np.pi/L)**2
    def maxgrowth(p):
        eqs=coexistence(p)
        if not eqs: return np.nan
        eq=eqs[0]
        return max(np.linalg.eigvals(M_k(eq,p,m)).real.max() for m in mu_k)
    def make_map(param,vals,xis):
        Z=np.full((len(vals),len(xis)),np.nan)
        for i,pv in enumerate(vals):
            for j,xv in enumerate(xis):
                d={**p0.__dict__,'xi':xv,param:pv}
                Z[i,j]=maxgrowth(Params(**d))
        return Z
    xis=np.linspace(0,30,60)
    specs=[('s1',np.linspace(0.05,1.5,50),r'$\sigma_1$'),
           ('delta',np.linspace(0.15,0.6,50),r'$\delta$'),
           ('d2',np.linspace(0.02,0.4,50),r'$d_2$')]
    fig,axes=plt.subplots(1,3,figsize=(14,4.0))
    for ax,(param,vals,lab) in zip(axes,specs):
        Z=make_map(param,vals,xis)
        vmax=np.nanmax(np.abs(Z))
        pc=ax.pcolormesh(xis,vals,Z,cmap='RdBu_r',vmin=-vmax,vmax=vmax,shading='auto')
        ax.contour(xis,vals,Z,levels=[0.0],colors='k',linewidths=1.6)
        ax.set(xlabel=r'$\xi$',ylabel=lab,title=f'max Re$\\lambda$ over modes')
        fig.colorbar(pc,ax=ax,shrink=0.85)
    fig.suptitle('Stability maps: blue = homogeneous stable, red = patterning (black line = onset)',
                 y=1.02,fontsize=10)
    fig.tight_layout(); fig.savefig(f'{FIGDIR}/fig_exp5.pdf',bbox_inches='tight'); plt.close(fig)
    RES['exp5']='generated'
    print('EXP5 done')

# ============================================================
# EXPERIMENT 6: grid convergence + residual certification
# ============================================================
def exp6():
    L=Lbase; p0=Params(); eq=coexistence(p0)[0]
    xic=float(np.load(f'{FIGDIR}/xic.npy')[0])
    pp=Params(**{**p0.__dict__,'xi':1.3*xic})   # mildly supercritical: smooth pattern
    t_eval=np.linspace(0,60,61)
    Ns=[64,128,256]
    sols={}; mins={}
    for N in Ns+[512]:
        x=cell_centers(N,L)
        seed=np.cos(2*np.pi*5*x/L)
        U0=np.concatenate([eq[0]+1e-3*seed,eq[1]+1e-3*seed,eq[2]+1e-3*seed])
        sol,h=solve(N,L,pp,U0,(0,60),t_eval,rtol=1e-7,atol=1e-9)
        sols[N]=sol
        m=np.inf
        for j in range(sol.y.shape[1]):
            u,v,w=split(sol.y[:,j],N); m=min(m,u.min(),v.min(),w.min())
        mins[N]=float(m)
    # spatial convergence: compare final u-field N vs 2N (project coarse->fine by repeat)
    def final_fields(N):
        u,v,w=split(sols[N].y[:,-1],N); return u,v,w
    conv=[]
    for N in [64,128,256]:
        uN,vN,wN=final_fields(N); u2,v2,w2=final_fields(2*N)
        # average fine pairs to coarse grid
        u2c=u2.reshape(N,2).mean(1); v2c=v2.reshape(N,2).mean(1); w2c=w2.reshape(N,2).mean(1)
        h=L/N
        e=np.sqrt(h*((uN-u2c)**2+(vN-v2c)**2+(wN-w2c)**2).sum())
        conv.append((N,e))
    # residual on finest grid trajectory
    from solver import rhs_factory
    N=512; Rh,h=rhs_factory(N,L,pp)
    Y=sols[N].y; tt=t_eval
    res_u=res_v=res_w=0.0; cnt=0
    rlist=[]
    for j in range(1,len(tt)-1):
        DtU=(Y[:,j+1]-Y[:,j-1])/(tt[j+1]-tt[j-1])
        R=DtU-Rh(tt[j],Y[:,j])
        ru,rv,rw=split(R,N)
        nrm=lambda a:np.sqrt(h*(a**2).sum())
        rlist.append((nrm(ru),nrm(rv),nrm(rw)))
    rlist=np.array(rlist)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    Narr=[c[0] for c in conv]; earr=[c[1] for c in conv]
    order=np.polyfit(np.log(Narr),np.log(earr),1)[0]
    axes[0].loglog(Narr,earr,'C0-o',label='measured  (order %.2f)'%(-order))
    ref=earr[0]*(np.array(Narr,float)/Narr[0])**(-2)
    axes[0].loglog(Narr,ref,'k--',label=r'$O(h^{2})$ ref')
    axes[0].set(xlabel='N',ylabel=r'$\|U_h-U_{h/2}\|_{L^2}$',title='(a) grid convergence')
    axes[0].legend(fontsize=8)
    axes[1].semilogy(tt[1:-1],rlist[:,0],'C0',label=r'$\|R_u\|$')
    axes[1].semilogy(tt[1:-1],rlist[:,1],'C1',label=r'$\|R_v\|$')
    axes[1].semilogy(tt[1:-1],rlist[:,2],'C3',label=r'$\|R_w\|$')
    axes[1].set(xlabel='t',ylabel='residual norm',title='(b) PDE residuals (N=512)')
    axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f'{FIGDIR}/fig_exp6.pdf'); plt.close(fig)
    # observed order
    order=np.polyfit(np.log(Narr),np.log(earr),1)[0]
    RES['exp6']=dict(conv=[(int(n),float(e)) for n,e in conv],
                     observed_order=float(-order),
                     min_over_run=mins,
                     residual_max=[float(rlist[:,0].max()),float(rlist[:,1].max()),float(rlist[:,2].max())])
    print('EXP6',json.dumps(RES['exp6'],indent=0))

if __name__=='__main__':
    import sys
    which=sys.argv[1] if len(sys.argv)>1 else 'all'
    table={'1':exp1,'2':exp2,'3':exp3,'4':exp4,'5':exp5,'6':exp6}
    if which=='all':
        for f in table.values(): f()
    else:
        for ch in which: table[ch]()
    # merge into results.json
    try:
        old=json.load(open(f'{FIGDIR}/results.json'))
    except Exception:
        old={}
    old.update(RES)
    json.dump(old, open(f'{FIGDIR}/results.json','w'), indent=2)
    print('\nDONE', list(RES.keys()))
