"""
Frequency-resolved EEG leadfield: column-space / observability test.
Three-shell concentric-sphere head with COMPLEX Cole-Cole admittivity per band.
Answers: do per-band leadfields L(w) built from sigma*(w)=sigma(w)+i w eps span
different column spaces (a 'colored-filters' resolution gain)? Result: no --
bands reweight the same spherical-harmonic basis (near-collinear leadfields),
principal angles <7 deg over 1-100 Hz, zero resolvable-rank gain from stacking.
"""
import numpy as np
from scipy.linalg import subspace_angles

eps0 = 8.8541878128e-12

# Gabriel-1996 (part III) 4-term Cole-Cole parameters (as in the IT'IS/Hasgall DB).
# tissue: (eps_inf, [(delta_eps, tau_s, alpha) x4], sigma_ionic S/m)
cole_cole = {
 "GreyMatter":  (4.0, [(45,7.958e-12,0.10),(400,15.915e-9,0.15),(2.0e5,106.103e-6,0.22),(4.5e7,5.305e-3,0.0)], 0.02),
 "WhiteMatter": (4.0, [(32,7.958e-12,0.10),(100,7.958e-9,0.10),(4.0e4,53.052e-6,0.30),(3.5e7,7.958e-3,0.02)], 0.02),
 "CSF":         (4.0, [(65,7.958e-12,0.10),(40,1.592e-9,0.0),(0.0,159.155e-6,0.0),(0.0,5.305e-3,0.0)], 2.0),
 "BoneCortical":(2.5, [(10,13.263e-12,0.20),(180,79.577e-9,0.20),(5.0e3,159.155e-6,0.20),(1.0e5,15.915e-3,0.0)], 0.020),
 "Skin":        (4.0, [(39,7.958e-12,0.10),(280,79.577e-9,0.0),(3.0e4,1.592e-3,0.16),(3.0e4,1.592e-3,0.20)], 0.0002),
}

def admittivity(tissue, f):
    ef, disp, si = cole_cole[tissue]; w = 2*np.pi*f
    eps = ef + 0j
    for de,tau,al in disp:
        eps += de/(1+(1j*w*tau)**(1-al))
    return si + 1j*w*eps0*eps          # complex conductivity S/m

def shell_gain(n, sig, R, b):
    """Legendre-degree-n scalp-surface coefficient for a unit radial dipole at r=b
    in a concentric multi-shell sphere with complex conductivities sig, outer radii R."""
    L=len(sig); Rs=R[-1]
    M=np.zeros((2*L,2*L),dtype=complex); rhs=np.zeros(2*L,dtype=complex)
    def phi(k,r):
        row=np.zeros(2*L,dtype=complex); row[2*k]=r**n; row[2*k+1]=r**(-(n+1)); return row
    def cur(k,r):
        row=np.zeros(2*L,dtype=complex)
        row[2*k]=sig[k]*n*r**(n-1); row[2*k+1]=-sig[k]*(n+1)*r**(-(n+2)); return row
    eq=0
    M[eq,1]=1.0; eq+=1                                    # exclude singular r^-(n+1) in core
    c_n=n*b**(n-1)/(4*np.pi*sig[0])                       # radial-dipole particular coeff
    for k in range(L-1):
        r=R[k]
        M[eq]=phi(k,r)-phi(k+1,r);   rhs[eq]=-(c_n*r**(-(n+1)) if k==0 else 0.0); eq+=1
        M[eq]=cur(k,r)-cur(k+1,r);   rhs[eq]=-(sig[0]*(-(n+1))*c_n*r**(-(n+2)) if k==0 else 0.0); eq+=1
    M[eq]=cur(L-1,Rs); eq+=1
    sol=np.linalg.solve(M,rhs); A=sol[0::2]; B=sol[1::2]
    return A[L-1]*Rs**n + B[L-1]*Rs**(-(n+1))

def fib_sphere(K):
    i=np.arange(K); phi=(1+5**0.5)/2
    z=1-2*(i+0.5)/K; th=np.arccos(z); ph=2*np.pi*i/phi
    return np.c_[np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), z]

def leadfield(freq, tissues, R, b, elec, src, Nmax=40, static=False):
    sig=[ (admittivity(t,freq).real+0j) if static else admittivity(t,freq) for t in tissues ]
    g=np.array([shell_gain(n,sig,R,b) for n in range(1,Nmax+1)])
    cg=np.clip(elec@src.T,-1,1); Lmat=np.zeros(cg.shape,dtype=complex)
    Pnm1=np.ones_like(cg); Pn=cg.copy()
    for n in range(1,Nmax+1):
        Lmat += g[n-1]*(2*n+1)/(4*np.pi)*Pn
        Pnm1,Pn = Pn, ((2*n+1)*cg*Pn-n*Pnm1)/(n+1)
    return Lmat

if __name__=="__main__":
    R=[0.080,0.086,0.092]; b=0.075
    tissues=("GreyMatter","BoneCortical","Skin")
    elec=fib_sphere(64); src=fib_sphere(200)
    freqs=[1,4,8,10,20,40,60,80,100]
    L={f:leadfield(f,tissues,R,b,elec,src) for f in freqs}
    Ls={f:leadfield(f,tissues,R,b,elec,src,static=True) for f in freqs}
    dom=lambda A,r: np.linalg.svd(A,full_matrices=False)[0][:,:r]
    Uref=dom(L[10],20)
    print("freq  ang_vs10Hz  ang_complex_vs_static")
    for f in freqs:
        a1=np.degrees(subspace_angles(Uref,dom(L[f],20))).max()
        a2=np.degrees(subspace_angles(dom(L[f],20),dom(Ls[f],20))).max()
        print(f"{f:>4}  {a1:8.2f}  {a2:8.2f}")
    emb=lambda A: np.vstack([A.real,A.imag])
    single=emb(L[10]); stack=np.vstack([emb(L[f]) for f in freqs])
    for snr in [20,30,40,60]:
        s1=np.linalg.svd(single,compute_uv=False); sk=np.linalg.svd(stack,compute_uv=False)
        r1=int(np.sum(s1>s1[0]*10**(-snr/20))); rk=int(np.sum(sk>sk[0]*10**(-snr/20)))
        print(f"SNR={snr}dB  single={r1}  9-band={rk}  gain={rk-r1}")
