import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # run from anywhere
import numpy as np, torch, time
from compgcn.ilr import *
from compgcn.data import *
from compgcn.models import CGCNLayer
from compgcn.train import fit, DEFAULT_HP
rng=np.random.default_rng(0)
# 1) ilr round trip + isometry
x=rng.dirichlet([1,2,3,4],5); assert np.allclose(ilr_inv(ilr(x)),x)
a,b=x[0],x[1]; aitch=np.sum((clr(a)-clr(b))**2); assert np.isclose(aitch,np.sum((ilr(a)-ilr(b))**2))
P=random_basis(4,rng); assert np.allclose(P@P.T,np.eye(3)) and np.allclose(P.sum(1),0)
# 2) CGCN layer == Eq.(4)+(5): ilr^-1(A ilr(x_j)) then Aitchison inner product with w_j
d=synthetic(n=200,p=3,D=3,seed=1); A=norm_adj(d.edges,d.n)
layer=CGCNLayer(d.D); Z=torch.tensor(block_ilr(d.X,d.D),dtype=torch.float32)
H=layer(Z,A).detach().numpy()
Ad=A.to_dense().numpy(); w=layer.w.detach().numpy()
for j in range(3):
    hj=ilr_inv(Ad@ilr(d.X[:,3*j:3*j+3]))            # back on the simplex
    wj=ilr_inv(w[2*j:2*j+2])                          # weight as a composition
    ref=(clr(hj)*clr(wj)).sum(1)                      # Aitchison inner product
    assert np.allclose(H[:,j],ref,atol=1e-4), j
print("ilr + CGCN equivalence checks passed")
print("SNR check", {k:round(snr(*v),2) for k,v in SNR_SETTINGS.items()}, {k:round(snr(*v,C=k),2) for k,v in MULTICLASS.items()})
# 3) timing
d=synthetic(n=1000,p=10,a=245,b=205,seed=0); A=norm_adj(d.edges,d.n)
sp=split_per_class(d.y,30,40,np.random.default_rng(0))
for m in ["COMP-GCN","GCN-scalar","GCN-ilr","linear-ilr"]:
    t=time.time(); r=fit(m,d,A,sp); print(m,r,f"{time.time()-t:.1f}s")
