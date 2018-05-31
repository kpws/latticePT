from greensFunction import *
import numpy as np

nx=2
ny=2
U=2
tx=1
ty=1 if ny>1 else 0
t2=0
beta=2/tx
tausPerBeta=8
mu=0
nc=int(tausPerBeta//beta)
nc=10240

G=greensFunction(-1,1/beta).fromFun(nc,nx,ny,lambda omega,kx,ky:1/(1j*omega-(-2*tx*np.cos(kx)-2*ty*np.cos(ky)-2*t2*np.cos(kx+ky)-mu)))
# G=greensFunction(-1,1/beta).fromFun(nc,nx,ny,lambda omega,kx,ky:kx)
print(-G.real()/2/nx/ny)
# print(G.real()*(2*np.pi**2))


chi0=-G*(G.reverse())
chis=chi0/(1-U*chi0)
chic=chi0/(1+U*chi0)
Vsa=U+(3./2)*(U**2)*chis-(1./2)*(U**2)*chic
Vta=-1./2*(U**2)*chis-1./2*(U**2)*chic
Vn=(3./2)*(U**2)*chis+(1./2)*(U**2)*chic-(U**2)*chi0

Sigma=Vn*G
G2=(G.inv()-Sigma).inv()

print("2nd order perturbation theory:")
print(G2.real()/2/nx/ny)