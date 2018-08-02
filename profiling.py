import dqmc
import math

nx=10
ny=10
tx=1
ty=.1
tne=.1
tnw=0
U=8
mu=0
beta=32

m=6
tausPerBeta=8

nTau=math.ceil(beta*tausPerBeta/m)*m

seed=0
nSamples=3

g,warmup,total=dqmc.dqmc(nx,ny,tx,ty,tne,tnw,U,mu,beta,nTau//m,m,seed,0,True,
	[],observablesTD=[],stabEps=1e-4,autoCorrN=0,profile=True,returnState=False,startState=None,measurePeriod=100,saveSamples=None,nWarmupSweeps=nSamples)
print("Time {t:.1f} ms".format(t=warmup*1000/nSamples))

#Time 13987.3 ms
#Time 13773.5 ms

# with partial products: