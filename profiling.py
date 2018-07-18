import dqmc
import math

nx=16
ny=16
tx=1
ty=.1
tne=0
tnw=0
U=4
mu=0
beta=32

m=8
tausPerBeta=8

nTau=math.ceil(beta*tausPerBeta/m)*m

seed=0
nSamples=10

g,warmup,total=dqmc.dqmc(nx,ny,tx,ty,tne,tnw,U,mu,beta,nTau//m,m,seed,nSamples,0,True,
	[],observablesTD=[],stabEps=1e-4,autoCorrN=0,profile=True,returnState=False,startState=None,measurePeriod=100,saveSamples=None)