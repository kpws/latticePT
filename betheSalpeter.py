import pylab as pl
import dqmc
import math

nx=10
ny=10
tx=1
ty=.1
tne=.1
tnw=0
U=4
mu=0
beta=8

m=8
tausPerBeta=8

nTau=math.ceil(beta*tausPerBeta/m)*m

measurePeriod=100
nSweepsPerRun=1#measurePeriod*20

# clean=True
# if clean:
# 	import os
# 	folder = dqmc.getDirName(nx,ny,nTau,tx,ty,tnw,tne,U,mu,beta,m,measurePeriod)
# 	for f in os.listdir(folder):
# 		file_path = os.path.join(folder, f)
# 		if os.path.isfile(file_path):
# 			os.unlink(file_path)

dqmc.genGSamples(nx,ny,tx,ty,tnw,tne,U,mu,beta,m,tausPerBeta,nSweepsPerRun,measurePeriod,nRuns=1,nThreads=1,startSeed=0)

runs=dqmc.loadRuns(nx,ny,nTau,tx,ty,tnw,tne,U,mu,beta,m,measurePeriod)

print(runs)

taus=[taui/nTau*beta for taui in range(nTau)]

def linOp(v):
	averageOverG(runs,op)
	vout=np.zeros()
	for g in gs:
		vout+=g*(g*v)
	return vout


