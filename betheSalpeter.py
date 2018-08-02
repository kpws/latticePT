import pylab as pl
import dqmc
import math	
import numpy as np

nx=2
ny=2
tx=1
ty=.1
tne=0
tnw=0
U=8
mu=0
beta=1

m=6
nTauPerBeta=8

nTau=dqmc.getNTau(beta, nTauPerBeta, m)

nSamplesPerRun=200#measurePeriod*20

# clean=True
# if clean:
# 	import os
# 	folder = dqmc.getDirName(nTau,nx,ny,tx,ty,tnw,tne,U,mu,beta,m)
# 	for f in os.listdir(folder):
# 		file_path = os.path.join(folder, f)
# 		if os.path.isfile(file_path):
# 			os.unlink(file_path)

dqmc.genGSamples(nx,ny,tx,ty,tnw,tne,U,mu,beta,m,nTauPerBeta,nSamplesPerRun,nRuns=20,nThreads=1,startSeed=0)

cl=nSamplesPerRun-1
runs=dqmc.loadRuns(nTau,nx,ny,tx,ty,tnw,tne,U,mu,beta,m)

ac=np.array(np.mean([dqmc.getAutocorrelator(r,cl) for r in runs],axis=0))

# pl.plot(ac[:,0,0,0,0,0])
pl.plot(np.mean(ac,axis=(1,2,3,4,5)))

g,g_err=dqmc.averageOverG(runs,lambda g:dqmc.symmetrizeG(g,nTau,nx,ny),4)

print('***Exact Diagonalization***')
import ed
taus=[taui/nTau*beta for taui in range(nTau)]
ed_g=ed.getG(nx,ny,tx,ty,tnw,tne,U,mu,beta,taus,eps=1e-2)

pl.figure()
colors=[u'b', u'g', u'r', u'c', u'm', u'y', u'k']
for x in range(nx):
	for y in range(ny):
		c=colors[(x+nx*y)%len(colors)]
		pl.plot(taus, ed_g[:,x,y],linestyle='--',color=c,label=r"$G_{{\mathrm{{ED}}}}({x},{y})$".format(x=x,y=y))
		pl.errorbar(taus, g[:,x+nx*y], yerr=g[:,x+nx*y],color=c,linestyle='-',label=r"$G_{{\mathrm{{DQMC}}}}({x},{y})$".format(x=x,y=y))
pl.ylim([-1,1])
pl.xlabel(r"$\tau$")
pl.ylabel(r"$G(x,y)$")
pl.legend()
pl.show()

# pl.figure()
# pl.plot(g[0,0,:,0,:])
# pl.plot(gsym[:,:])

pl.show()