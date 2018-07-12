import pylab as pl
import dqmc
import ed
import math


nx=2
ny=2
tx=1
ty=1/2
U=4
mu=0
beta=2

m=8
tausPerBeta=16

nTau=math.ceil(beta*tausPerBeta/m)*m

nThreads=1  #multithreading doesn't work on mac, (wtf) https://stackoverflow.com/questions/9879371/segfault-using-numpys-lapack-lite-with-multiprocessing-on-osx-not-linux)
nSamples=10
nWarmupSweeps=nTau*20
nSweepsPerThread=30*nWarmupSweeps

taus=[taui/nTau*beta for taui in range(nTau)]

print('***Exact Diagonalization***')
ed_g=ed.getG(nx,ny,tx,ty,U,mu,beta,taus,eps=0)

print('***Determinant Quantum Monte Carlo***')
dqmc_g, dqmc_gerr=dqmc.getG(nx,ny,tx,ty,U,mu,beta,m,tausPerBeta,nThreads,nWarmupSweeps,nSweepsPerThread,nSamples=nSamples)

colors=[u'b', u'g', u'r', u'c', u'm', u'y', u'k']
for x in range(nx):
	for y in range(ny):
		c=colors[(x+nx*y)%len(colors)]
		pl.plot(taus, ed_g[:,x,y],linestyle='--',color=c,label=r"$G_{{\mathrm{{ED}}}}({x},{y})$".format(x=x,y=y))
		pl.errorbar(taus, dqmc_g[:,x,y], yerr=dqmc_gerr[:,x,y],color=c,linestyle='-',label=r"$G_{{\mathrm{{DQMC}}}}({x},{y})$".format(x=x,y=y))
		# pl.errorbar(taus, dqmc_ns_g[:,x,y], yerr=dqmc_ns_gerr[:,x,y],linestyle='--',color='b')
pl.ylim([-1,1])
pl.xlabel(r"$\tau$")
pl.ylabel(r"$G(x,y)$")
pl.legend()
pl.show()