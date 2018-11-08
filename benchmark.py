import time as time
import subprocess
import os
import sys
from dqmc import dqmc, getNTau

nx=32
ny=16
tx=1
ty=tx
tne=0
tnw=0
U=4
mu=0
beta=16
m=2
tausPerBeta=8
ntau=getNTau(beta, tausPerBeta, m)
ntauOverm=8
seed=0
nSweeps=2
stabilize=True
stabEps=1e-2
observables=[lambda g:1]

if len(sys.argv)==1:
	import pickle
	if False:
		ns=list(range(1,33))
		ts=[]
		for n in ns:
			ts.append(float(subprocess.check_output(["python","benchmark.py",str(n)],env=os.environ if n==0 else dict(os.environ, OMP_NUM_THREADS=str(n))).strip().decode('ascii')))
		pickle.dump((ns,ts),open("cache/benchmark","wb"))
	else:
		ns,ts=pickle.load(open("cache/benchmark","rb"))
	print(ts)
	import matplotlib as mpl
	import numpy as np
	mpl.use('Agg')
	from matplotlib import rc
	rc('text', usetex=True)
	import matplotlib.pyplot as pl
	pl.plot(ns,nSweeps/np.array(ts),marker='.',color='black')
	pl.grid()
	pl.xlim([0,ns[-1]])
	pl.ylim([0,nSweeps/np.min(ts)])
	pl.xlabel("$n_{\mathrm{threads}}$")
	pl.ylabel("$n_{\mathrm{sweeps}}/t\\ [\mathrm{s}^{-1}]$")
	pl.savefig("plots/benchmark.pdf")
else:
	n=int(sys.argv[1])
	start=time.time()
	
	dqmc(nx,ny,tx,ty,tne,tnw,U,mu,beta,ntauOverm,m,seed,nSweeps,stabilize,
    	observables,observablesTD=[],measurePeriod=1,saveSamples=None,nWarmupSweeps=0,showProgress=False,stabEps=stabEps)

	end=time.time()
	print(end-start)
	exit(0)
