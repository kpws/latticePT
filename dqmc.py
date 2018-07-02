import numpy as np
import random
import scipy
from scipy import linalg
from numpy import linalg
from functools import reduce

def calcBs(Kexp,lamb,deltaTau,state,mu,N,ntau):
	return [[Kexp@np.diag([np.exp( lamb*sig*state[li][i] + deltaTau*mu ) for i in range(N)]) for li in range(ntau)] for sig in [-1,1]]

def calcGFromScratch(Kexp,lamb,state,tau,deltaTau,mu,N,ntau,B):
	stabilize=False
	if stabilize:
		mB=[[reduce(lambda a,b:a@b,(B[sig][(tau-(j+1)*m+i+1)%ntau] for i in range(m))) for j in range(ntauOverm)] for sig in range(2)]
		for mb in mB:
			U,S,VH=numpy.linalg.svd(mb)
	else:
		Ml=[B[0][(tau+1)%ntau], B[1][(tau+1)%ntau]]
		for il in range(1,ntau):
			for sigi in range(2):
				Ml[sigi]=Ml[sigi] @ B[sigi][(tau+1+il)%ntau]

	return [np.linalg.inv(np.eye(N,N)+Ml[sigi]) for sigi in range(2)]

def dqmc(nx,ny,tx,ty,U,mu,beta,ntauOverm,m,seed,nWarmupSweeps,nSweeps,observables,observablesTD=[]):
	rand=random.Random()
	rand.seed(seed)
	
	ntau=ntauOverm*m
	deltaTau=beta/ntau

	lamb=np.arccosh(np.exp(U*deltaTau/2))
	N=nx*ny
	K=np.zeros((N,N))
	sigs=[-1,1]
	
	for x in range(nx):
		for y in range(ny):
			xp=(x-1)%nx
			xn=(x+1)%nx
			yp=(y-1)%ny
			yn=(y+1)%ny
			here=x+nx*y
			K[here, xp%nx+nx*y]-=tx
			K[here, xn%nx+nx*y]-=tx
			if ny>1:
				K[here, x%nx+nx*yp]-=ty
				K[here, x%nx+nx*yn]-=ty

	Kexp=scipy.linalg.expm(-deltaTau*K)
	
	state=[np.array([rand.choice([-1,1]) for i in range(N)]) for tau in range(ntau)]
	res=[0 for o in observables]
	resTD=[0 for o in observablesTD]

	for sweep in range(nWarmupSweeps+nSweeps):
		B=calcBs(Kexp,lamb,deltaTau,state,mu,N,ntau)
		g=calcGFromScratch(Kexp,lamb,state,0,deltaTau,mu,N,ntau,B)
		gTimeDep=np.zeros((2,ntau,nx*ny,nx*ny))
		for tau in range(ntau):
			# B=calcBs(Kexp,lamb,deltaTau,state,mu,N,ntau)
			# g=calcGFromScratch(Kexp,lamb,state,tau,deltaTau,mu,N,ntau,B)
			for u in range(nx*ny):
				x=rand.randrange(nx)
				y=rand.randrange(ny)
				ii=x+nx*y
				
				delta=[np.exp(-2*sigs[sigi]*lamb*state[tau][ii])-1 for sigi in range(2)]
				R=[1+(1-g[sigi][ii,ii])*delta[sigi] for sigi in range(2)]
				Rprod=R[0]*R[1]
				p=abs(Rprod/(1+Rprod))
				# print(p)
				if p>1 or rand.random()<p:
					state[tau][ii]*=-1
					# g=calcGFromScratch(Kexp,lamb,state,tau,deltaTau,mu,N,ntau)
					for sigi in range(2):
						g[sigi]-=np.outer([1 if ii==i else 0 for i in range(N)]-g[sigi][::,ii],g[sigi][ii,::]*delta[sigi])/R[sigi]
			# for sigi in range(2):
				# B[sigi][tau]=Kexp@np.diag([np.exp(lamb*sigs[sigi]*state[tau][i]+deltaTau*mu) for i in range(N)])
			
			if sweep>=nWarmupSweeps:
				for oi in range(len(observables)):
					res[oi]+=observables[oi](g)
				if len(observablesTD)!=0:
					for sigi in range(2):
						gTimeDep[sigi][0]=g[sigi]
						# https://link.springer.com/content/pdf/10.1007%2F978-1-4613-0565-1.pdf
						for i in range(1,ntau):
							gTimeDep[sigi][i]=gTimeDep[sigi][i-1]@B[sigi][(tau+i)%ntau]
					for oi in range(len(observablesTD)):
						resTD[oi]+=observablesTD[oi](gTimeDep)
				
			# B=calcBs(Kexp,lamb,deltaTau,state,mu,N,ntau)
			# g=calcGFromScratch(Kexp,lamb,state,(tau+1)%ntau,deltaTau,mu,N,ntau,B)
			g=[ np.linalg.inv(B[sigi][(tau+1)%ntau])@g[sigi]@B[sigi][(tau+1)%ntau] for sigi in range(2)]
		if False:
			for y in range(ny):
				print(reduce(lambda a,b:a+b,('O' if state[0][x+nx*y]==-1 else '*' for x in range(nx))))
			print()
		
		# if sweep%10==0:
		#  	print([r/(sweep+1) for r in res])
			
	if len(observablesTD)!=0:
		return ([r/nSweeps/ntau for r in res],[r/nSweeps/ntau for r in resTD])
	else:
		return [r/nSweeps/ntau for r in res]

def main():
	nx=2
	ny=2
	U=4
	tx=1
	ty=1
	beta=1/tx #2/tx
	m=8
	tausPerBeta=8 #8 recommended in literature
	mu=0

	ntau=(int(beta*tausPerBeta)//m)*m

	nThreads=8
	nWarmupSweeps=4*100
	nSweepsPerThread=4*650

	# ops=[(lambda g:g[0]),(lambda g:g[1]),(lambda g:2-g[0][0,0]-g[1][0,0])]
	opnames=["<n>","<nn>"]
	ops=[(lambda g:2-g[0][0,0]-g[1][0,0]),(lambda g:1-g[0][0,0]-g[1][0,0]+g[0][0,0]*g[1][0,0])]

	opnames=["<g{i}>".format(i=i) for i in range(nx)]
	ops=[lambda g,i=i:g[0][0,i] for i in range(nx)]

	opnames=["g"]
	ops=[lambda g:g[0]]
	# ops=[lambda g:np.transpose(np.reshape(g[0][0,:],(ny,nx)))]

	import multiprocessing
	import time
	def work(seed,return_dict):
		ret=dqmc(nx,ny,tx,ty,U,mu,beta,int(beta*tausPerBeta)//m,m,seed,nWarmupSweeps,nSweepsPerThread,[],observablesTD=ops)
		return_dict[seed] = ret

	manager = multiprocessing.Manager()
	return_dict = manager.dict()
	job=[multiprocessing.Process(target=work, args=(i,return_dict)) for i in range(nThreads)]
	startTime=time.time()
	print("Starting {n} jobs...".format(n=nThreads))
	for t in job: t.start()
	print("Waiting for {n} jobs...".format(n=nThreads))
	for t in job: t.join()
	print("Time per sweep: {time:.2f} ms".format(time=1000*(time.time()-startTime)/(nSweepsPerThread+nWarmupSweeps)/nThreads))
	print("Operators:")
	res=return_dict.values()
	print(len(res))
	for i in range(len(ops)):
		#res[t][1][i] - time dep part
		r=[res[t][1][i] for t in range(nThreads)]
		mean=np.mean(r,axis=0)
		std=np.std(r,axis=0)
		if opnames[i]=="g":
			g=np.zeros((ntau,nx,ny))
			for tau in range(ntau):
				for x in range(nx):
					for y in range(ny):
						for dx in range(nx):
							for dy in range(ny):
								g[tau,x,y]+=mean[tau,dx+dy*nx,(dx+x)%nx+((dy+y)%ny)*nx]/(nx*ny)
			print(g)
			# print("{name} = {mean} ± {std}".format(name=opnames[i],mean=mean,std=std))
		else:
			ri=[res[t][i] for t in range(nThreads)]
			mean=np.mean(ri,axis=0)
			std=np.std(ri,axis=0,ddof=1)/np.sqrt(nThreads)
			print("{name} = {mean} ± {std}".format(name=opnames[i],mean=mean,std=std))


if __name__ == "__main__":
	main()