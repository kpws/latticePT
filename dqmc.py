import numpy as np
import random
import scipy
from scipy import linalg
from numpy import linalg
import pickle
import os.path
from functools import reduce
			
def dqmc(nx,ny,tx,ty,U,mu,beta,ntauOverm,m,seed,nSweeps,observables):
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

	for sweep in range(nSweeps):
		for tau in range(ntau):
			for u in range(nx*ny):
				x=rand.randrange(nx)
				y=rand.randrange(ny)
				ii=x+nx*y

				B=[[Kexp@np.diag([np.exp(lamb*sig*state[li][i]+deltaTau*mu) for i in range(N)]) for li in range(ntau)] for sig in sigs]

				stabilize=False
				if stabilize:
					mB=[[reduce(lambda a,b:a@b,(B[sig][(tau-(j+1)*m+i+1)%ntau] for i in range(m))) for j in range(ntauOverm)] for sig in range(2)]
					for mb in mB:
						U,S,VH=numpy.linalg.svd(mb)
				else:
					Ml=[B[0][(tau+1)%ntau], B[1][(tau+1)%ntau]]
					for il in range(1,ntau):
						for sig in range(2):
							Ml[sig]=Ml[sig] @ B[sig][(tau+1+il)%ntau]

				g=[np.linalg.inv(np.eye(N,N)+Ml[sigi]) for sigi in range(2)]
				
				# print(g[0])
				# print(np.eye(N,N)+Ml[0])
				# print(g[0]@(np.eye(N,N)+Ml[0]))
				# return
				R=[1+(1-g[sigi][ii,ii])*(np.exp(-2*sigs[sigi]*lamb*state[tau][ii])-1) for sigi in range(2)]
				Rprod=R[0]*R[1]
				p=abs(Rprod/(1+Rprod))
				# print(p)
				if p>1 or rand.random()<p:
					state[tau][ii]*=-1
			if sweep==0:
				res=[o(g) for o in observables]
			else:
				for oi in range(len(observables)):
					res[oi]+=observables[oi](g)
		if False:
			for y in range(ny):
				print(reduce(lambda a,b:a+b,('O' if state[0][x+nx*y]==-1 else '*' for x in range(nx))))
			print()
		
		# if sweep%10==0:
		#  	print([r/(sweep+1) for r in res])
			
	return [r/nSweeps/ntau for r in res]

def main():
	nx=2
	ny=1
	U=4
	tx=1
	ty=1
	beta=2/tx
	m=8
	tausPerBeta=16 #8 recommended in literature
	mu=0

	# ops=[(lambda g:g[0]),(lambda g:g[1]),(lambda g:2-g[0][0,0]-g[1][0,0])]
	opnames=["<n>","<nn>"]
	ops=[(lambda g:2-g[0][0,0]-g[1][0,0]),(lambda g:1-g[0][0,0]-g[1][0,0]+g[0][0,0]*g[1][0,0])]

	nThreads=8
	import threading
	class Worker (threading.Thread):
		def __init__(self, seed):
			threading.Thread.__init__(self)
			self.seed=seed
		def run(self):
			self.res=dqmc(nx,ny,tx,ty,U,mu,beta,int(beta*tausPerBeta)//m,m,self.seed,1500,ops)
	threads=[Worker(i) for i in range(nThreads)]
	for t in threads: t.start()
	for t in threads: t.join()
	for i in range(len(ops)):
		res=[t.res[i] for t in threads]
		mean=np.mean(res,axis=0)
		std=np.std(res,axis=0,ddof=1)/np.sqrt(nThreads)
		print(f"{opnames[i]} = {mean} ± {std}")


if __name__ == "__main__":
	main()