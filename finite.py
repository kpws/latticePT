import numpy as np

#we represent a product state by a set of numbers corresponding to applications of cdaggger on the vacuum. Each number is an index representing spin, x, y, ... The canonical representation has these in increasing order

class ProdState:
	def __init__(self,amp,state):
		self.state=[s for s in state]
		self.amp=amp

	def c(self,i):
		pos=-1
		for j in range(len(self.state)):
			if i==self.state[j]:
				pos=j
				break
		if pos==-1:
			return ProdState(0,[])
		else:
			return ProdState(self.amp*(1 if pos%2==0 else -1),self.state[:pos]+self.state[pos+1:])

	def cd(self,i):
		return ProdState(self.amp,self.state+[i])

	def canonicalize(self):
		l=self.state
		parity=1
		for i in range(1,len(l)):
			currentvalue = l[i]
			position = i

			while position>0 and l[position-1]>currentvalue:
				l[position]=l[position-1]
				position = position-1
			
			l[position]=currentvalue
			if (position-i)%2==1:
				parity=-parity
		for i in range(1,len(l)):
			if l[i-1]==l[i]:
				self.amp=0
				self.state=[]
				return
		self.state=l
		self.amp*=parity

	def toEquivClass(self,group,repr):
		for i in range(len(group)):
			nstate=group[i](self.state)
			if nstate<self.state:
				self.state=nstate
				self.amp/=repr[i]

	def same(self,other):
		if len(self.state)!=len(other.state):
			return False
		for i in range(len(self.state)):
			if self.state[i]!=other.state[i]:
				return False
		return True

	def __str__(self):
		return str(self.amp)+"*"+str(self.state)

	def __repr__(self):
		return str(self)

class MultiState:
	def __add__(self,other):
		return 1

#visa applicaiton STH2180329AL5442801

def coordToi(sys,coords):
	n=1
	i=0
	for j in range(len(sys)):
		i+=n*coords[j]
		n*=sys[j]
	return i

def iToCoord(sys,i):
	n=1
	coords=[]
	for j in range(len(sys)):
		coords.append(i%sys[j])
		i//=sys[j]
	return coords

def addToMultiState(ms,ps):
	if ps.amp==0:
		return
	ps.canonicalize()
	for i in range(len(ms)):
		if ms[i].same(ps):
			ms[i].amp+=ps.amp
			if ms[i].amp==0:
				ms.pop(i)
			return
	ms.append(ps)

def symmetrize(sys,psi,group,rep):
	out=[]
	for p in psi:
		for i in range(len(group)):
			p2=ProdState(p.amp*rep[i],[coordToi(sys,group[i](iToCoord(sys,s))) for s in p.state])
			addToMultiState(out,p2)
	return out

def inner(a,b):
	acc=0
	for i in a:
		for j in b:
			if i.same(j):
				acc+=np.conj(i.amp)*j.amp
	return acc
def fac(n):
	return 1 if n==0 else n*fac(n-1)
def binom(a,b):
	return fac(a)//fac(a-b)//fac(b)

def main():
	nspins=2
	nx=8
	ny=1
	sys=[nspins,nx,ny]
	inverseFilling=2

	#We have several indep symmetries, electron number, total spin, momentum, two mirrors
	nstates=nspins*nx*ny

	ne=nstates//inverseFilling

	vac=ProdState(1,[])
	psi=ProdState(vac.amp,vac.state)
	# for i in range(ne):
	# 	start=start.cd(i)

	# from numpy.random import choice
	# l=choice(range(nstates),ne,False)
	# for i in l:
	# 	psi=psi.cd(i)
	
	for i in range(0,nx,2):
		psi=psi.cd(coordToi(sys,[0,i,0]))
		# if i!=0:
		psi=psi.cd(coordToi(sys,[1,(i+1)%nx,0]))
	nSubStates=binom(nx*ny,nx*ny//2)**2
	
	# for i in range(nx):
	# 	for j in range(ny//2):
	# 		jj=2*j+i%2
	# 		psi=psi.cd(coordToi(sys,[0,i,jj]))
	# 		psi=psi.cd(coordToi(sys,[1,(i+1)%nx,jj]))

	# nSubStates=binom(nx*ny,nx*ny//2)**2
	# nup=4
	# l=choice(range(nx*ny),nup,False)
	# for i in l:
	# 	start=start.cd(coordToi(sys,[0,]))
	
	psi.canonicalize()
	print(psi)


	# apply symmetry transfs
	# spin flip
	psi=[psi]
	print(psi)
	
	identity=lambda x:x
	#psi=symmetrize(sys,psi,[identity,lambda x:[1-x[0],x[1],x[2]]],[1,1])
	# psi=symmetrize(sys,psi,[lambda x,i=i:[x[0],(x[1]+i)%nx,x[2]] for i in range(nx)],[1 for i in range(nx)])
	# psi=symmetrize(sys,psi,[lambda x,i=i:[x[0],x[1],(x[2]+i)%ny] for i in range(ny)],[1 for i in range(ny)])
	# psi=symmetrize(sys,psi,[identity,lambda x:[x[0],x[2],x[1]] ],[1 ,1])
	# psi=symmetrize(sys,psi,[identity,lambda x:[x[0],nx-x[1]-1,x[2]] ],[1 ,1])
	print(psi)

	tx=1
	ty=0
	U=1.7
	shift=-3
	krylov=[psi]
	nold=1
	for it in range(30):
		newPsi=[]
		for p in psi:
			p2=ProdState(shift*p.amp,p.state)
			addToMultiState(newPsi,p2)
			for ix in range(nx):
				for iy in range(ny):
					i0u=coordToi(sys,[0,ix,iy])
					i0d=coordToi(sys,[1,ix,iy])
					p2=p.c(i0d).cd(i0d).c(i0u).cd(i0u)
					p2.amp*=U
					addToMultiState(newPsi,p2)

					for ialpha in range(nspins):
						i0=coordToi(sys,[ialpha,ix,iy])
						ix1=coordToi(sys,[ialpha,(ix+1)%nx,iy])
						iy1=coordToi(sys,[ialpha,ix,(iy+1)%ny])

						p2=p.c(i0).cd(ix1)
						p2.amp*=tx
						addToMultiState(newPsi,p2)
						p2=p.c(ix1).cd(i0)
						p2.amp*=tx
						addToMultiState(newPsi,p2)

						p2=p.c(i0).cd(iy1)
						p2.amp*=ty
						addToMultiState(newPsi,p2)
						p2=p.c(iy1).cd(i0)
						p2.amp*=ty
						addToMultiState(newPsi,p2)

		print("before: "+str(len(newPsi))+"/"+str(nSubStates))

		# for k in krylov:
		# 	overlap=inner(k,newPsi)
		# 	for ik in k:
		# 		p2=ProdState(-ik.amp*overlap,ik.state)
		# 		addToMultiState(newPsi,p2)
		# print("after: "+str(len(newPsi))+"/"+str(nSubStates))
		n=inner(newPsi,newPsi)
		if it>20:
			for deltaX in range(nx):
				x1=0
				x1p=1
				y1=0
				x2=(x1+deltaX)%nx
				x2p=(x1+deltaX+1)%nx
				y2=0
				i1u=coordToi(sys,[0,x1,y1])
				i1d=coordToi(sys,[1,x1,y1])
				i2u=coordToi(sys,[0,x2,y2])
				i2d=coordToi(sys,[1,x2,y2])

				i1pu=coordToi(sys,[0,x1p,y1])
				i1pd=coordToi(sys,[1,x1p,y1])
				i2pu=coordToi(sys,[0,x2p,y2])
				i2pd=coordToi(sys,[1,x2p,y2])
				pairpair=0
				ppairppair=0
				densdens=0
				uddensdens=0
				for p2 in newPsi:
					s2=p2.c(i2u).c(i2d).cd(i1u).cd(i1d)
					s2.canonicalize()
					pairpair+=inner(newPsi,  [s2])

					s2=p2.c(i2u).c(i2pd).cd(i1u).cd(i1pd)
					s2.canonicalize()
					ppairppair+=inner(newPsi,  [s2])

					s2=p2.c(i2u).cd(i2u).c(i1u).cd(i1u)
					s2.canonicalize()
					densdens+=inner(newPsi,  [s2])

					s2=p2.c(i2u).cd(i2u).c(i1d).cd(i1d)
					s2.canonicalize()
					uddensdens+=inner(newPsi,  [s2])
						# print(pairpair)
				print("Dx="+str(deltaX)+", s-pair s-pair: "+str(pairpair/n))
				print("Dx="+str(deltaX)+", p-pair p-pair: "+str(ppairppair/n))
				print("Dx="+str(deltaX)+",     dens dens: "+str(densdens/n))
				print("Dx="+str(deltaX)+",   udens ddens: "+str(uddensdens/n))
				print()

		
		lam=inner(psi,newPsi)

		assert(lam<0)
		print("eig: "+str(lam-shift))

		for p in newPsi:
			p.amp/=np.sqrt(n)

		# krylov.append(newPsi)

		# print(str(len(newPsi))+"/"+str(2**nstates))
		
		# print(newPsi)
		# n=inner(newPsi,newPsi)
		
		# print(n)
		psi=newPsi

	# for k1 in krylov:
	# 	for k2 in krylov:
	# 			print(inner(k1,k2))

	#1, create state with finite electron number that obeys all symmetries. This is possibly nontrivial
	#1b, apply symmetry operators and verify it is an eigenstate and obtain eigenvalues
	#2, apply hamiltonian repeatedly to generate all eigenvectors and eigenvalues in the subspace of eigenstates with correct symmetries
	#3 check that symmetries are still preserved? Where might leakage come from?
	#go to different electron number/spin/momentum/... check orthogonality
	#this way we get e.g. different groundstate energy as a function of electron number

	# we want a basis that are eigenstates of as many symmetries as possible. That way when we apply H we don't expand..
	# consider product of momentum eigenstates

	# psi(1) is picked
	# psi(k+1)=H psi(k)

if __name__ == "__main__":
	main()

