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

def c2i(sys,coords):
	n=1
	i=0
	for j in range(len(sys)):
		i+=n*coords[j]
		n*=sys[j]
	return i

def i2c(sys,i):
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

	if tuple(ps.state) in ms:
		ms[tuple(ps.state)]+=ps.amp
		if ms[tuple(ps.state)]==0:
			ms.pop(tuple(ps.state))
	else:
		ms[tuple(ps.state)]=ps.amp

def symmetrize(sys,psi,group,rep):
	out=[]
	for p in psi:
		for i in range(len(group)):
			p2=ProdState(p.amp*rep[i],[c2i(sys,group[i](i2c(sys,s))) for s in p.state])
			addToMultiState(out,p2)
	return out

def inner(a,b):
	acc=0
	for state,amp in b.items():
		if tuple(state) in a:
			acc+=np.conj(a[state])*amp
	return acc

def fac(n):
	return 1 if n==0 else n*fac(n-1)

def binom(a,b):
	return fac(a)//fac(a-b)//fac(b)

def Lanczos( A, v, getZero, dot, mul, m):
    # https://en.wikipedia.org/wiki/Lanczos_algorithm
    V = [getZero() for i in range(m)]
    T = np.zeros( (m,m) )
    vo   = getZero()
    beta = 0
    for j in range( m-1 ):
        w    = A(v)
        alfa = dot(w,v)
        w    = w - alfa * v - beta * vo
        beta = np.sqrt( np.dot( w, w ) ) 
        vo   = v
        v    = w / beta 
        T[j,j  ] = alfa 
        T[j,j+1] = beta
        T[j+1,j] = beta
        V[j,:]   = v
    w    = np.dot( A,  v )
    alfa = np.dot( w, v )
    w    = w - alfa * v - beta * vo
    T[m-1,m-1] = np.dot( w, v )
    V[m-1]     = w / np.sqrt( np.dot( w, w ) ) 
    return T, V

def HubbardH(nx,ny,mu,tx,ty,U,psi):
	sys=[2,nx,ny]
	newPsi={}
	for pstate,pamp in psi.items():
		p=ProdState(pamp,pstate)
		p2=ProdState(mu*p.amp,p.state)
		addToMultiState(newPsi,p2)

		for ix in range(nx):
			for iy in range(ny):
				i0u=c2i(sys,[0,ix,iy])
				i0d=c2i(sys,[1,ix,iy])
				p2=p.c(i0d).cd(i0d).c(i0u).cd(i0u)
				p2.amp*=U
				addToMultiState(newPsi,p2)

				for ialpha in range(2):
					i0=c2i(sys,[ialpha,ix,iy])
					ix1=c2i(sys,[ialpha,(ix+1)%nx,iy])
					iy1=c2i(sys,[ialpha,ix,(iy+1)%ny])

					p2=p.c(i0).cd(ix1)
					p2.amp*=tx
					addToMultiState(newPsi,p2)
					p2=p.c(ix1).cd(i0)
					p2.amp*=tx
					addToMultiState(newPsi,p2)

					if ny>1:
						p2=p.c(i0).cd(iy1)
						p2.amp*=ty
						addToMultiState(newPsi,p2)
						p2=p.c(iy1).cd(i0)
						p2.amp*=ty
						addToMultiState(newPsi,p2)
	return newPsi

def main():
	nspins=2
	nx=6
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
	if ny==1:	
		for i in range(0,nx,2):
			psi=psi.cd(c2i(sys,[0,i,0]))
			# if i!=0:
			psi=psi.cd(c2i(sys,[1,(i+1)%nx,0]))

	else:
		for i in range(nx):
			for j in range(ny//2):
				jj=2*j+i%2
				psi=psi.cd(c2i(sys,[0,i,jj]))
				psi=psi.cd(c2i(sys,[1,(i+1)%nx,jj]))
	
	nSubStates=binom(nx*ny,nx*ny//2)**2
	# nSubStates=binom(nx*ny,nx*ny//2)**2
	# nup=4
	# l=choice(range(nx*ny),nup,False)
	# for i in l:
	# 	start=start.cd(c2i(sys,[0,]))
	
	psi.canonicalize()
	print(psi)


	# apply symmetry transfs
	# spin flip
	psi={tuple(psi.state):psi.amp}
	print(psi)
	
	identity=lambda x:x
	#psi=symmetrize(sys,psi,[identity,lambda x:[1-x[0],x[1],x[2]]],[1,1])
	# psi=symmetrize(sys,psi,[lambda x,i=i:[x[0],(x[1]+i)%nx,x[2]] for i in range(nx)],[1 for i in range(nx)])
	# psi=symmetrize(sys,psi,[lambda x,i=i:[x[0],x[1],(x[2]+i)%ny] for i in range(ny)],[1 for i in range(ny)])
	# psi=symmetrize(sys,psi,[identity,lambda x:[x[0],x[2],x[1]] ],[1 ,1])
	# psi=symmetrize(sys,psi,[identity,lambda x:[x[0],nx-x[1]-1,x[2]] ],[1 ,1])
	print(psi)

	tx=1
	ty=tx
	U=0
	shift=-18
	
	'''Lanczos(lambda psi:HubbardH(nx,ny,shift,tx,ty,U,psi), psi,
			lambda : {},
			inner,
			lambda f,psi:{k: f*v for k,v in psi.items()},
			2)'''

	nold=1
	steps=10
	ssteps=6
	totSteps=0
	for sstep in range(ssteps):
		steps*=2
		totSteps+=steps
		for it in range(steps):
			newPsi=HubbardH(nx,ny,shift,tx,ty,U,psi)

			print("states: "+str(len(newPsi))+"/"+str(nSubStates))
			n=inner(newPsi,newPsi)
			
			lam=inner(psi,newPsi)
			assert(lam<0)
			print("eig: "+str(lam-shift))

			nn=np.sqrt(n)
			for key in newPsi:
			 newPsi[key]/=nn

			psi=newPsi

		pairpairList=[]
		ppairppairList=[]
		densdensList=[]
		uddensdensList=[]
		for deltaX in range(nx):
			x1=0
			x1p=1
			y1=0
			x2=(x1+deltaX)%nx
			x2p=(x1+deltaX+1)%nx
			y2=0
			i1u=c2i(sys,[0,x1,y1])
			i1d=c2i(sys,[1,x1,y1])
			i2u=c2i(sys,[0,x2,y2])
			i2d=c2i(sys,[1,x2,y2])

			i1pu=c2i(sys,[0,x1p,y1])
			i1pd=c2i(sys,[1,x1p,y1])
			i2pu=c2i(sys,[0,x2p,y2])
			i2pd=c2i(sys,[1,x2p,y2])
			pairpair=0
			ppairppair=0
			densdens=0
			uddensdens=0
			for pstate,pamp in psi.items():
				p2=ProdState(pamp,pstate)
				s2=p2.c(i2u).c(i2d).cd(i1u).cd(i1d)
				s2.canonicalize()
				pairpair+=inner(psi,  {tuple(s2.state):s2.amp})

				s2=p2.c(i2u).c(i2pd).cd(i1u).cd(i1pd)
				s2.canonicalize()
				ppairppair+=inner(psi,  {tuple(s2.state):s2.amp})

				s2=p2.c(i2u).cd(i2u).c(i1u).cd(i1u)
				s2.canonicalize()
				densdens+=inner(psi,  {tuple(s2.state):s2.amp})

				s2=p2.c(i2u).cd(i2u).c(i1d).cd(i1d)
				s2.canonicalize()
				uddensdens+=inner(psi,  {tuple(s2.state):s2.amp})

			upnavg=1/2
			downnavg=1/2
			pairpairList.append(pairpair)
			ppairppairList.append(ppairppair)
			densdensList.append(densdens-upnavg*upnavg)
			uddensdensList.append(uddensdens-upnavg*downnavg)

			print("Dx="+str(deltaX)+", s-pair s-pair: "+str(pairpair))
			print("Dx="+str(deltaX)+", p-pair p-pair: "+str(ppairppair))
			print("Dx="+str(deltaX)+",     dens dens: "+str(densdens-upnavg*upnavg))
			print("Dx="+str(deltaX)+",   udens ddens: "+str(uddensdens-upnavg*downnavg))
			print()
		import pylab as pl
		pl.figure(sstep)
		corrList=[pairpairList,ppairppairList,densdensList,uddensdensList]
		corrList=[c+[c[0]] for c in corrList]
		assert(tx==1)
		pl.title("$U="+str(U)+"t_x$")
		for i in range(len(corrList)):
			pl.plot(corrList[i],label=[
				'$\\langle c_{\\uparrow}(0)c_{\\downarrow}(0)c^\dagger_{\\uparrow}(x)c^\dagger_{\\downarrow}(x)\\rangle$',
									'$\\langle c_{\\uparrow}(0)c_{\\downarrow}(1)c^\dagger_{\\uparrow}(x)c^\dagger_{\\downarrow}(x+1)\\rangle$',
									'$\\langle n_{\\uparrow}(0)n_{\\uparrow}(x)\\rangle-\\langle n_{\\uparrow}\\rangle\\langle n_{\\uparrow}\\rangle$',
									'$\\langle n_{\\uparrow}(0)n_{\\downarrow}(x)\\rangle-\\langle n_{\\uparrow}\\rangle\\langle n_{\\downarrow}\\rangle$'][i])
		pl.xlabel("$x$")
		pl.legend()
		pl.savefig('plots/corr_U='+str(U)+'_nx='+str(nx)+'_ny='+str(ny)+'_steps='+str(totSteps)+'.pdf', bbox_inches='tight',figsize=(2,1))
		
	# pl.figure(1)
	# pl.hist(psi.values(),bins=300)

	pl.show()


if __name__ == "__main__":
	main()