import numpy as np
from scipy import sparse
from scipy.sparse import linalg
import pickle
import os.path

#we represent a product state by a set of numbers corresponding to applications of cdaggger on the vacuum.
#Each number is an index representing spin, x, y, ...
#The canonical representation has these in increasing order
#The first index is the last operator to be applied, ie they appear in same order as when acting on ket

# a general state is represented by a dictionary of canonical single state tuples:amplitudes

# an operator is represented by a function that takes a canonical tuple and an amplitude and gives a new canonical tuple and an amplitude

class ProdState:
	def __init__(self,amp,state):
		self.state=state
		self.amp=amp

	def c(self,i):
		pos=-1
		for j in range(len(self.state)):
			if i==self.state[j]:
				pos=j
				break
		if pos==-1:
			return ProdState(0,())
		else:
			return ProdState(self.amp*(1 if pos%2==0 else -1),self.state[:pos]+self.state[pos+1:])

	def cd(self,i):
		parity=1
		ip=len(self.state)
		for pos in range(len(self.state)):
			p=self.state[pos]
			if p==i:
				self.amp=0
				return ProdState(0,())
			if p>i:
				ip=pos
				break
		return ProdState(self.amp if ip%2==0 else -self.amp,self.state[:ip]+(i,)+self.state[ip:])
	
	def op(O):
		O()
		ProdState(self.amp if ip%2==0 else -self.amp,self.state[:ip]+(i,)+self.state[ip:])

	def toEquivClass(self,group,repr):
		for i in range(len(group)):
			nstate=group[i](self.state)
			if nstate<self.state:
				self.state=nstate
				self.amp/=repr[i]

	def __str__(self):
		return str(self.amp)+"*"+str(self.state)

	def __repr__(self):
		return str(self)


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
	if ps.state in ms:
		ms[ps.state]+=ps.amp
		# if ms[ps.state]==0:
		# 	ms.pop(ps.state) //TODO uncomment, why does this even get called??
	else:
		ms[ps.state]=ps.amp

def symmetrize(sys,psi,group,rep):
	out=[]
	for p in psi:
		for i in range(len(group)):
			p2=ProdState(p.amp*rep[i],[c2i(sys,group[i](i2c(sys,s))) for s in p.state])
			addToMultiState(out,p2)
	return out

def inner(a,b):
	#a,b multistates
	acc=0
	for state,amp in b.items():
		if state in a:
			acc+=np.conj(a[state])*amp
	return acc

def fac(n):
	return 1 if n==0 else n*fac(n-1)

def binom(a,b):
	return fac(a)//fac(a-b)//fac(b)

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
	nx=12
	ny=1
	sys=[nspins,nx,ny]
	tx=1
	ty=tx

	U=4
	nups=nx*ny//2
	ndowns=nx*ny//2

	filename="cache/ED_U="+str(U)+"_nx="+str(nx)+"_ny="+str(ny)
	if nups!=nx*ny//2:
		filename+="_nups="+str(nups)
	if ndowns!=nx*ny//2:
		filename+="_nups="+str(ndowns)
	if ty!=tx:
		filename+="_tyotx="+str(ty/tx)
	
	# pl.figure(1)
	# pl.hist(psi.values(),bins=1000,range=[-.00001,.00001])
	# pl.show()
	corrv=2
	corrFilename=filename+"_corr"+"_v="+str(corrv)
	if os.path.isfile(corrFilename):
		with open(corrFilename, 'rb') as f:
			corrList = pickle.load(f)
	else:
		if os.path.isfile(filename):
			with open(filename, 'rb') as f:
				psi = pickle.load(f)
		else:
			print("Constructing basis...")
			import itertools
			ups=list(itertools.combinations([c2i(sys,[0,x,y]) for y in range(ny) for x in range(nx) ], nups))
			downs=list(itertools.combinations([c2i(sys,[1,x,y]) for y in range(ny) for x in range(nx)], ndowns))
			print("Up basis size: "+str(len(ups)))
			print("Down basis size: "+str(len(downs)))
			import heapq
			basis=[tuple(heapq.merge(u,d)) for d in downs for u in ups]
			print("Total basis size: "+str(len(basis)))
			print("Indexing basis...")
			index={basis[i]:i for i in range(len(basis))}
			print("Creating sparse matrix...")
			Hmat = sparse.lil_matrix((len(basis), len(basis)))
			print("Constructing sparse matrix...")
			progress=-1
			for i in range(len(basis)):
				p=i*100//len(basis)
				if p!=progress:
					print(str(p)+"%")
					progress=p
				res=HubbardH(nx,ny,0,tx,ty,U,{basis[i]:1})
				for k,v in res.items():
					Hmat[i,index[k]]=v
		
			print("Matrix constructed")
			print("Finding groundstate...")
			vals, vecs = linalg.eigsh(Hmat, k=1,which='SA')
			print(vals)
			psi={basis[i]:vecs[i,0] for i in range(len(basis))}
			print("Saving result...")
			with open(filename, 'wb') as f:
				pickle.dump(psi, f)
		print("Calculating correlation functions...")
		pairpairList=[]
		ppairppairList=[]
		ofpairofpairList=[]
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
			ofpairofpair=0
			densdens=0
			uddensdens=0

			upnavg=nups/(nx*ny)
			downnavg=ndowns/(nx*ny)

			for pstate,pamp in psi.items():
				p2=ProdState(pamp,pstate)

				s2=p2.c(i2u).c(i2d).cd(i1u).cd(i1d)
				pairpair+=inner(psi,  {s2.state:s2.amp})

				s2=p2.c(i2u).c(i2pd).cd(i1u).cd(i1pd)
				ppairppair+=inner(psi,  {s2.state:s2.amp})

				

				s2=( p2.c(i2u).cd(i2u).c(i2pu).c(i2d) ).cd(i1d).cd(i1pu).c(i1u).cd(i1u)
				ofpairofpair+=inner(psi,  {s2.state:s2.amp})

				s2=( p2.c(i2pu).c(i2d) ).cd(i1d).cd(i1pu).c(i1u).cd(i1u)
				ofpairofpair+= -upnavg*inner(psi,  {s2.state:s2.amp})

				s2=( p2.c(i2u).cd(i2u).c(i2pu).c(i2d) ).cd(i1d).cd(i1pu)
				ofpairofpair+= -upnavg*inner(psi,  {s2.state:s2.amp})
				
				s2=( p2.c(i2pu).c(i2d) ).cd(i1d).cd(i1pu)
				ofpairofpair+= upnavg*upnavg*inner(psi,  {s2.state:s2.amp})


				s2=p2.c(i2u).cd(i2u).c(i1u).cd(i1u)
				densdens+=inner(psi,  {s2.state:s2.amp})

				s2=p2.c(i2u).cd(i2u).c(i1d).cd(i1d)
				uddensdens+=inner(psi,  {s2.state:s2.amp})

			
			pairpairList.append(pairpair)
			ppairppairList.append(ppairppair)
			ofpairofpairList.append(ofpairofpair)
			densdensList.append(densdens-upnavg*upnavg)
			uddensdensList.append(uddensdens-upnavg*downnavg)

			print("Dx="+str(deltaX)+", s-pair s-pair: "+str(pairpair))
			print("Dx="+str(deltaX)+", p-pair p-pair: "+str(ppairppair))
			print("Dx="+str(deltaX)+", t-pair t-pair: "+str(ofpairofpair))
			print("Dx="+str(deltaX)+",     dens dens: "+str(densdens-upnavg*upnavg))
			print("Dx="+str(deltaX)+",   udens ddens: "+str(uddensdens-upnavg*downnavg))
			print()

		corrList=[pairpairList,ppairppairList,ofpairofpairList,densdensList,uddensdensList]
		corrList=[c+[c[0]] for c in corrList]

		with open(corrFilename, 'wb') as f:
			pickle.dump(corrList, f)
	
	assert(tx==1)
	corrNames=[	'$\\langle c_{\\uparrow}(0)c_{\\downarrow}(0)c^\dagger_{\\uparrow}(x)c^\dagger_{\\downarrow}(x)\\rangle$',
				'$\\langle c_{\\uparrow}(0)c_{\\downarrow}(1)c^\dagger_{\\uparrow}(x)c^\dagger_{\\downarrow}(x+1)\\rangle$',
				'$\\langle n_{\\uparrow}(0)c^\dagger_{\\uparrow}(1)c^\dagger_{\\downarrow}(0)  c_{\\downarrow}(x)c_{\\uparrow}(x+1)n_{\\uparrow}(x)\\rangle$',
				'$\\langle n_{\\uparrow}(0)n_{\\uparrow}(x)\\rangle-\\langle n_{\\uparrow}\\rangle\\langle n_{\\uparrow}\\rangle$',
				'$\\langle n_{\\uparrow}(0)n_{\\downarrow}(x)\\rangle-\\langle n_{\\uparrow}\\rangle\\langle n_{\\downarrow}\\rangle$']
	import pylab as pl
	pl.figure(0)
	pl.title("$U="+str(U)+"t_x$")
	for i in range(len(corrList)):
		pl.plot(corrList[i],label=corrNames[i],marker='.')
	# xs=np.linspace(0,nx,200)
	# pl.loglog(xs,xs*2+0*(nx-xs)**-5,ls='--')

	pl.xlabel("$x$")
	pl.legend()
	pl.savefig('plots/corr_U='+str(U)+'_nx='+str(nx)+'_ny='+str(ny)+'.pdf', bbox_inches='tight',figsize=(2,1))

	pl.figure(1)
	pl.title("$U="+str(U)+"t_x$")
	for i in range(len(corrList)):
		subset=[s for s in range(1,nx//2+1) if corrList[i][s]!=0]
		pl.loglog(subset,np.abs(corrList[i])[subset],label=corrNames[i],marker='.')
	xs=np.linspace(1,nx//2,200)
	
	for i in range(2):
		p=[1,4][i]
		a=[.08,.04][i]
		pl.loglog(xs,a*(xs**-p+(nx-xs)**-p),linestyle='--')


	pl.xlabel("$x$")
	pl.legend()
	pl.savefig('plots/corr_U='+str(U)+'_nx='+str(nx)+'_ny='+str(ny)+'_log.pdf', bbox_inches='tight',figsize=(2,1))
	pl.show()


if __name__ == "__main__":
	main()