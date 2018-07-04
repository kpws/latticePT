import numpy as np
from scipy import sparse
from scipy.sparse import linalg
import pickle
import os.path

from finite import *
#we represent a product state by a set of numbers corresponding to applications of cdaggger on the vacuum.
#Each number is an index representing spin, x, y, ...
#The canonical representation has these in increasing order
#The first index is the last operator to be applied, ie they appear in same order as when acting on ket

# a general state is represented by a dictionary of canonical single state tuples:amplitudes

# an operator is represented by a function that takes a canonical tuple and an amplitude and gives a new canonical tuple and an amplitude
def evolve(state, eigenstates, energies, tau):
	ret={}
	for i in range(len(eigenstates)):
		addToFirst(ret, eigenstates[i], inner(state,  eigenstates[i])*np.exp(-tau*energies[i])  )
	return ret

def solveHubbard(nx,ny,tx,ty,U,nUps,nDowns,nStates,verbose=False):
	#half filling: mu=U/2
	sys=(2,nx,ny)
	# if verbose: print("Constructing basis...")
	# import itertools
	# ups=list(itertools.combinations([c2i(sys,[1,x,y]) for y in range(ny) for x in range(nx)], nDowns))
	# downs=list(itertools.combinations([c2i(sys,[1,x,y]) for y in range(ny) for x in range(nx)], nDowns))
	# if verbose: print("Up basis size: "+str(len(ups)))
	# if verbose: print("Down basis size: "+str(len(downs)))
	# import heapq
	# basis=[tuple(heapq.merge(u,d)) for d in downs for u in ups]
	import itertools
	basis=[s for n in range(2*nx*ny+1) for s in list(itertools.combinations(range(2*nx*ny), n))]

	if verbose: print("Total basis size: "+str(len(basis)))
	if verbose: print("Indexing basis...")
	index={basis[i]:i for i in range(len(basis))}
	if verbose: print("Creating sparse matrix...")
	Hmat = sparse.lil_matrix((len(basis), len(basis)))
	if verbose: print("Constructing sparse matrix...")
	progress=-1
	for i in range(len(basis)):
		# p=i*100//len(basis)
		# if p!=progress:
		# 	print(str(p)+"%")
		# 	progress=p
		res=HubbardH(nx,ny,U/2,tx,ty,U,{basis[i]:1})
		for k,v in res.items():
			Hmat[i,index[k]]=v
	if verbose: print("Finding groundstate...")
	
	if nStates==-1:
		Es, vecs = np.linalg.eigh(Hmat.todense())
	else:
		Es, vecs = linalg.eigsh(Hmat, k=nStates,which='SA')
	return ([({basis[i]:vecs[i,vi] for i in range(len(basis))}) for vi in range(len(Es))], Es)

def main():
	nspins=2
	nx=2
	ny=1
	sys=(nspins,nx,ny)
	tx=1
	ty=0*tx

	U=8
	beta=2 #2/tx
	nUps=nx*ny//2
	nDowns=nx*ny//2

	# for nups in range(nx*ny):
	# 	_,E=solveHubbard(nx,ny,tx,ty,U,nups,nups,-1)
	# 	print("{n}: {E}".format(n=nups,E=min(E)))

	tausPerBeta=8
	ntau=int(beta*tausPerBeta)

	psis,Es=solveHubbard(nx,ny,tx,ty,U,nUps,nDowns,-1)
	psism1u,Esm1u=solveHubbard(nx,ny,tx,ty,U,nUps-1,nDowns,-1)
	# Es=-Es
	

	# weights=[np.exp(-beta*e) for e in Es]
	weights=np.exp(-beta*Es)
	Z=sum(weights)
	weights/=Z
	print("Calculating correlation function...")
	
	# if nx==2 and ny==1:
	# 	print("Exact: {E1}, {E2}, {E3}, {E4},".format(E1=U/2-np.sqrt((U/2)**2+4*(2*tx)**2),E2=0,E3=U,E4=U/2+np.sqrt((U/2)**2+4*(2*tx)**2)))
	
	iup=c2i(sys,[0,0,0])
	idown=c2i(sys,[1,0,0])
	i2down=c2i(sys,[1,1,0])

	g=np.zeros((ntau,nx,ny),dtype=complex)
	nupndown=0
	E=0
	S=0
	n=0
	for psii in range(len(Es)):
		psi=psis[psii]
		w=weights[psii]

		# print(psi)
		# print(inner(psi,HubbardH(nx,ny,0,tx,ty,U,psi)))
		# print(psi)
		# print(cd(iup,c(iup,psi )))
		# print(inner(psi,cd(idown,c(idown,cd(iup,c(iup,psi ))))))
		E+=w*Es[psii]
		nupndown+=w*inner(psi,cd(idown,c(idown,cd(iup,c(iup,psi )))))
		S+=w*(2*inner(psi,cd(i2down,c(i2down,cd(idown,c(idown,psi )))))-2*inner(psi,cd(i2down,c(i2down,cd(iup,c(iup,psi ))))))
		for i in range(2*nx*ny):
			n+=w*inner(psi,cd(i,c(i,psi)))
		for x in range(nx):
			for y in range(ny):
				for taui in range(ntau):
					tau=beta*taui/ntau
					origin=c2i(sys,[0,0,0])
					i=c2i(sys,[0,x,y])
					# g[taui,x,y]+=w*inner(psi, cd(origin, evolve(c(i,evolve(psi, psis, Es, -tau)), psism1u, Esm1u, tau)))	
	print("filling: {n}".format(n=n/(2*nx*ny)))
	print("E per site: {E}".format(E=(E+n*U/2)/(nx*ny)))
	print("nupndown: {nupndown}".format(nupndown=nupndown))
	print("SiSi+1: {S}".format(S=S))
	# print(g)


if __name__ == "__main__":
	main()