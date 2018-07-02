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
	if verbose: print("Constructing basis...")
	import itertools
	ups=list(itertools.combinations([c2i(sys,[0,x,y]) for y in range(ny) for x in range(nx) ], nUps))
	downs=list(itertools.combinations([c2i(sys,[1,x,y]) for y in range(ny) for x in range(nx)], nDowns))
	if verbose: print("Up basis size: "+str(len(ups)))
	if verbose: print("Down basis size: "+str(len(downs)))
	import heapq
	basis=[tuple(heapq.merge(u,d)) for d in downs for u in ups]
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
	Es, vecs = linalg.eigsh(Hmat, k=nStates,which='SA')
	return ([({basis[i]:vecs[i,vi] for i in range(len(basis))}) for vi in range(nStates)], Es)

def main():
	nspins=2
	nx=2
	ny=2
	sys=(nspins,nx,ny)
	tx=1
	ty=tx

	U=4*0
	beta=1 #2/tx
	nUps=nx*ny//2
	nDowns=nx*ny//2

	tausPerBeta=8
	ntau=tausPerBeta*beta
	print(ntau)
	nStates=15

	psis,Es=solveHubbard(nx,ny,tx,ty,U,nUps,nDowns,nStates)
	psism1u,Esm1u=solveHubbard(nx,ny,tx,ty,U,nUps-1,nDowns,nStates)

	# for nups in range(5):
	# 	_,E=solveHubbard(nx,ny,tx,ty,U,nups,nDowns,1)
	# 	print("{n}: {E}".format(n=nups,E=E))

	weights=[np.exp(-beta*e) for e in Es]
	Z=sum(weights)
	print("Last weight: {w} %".format(w=100*weights[-1]/Z))
	print("Calculating correlation function...")
	g=np.zeros((ntau,nx,ny))
	for psii in range(nStates):
		psi=psis[psii]
		w=weights[psii]/Z
		for x in range(nx):
			for y in range(ny):
				for taui in range(ntau):
					tau=beta*taui/ntau
					origin=c2i(sys,[0,0,0])
					i=c2i(sys,[0,x,y])

					# g[x,y,taui]+=w*inner(psi, cd(origin, c(i,psi),))
					g[taui,x,y]+=w*inner(psi, cd(origin, evolve(c(i,evolve(psi, psis, Es, -tau)), psism1u, Esm1u, tau)))
					# for pstate,pamp in psi.items():

					# 	p2=ProdState(pamp,pstate)
					# 	s1=p2.c(i)
					# 	s2=evolve({s1.state,s1.amp}, psis, Es, tau)
					# 	for s3 in 
					# 		s4=s2.cd(origin)
					# 		g[x,y,taui]+=w*inner(psi,  {s4.state:s4.amp})
						
						# s2=p2.c(i).cd(origin)
						
	print(g)


if __name__ == "__main__":
	main()