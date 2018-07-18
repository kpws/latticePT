import numpy as np
from scipy import sparse
from finite import HubbardH,c2i,c,cd,inner,addToFirst

def getG(nx,ny,tx,ty,tnw,tne,U,mu,beta,taus,eps=1e-4):
	print("Generating basis...")
	sys=(2,nx,ny)
	import itertools
	basis=[s for n in range(2*nx*ny+1) for s in list(itertools.combinations(range(2*nx*ny), n))]

	index={basis[i]:i for i in range(len(basis))}
	Hmat = sparse.lil_matrix((len(basis), len(basis)))

	print("Filling Hamiltonian...")
	for i in range(len(basis)):
		res=HubbardH(nx,ny,U/2,tx,ty,tnw,tne,U,{basis[i]:1})
		for k,v in res.items():
			Hmat[i,index[k]]=v

	print("Diagonalizing Hamiltonian...")
	Es, vecs = np.linalg.eigh(Hmat.todense())
	psis=[({basis[i]:vecs[i,vi] for i in range(len(basis)) if vecs[i,vi]!=0}) for vi in range(len(Es))]


	weights=np.exp(-beta*Es)
	Z=sum(weights)
	weights/=Z
	print("Calculating correlation function...")
	g=np.zeros((len(taus),nx,ny),dtype=complex)
	for psii in range(len(Es)):
		print("Progress {:2.1%}".format(psii/len(Es)), end="\r")
		psi=psis[psii]
		w=weights[psii]
		# print(w)
		# if w<eps:
		# 	break
		#acknowldegement: joost.slingerland
		origin=c2i(sys,[0,0,0])
		p2=cd(origin,psi)
		rep=[inner(e,p2) for e in psis]
		# rep2=[[[inner(psi, c(c2i(sys,[0,x,y]), p)) for p in psis] for y in range(ny)] for x in range(nx)]
		for taui in range(len(taus)):
			ev={}
			for i in range(len(psis)):
				f=w*rep[i]*np.exp(taus[taui]*(Es[psii]-Es[i]))
				if abs(f)>eps:
					addToFirst(ev, psis[i], f)

			for x in range(nx):
				for y in range(ny):
					i=c2i(sys,[0,x,y])
					g[taui,x,y]+=inner(psi, c(i, ev))

			# for x in range(nx):
			# 	for y in range(ny):
			# 		for i in range(len(psis)):
			# 			f=w*rep[i]*np.exp(taus[taui]*(Es[psii]-Es[i]))
			# 			g[taui,x,y]+=f*rep2[x][y][i]
			
	return g