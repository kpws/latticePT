import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as pl

import dqmc
import math	
import numpy as np
import numpy.linalg
import scipy.sparse.linalg

experimentName='1'

def savefig(n):
	pl.savefig('plots/'+experimentName + n +'.pdf')
# nx=1
# ny=2
# tx=1
# ty=0.5
# tne=0
# tnw=0
# U=0
# mu=0
# beta=4

#https://sci-hub.tw/https://journals.aps.org/prb/abstract/10.1103/PhysRevB.47.14599
#scalapino bulut white, fig2a, rightmost point
nx=8
ny=8
tx=1
ty=1
tne=0
tnw=0
U=4
# U=.05
mu=0
beta=4

# fast for testing
# nx=2
# ny=2
# tx=1
# ty=1
# tne=0
# tnw=0
# U=4
# mu=0
# beta=1

m=6
nTauPerBeta=8

nTau=dqmc.getNTau(beta, nTauPerBeta, m)

nSamplesPerRun=300#measurePeriod*20

bandWidth=(2*tx+2*ty)
cutOff=2*bandWidth
cutOut=[i for i in range(nTau) if (i<nTau//2 and (2*i+1)*np.pi/beta>cutOff) or (i>=nTau//2 and (2*((nTau-1)-i)+1)*np.pi/beta>cutOff)]
IRnTau=nTau-len(cutOut)

clean=False
if clean:
	import os
	folder = dqmc.getDirName(nTau,nx,ny,tx,ty,tnw,tne,U,mu,beta,m)
	for f in os.listdir(folder):
		file_path = os.path.join(folder, f)
		if os.path.isfile(file_path):
			os.unlink(file_path)
if False:
	'''l=np.load('cache/G2G4{U}.npz'.format(U=U))
	G4ppQ0=l['G4ppQ0']
	G4ppQ0Err=l['G4ppQ0Err']
	Gmom=l['Gmom']
	GmomErr=l['GmomErr']'''
	#l=np.load('cache/all_U={U}_beta={beta}.npz'.format(U=U,beta=beta))
	#allRes=l['allRes']

	allRes=[]	
	for i in range(3):
		allRes.append(np.load('cache/all_U={U}_beta={beta}_{i}.npy'.format(i=i,U=U,beta=beta)))
	allRes=zip(*allRes)

	#allErr=l['allErr']
else:
	print("Generating field configurations..")
	dqmc.genGSamples(nx,ny,tx,ty,tnw,tne,U,mu,beta,m,nTauPerBeta,nSamplesPerRun,nRuns=20,nThreads=1,startSeed=0)

	print("Loading field configurations..")
	cl=nSamplesPerRun-1
	warmUp=4*nTau+1
	
	def allF(g,nTau,nx,ny):
		gmom=nx*ny/beta*dqmc.momentumG(g,nTau,nx,ny,beta)
		gmomRev=np.roll(np.flip(gmom,axis=range(1,7)),(0,1,1)*2,range(1,7))
		gmomRevQ=np.roll(np.flip(gmom,axis=range(1,7)),(0,1-nx//2,1-ny//2)*2,range(1,7))
		return np.array([   (gmom[0]+gmom[1])/2,
							(gmom[0]*gmomRev[1]+gmom[1]*gmomRev[0])/2,
							(gmom[0]*gmomRev[0]+gmom[1]*gmomRev[1])/2,
							(gmom[0]*gmomRevQ[1]+gmom[1]*gmomRevQ[0])/2,
							(gmom[0]*gmomRevQ[0]+gmom[1]*gmomRevQ[1])/2
						])

	print("Averaging over configurations..")
	#allRes,allErr=dqmc.averageOverGFiles(nTau,nx,ny,tx,ty,tnw,tne,U,mu,beta,m,lambda g:allF(g,nTau,nx,ny),warmUp,showProgress=True,limitNumRuns=15,nThreads=30)
	allRes=dqmc.measure(nTau,nx,ny,tx,ty,tnw,tne,U,mu,beta,m,lambda g:allF(g,nTau,nx,ny),warmUp,showProgress=True,limitNumRuns=-1,nThreads=2,nBins=4)

	#np.savez_compressed('cache/all_U={U}_beta={beta}.npz'.format(U=U,beta=beta), allRes=allRes,allErr=allErr)
	# another numpy bug... https://github.com/numpy/numpy/issues/10776
	for i in range(3):
		np.save('cache/all_U={U}_beta={beta}_{i}.npy'.format(i=i,U=U,beta=beta),np.array([a[i] for a in allRes]))

def symmetrizeXY(a):
	return (a+np.swapaxes(np.swapaxes(a,1,2),4,5))/2

def symmetrizeX(a):
	return (a+np.roll(np.flip(np.roll(np.flip(a,axis=1),1,1),axis=4),1,4))/2

Gmoms=[]
Lireds=[]
lambdaPHs=[]
phiPHs=[]
for b in allRes:
	op=b[0]/b[1]
	print("Average sign: {s}".format(s=b[1]/b[2]))
	for i in range(len(op)):
		op[i]=symmetrizeX(symmetrizeXY(symmetrizeX(op[i])))

	Gmom=op[0]
	Gmom=np.diagonal(Gmom,axis1=0,axis2=3) 
	Gmom=np.diagonal(Gmom,axis1=0,axis2=2)
	Gmom=np.diagonal(Gmom,axis1=0,axis2=1)

	GmomRev=np.roll(np.flip(Gmom),(1,1),(1,2))
	
	G4=-op[1]
	
	Lred=G4/Gmom[np.newaxis,np.newaxis,np.newaxis,:,:,:]/GmomRev[np.newaxis,np.newaxis,np.newaxis,:,:,:]
	Lred+=np.identity(nTau*nx*ny).reshape((nTau,nx,ny)*2)
	Lred=(beta*nx*ny)*Lred/Gmom[:,:,:,np.newaxis,np.newaxis,np.newaxis]/GmomRev[:,:,:,np.newaxis,np.newaxis,np.newaxis]

	M2=1/(beta*nx*ny)*Lred*Gmom[:,:,:,np.newaxis,np.newaxis,np.newaxis]*GmomRev[:,:,:,np.newaxis,np.newaxis,np.newaxis]
	M2=np.identity(nTau*nx*ny) - M2.reshape((nTau*nx*ny,)*2)
	Lired=np.transpose(np.linalg.solve(np.transpose(M2),np.transpose(Lred.reshape((nTau*nx*ny,)*2)))).reshape((nTau,nx,ny)*2)

	Mpp=-1/(beta*nx*ny)*Lired*Gmom[np.newaxis,np.newaxis,np.newaxis,:,:,:]*GmomRev[np.newaxis,np.newaxis,np.newaxis,:,:,:]
	Mpp=np.delete(Mpp,cutOut,0)
	Mpp=np.delete(Mpp,cutOut,3)

	nEigen=2
	lambdaPH,phiPH=scipy.sparse.linalg.eigs(Mpp.reshape((IRnTau*nx*ny,)*2),nEigen,which='LR')
	
	def canon(v):
		return v/v[0,1,2]
		
	Gmoms.append(Gmom)
	Lireds.append(Lired)
	lambdaPHs.append(lambdaPH)
	phiPHs.append(np.array(list(map(canon,np.transpose(phiPH).reshape(nEigen,IRnTau,nx,ny)))))

def average(a):
	return np.mean(a,axis=0),np.std(a,axis=0)/np.sqrt(len(a))

Gmom,GmomErr=average(Gmoms)
Lired,LiredErr=average(Lireds)
lambdaPH,lambdaPHErr=average(lambdaPHs)
phiPH,phiPHErr=average(phiPHs)

ebe=10

pilist=([[nx//2-i,0+i] for i in range(nx//2)]
        +[[(-i)%nx,nx//2-i] for i in range(nx//2)]
        +[[(nx//2+i)%nx,(-i)%nx] for i in range(nx//2)]
        +[[i%nx,(-nx//2+i)%nx] for i in range(nx//2+1)])

pl.figure()
wi=0
ppx=nx//2
ppy=0
y=[np.real(Lired[wi,p[0],p[1],wi,ppx,ppy]) for p in pilist]
yerr=[ebe*np.real(LiredErr[wi,p[0],p[1],wi,ppx,ppy]) for p in pilist]
pl.errorbar(range(len(pilist)),y,yerr=yerr,label='ireducible')

pl.title("Error bars are ${ebe}\\sigma$".format(ebe=ebe))
pl.legend()
savefig('Chi')

IRws=range(IRnTau)

pl.figure()
pl.title("Error bars are ${ebe}\\sigma$".format(ebe=ebe))
for pi in range(len(phiPH)):
	pl.errorbar(range(len(pilist)),[np.real(phiPH[pi][0,i[0],i[1]]) for i in pilist],yerr=[ebe*np.real(phiPHErr[pi][0,i[0],i[1]]) for i in pilist])
savefig("phi_vs_p")

pl.figure()
pl.title("Error bars are ${ebe}\\sigma$".format(ebe=ebe))
for pi in range(len(phiPH)):
	pxi=nx//2
	pyi=0
	pl.errorbar(IRws,np.roll(np.real(phiPH[pi][:,pxi,pyi]),IRnTau//2,axis=0),yerr=np.roll(ebe*np.real(phiPHErr[pi][:,pxi,pyi]),IRnTau//2,axis=0))
savefig("phi_vs_w")
exit(0)

# print(la)
# print(1-1/la)
# for p in phi[:1]:
# 	for t in p.reshape(nTau,nx,ny)[nTau//2:nTau//2+1]:
# 		pl.figure()
# 		pl.contourf(np.arange(nx)*2*np.pi/nx,np.arange(ny)*2*np.pi/ny, np.real(t) )
# for p in phi[-1:]:
# 	for t in p.reshape(nTau,nx,ny)[nTau//2:nTau//2+1]:
# 		pl.figure()
# 		pl.contourf(np.arange(nx)*2*np.pi/nx,np.arange(ny)*2*np.pi/ny, np.real(t) )
# pl.show()
# exit(0)



print("Calculate reducible vertices..")


print("Calculate ireducible vertices..")
#calculate reducible vertex

#calculate irreducible vertex
# Gamma=GammaR+GammaR*Gamma
# GammaR=Gamma/(1+Gamma)

#create eigenvalue problem

#solve for antiferromagnetic and supercond

gmom_free=np.zeros((nTau,nx,ny), dtype=complex)
for n in range(nTau):
	for kxi in range(nx):
		for kyi in range(ny):
			w=(2*(n-nTau//2)+1)*np.pi/beta
			kx=kxi*2*np.pi/nx
			ky=kyi*2*np.pi/ny
			eps=2*(np.cos(kx)*tx + np.cos(ky)*ty + np.cos(kx+ky)*tne + np.cos(-kx+ky)*tnw)
			gmom_free[n,kxi,kyi]=1/(1j*w-eps)

# pl.figure()
# pl.imshow(np.real(gmom[0,:,0,0,:,0]))
# pl.colorbar()
pl.figure()
pl.title("Gmom")
colors=[u'b', u'g', u'r', u'c', u'm', u'y', u'k']
for kx in range(nx):
	for ky in range(ny):
		# c=colors[(x+nx*y)%len(colors)]
		ws=(2*np.arange(-nTau//2,nTau//2)+1)*np.pi/beta
		pl.plot(ws,kx+nx*ky+np.real(np.roll(Gmom[:,kx,ky],nTau//2,axis=0)),linestyle='-',color='b')
		pl.plot(ws,kx+nx*ky+np.imag(np.roll(Gmom[:,kx,ky],nTau//2,axis=0)),linestyle='-',color='r')
		# pl.plot(kx+np.diag(np.imag(gmom[:,kx,ky,:,kx,ky])),linestyle='--')
		pl.plot(ws,kx+nx*ky+np.real(gmom_free[:,kx,ky]),linestyle=':',color='b')
		pl.plot(ws,kx+nx*ky+np.imag(gmom_free[:,kx,ky]),linestyle=':',color='r')
		# pl.plot(np.imag(gmom_free[:,kx,ky])/beta,linestyle=':')

pl.xlabel(r"$\omega_n$")
pl.ylabel(r"$G(x,y)$")
pl.legend()

if nx*ny<5:
	print('***Exact Diagonalization***')
	import ed
	taus=[taui/nTau*beta for taui in range(nTau)]
	ed_g=ed.getG(nx,ny,tx,ty,tnw,tne,U,mu,beta,taus,eps=1e-9)

	pl.figure()
	colors=[u'b', u'g', u'r', u'c', u'm', u'y', u'k']
	for x in range(nx):
		for y in range(ny):
			c=colors[(x+nx*y)%len(colors)]
			pl.plot(taus, ed_g[:,x,y],linestyle='--',color=c,label=r"$G_{{\mathrm{{ED}}}}({x},{y})$".format(x=x,y=y))
			pl.errorbar(taus, G[:,x+nx*y], yerr=0*G[:,x+nx*y],color=c,linestyle='-',label=r"$G_{{\mathrm{{DQMC}}}}({x},{y})$".format(x=x,y=y))
	pl.ylim([-1,1])
	pl.xlabel(r"$\tau$")
	pl.ylabel(r"$G(x,y)$")
	pl.legend()


# pl.figure()
# pl.plot(g[0,0,:,0,:])
# pl.plot(gsym[:,:])
savefig('G')
