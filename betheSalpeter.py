import pylab as pl
import dqmc
import math	
import numpy as np
import numpy.linalg

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
beta=2

# fast for testing
# nx=4
# ny=4
# tx=1
# ty=1
# tne=0
# tnw=0
# U=4
# mu=0
# beta=2

m=6
nTauPerBeta=8

nTau=dqmc.getNTau(beta, nTauPerBeta, m)

nSamplesPerRun=80#measurePeriod*20

clean=False
if clean:
	import os
	folder = dqmc.getDirName(nTau,nx,ny,tx,ty,tnw,tne,U,mu,beta,m)
	for f in os.listdir(folder):
		file_path = os.path.join(folder, f)
		if os.path.isfile(file_path):
			os.unlink(file_path)
if True:
	l=np.load('cache/G2G4{U}.npz'.format(U=U))
	G4ppQ0=l['G4ppQ0']
	G4ppQ0Err=l['G4ppQ0']
	Gmom=l['Gmom']
	GmomErr=l['GmomErr']
else:
	print("Generating field configurations..")
	dqmc.genGSamples(nx,ny,tx,ty,tnw,tne,U,mu,beta,m,nTauPerBeta,nSamplesPerRun,nRuns=16,nThreads=1,startSeed=0)

	print("Loading field configurations..")
	cl=nSamplesPerRun-1
	runs=dqmc.loadRuns(nTau,nx,ny,tx,ty,tnw,tne,U,mu,beta,m)
	runs=[r for r in runs if len(r)==80]
	# runs=runs[:2]
	#TODO, figure out optimal warmUp
	warmUp=3

	pl.figure()
	ac=np.array(np.mean([dqmc.getAutocorrelator(r,cl) for r in runs],axis=0))
	pl.plot(ac[:,0,0,0,0,0])
	pl.plot(np.mean(ac,axis=(1,2,3,4,5)))
	pl.show()

	print("Calculating G in real space..")
	G,G_err=dqmc.averageOverG(runs,lambda g:dqmc.symmetrizeG(g,nTau,nx,ny),warmUp,showProgress=True)
	# print(G)
	#time order?
	
	print("Calculating spin averaged G in momentum space..")
	Gmom,GmomErr=dqmc.averageOverG(runs,lambda g:nx*ny/beta*np.mean(dqmc.momentumG(g,nTau,nx,ny,beta),axis=0),warmUp,showProgress=True)
	pl.figure()
	pl.imshow(np.real(Gmom[:,0,0,:,0,0]))
	pl.figure()
	pl.imshow(np.imag(Gmom[:,0,0,:,0,0]))
	# pl.show()
	# exit(0)
	Gmom=np.diagonal(Gmom,axis1=0,axis2=3) 
	Gmom=np.diagonal(Gmom,axis1=0,axis2=2)
	Gmom=np.diagonal(Gmom,axis1=0,axis2=1)

	GmomErr=np.diagonal(GmomErr,axis1=0,axis2=3) 
	GmomErr=np.diagonal(GmomErr,axis1=0,axis2=2)
	GmomErr=np.diagonal(GmomErr,axis1=0,axis2=1)

	#Gmom=dqmc.momentumG(np.array([G,G]),nTau,nx,ny,beta) #todo, extract diagonal pof above instead?

	def G4pp(g,nTau,nx,ny,Q,s1=0,s2=1):
		gmom=nx*ny/beta*dqmc.momentumG(g,nTau,nx,ny,beta)
		# gmom=np.roll(gmom,(nTau//2,nTau//2),axis=(1,4))
		return gmom[s1]*np.roll(np.flip(gmom[s2]),(0,1-Q[0],1-Q[1])*2,range(6))
		#TODO symmetrize before, if s1==s2, then sum other contraction

	def G4ph(g,nTau,nx,ny,Q,s1=0,s2=1):
		gmom=dqmc.momentumG(g,nTau,nx,ny,beta)
		return gmom[s1]*np.roll(gmom[s2],Q*2,axis=range(6))

	#get a modest speedup if we combine these and FT only once..
	print("Calculating 4-point function with (0,0) p-p COM momentum..")
	G4ppQ0,G4ppQ0Err=dqmc.averageOverG(runs,lambda g:G4pp(g,nTau,nx,ny,(0,0)),warmUp,showProgress=True)
	np.savez_compressed('cache/G2G4{U}.npz'.format(U=U), G4ppQ0=G4ppQ0,G4ppQ0Err=G4ppQ0Err,Gmom=Gmom,GmomErr=GmomErr)

# print("Calculating 4-point function with (π,π) p-h COM momentum..")
# G4phQpipi,G4phQpipiErr=dqmc.averageOverG(runs,lambda g:G4ph(g,nTau,nx,ny,(0,nx//2,ny//2)),warmUp,showProgress=True)

# GmomExt=np.roll(Gmom,(nTau//2,),axis=(0,))
GmomExt=Gmom
GmomReverse=np.roll(np.flip(GmomExt),(1,1),axis=(1,2))

# pl.figure()
# for i in range(nx):
# 	pl.errorbar(range(nTau),Gmom[:,i,0],yerr=GmomErr[:,i,0])
# # pl.plot(GmomReverse[:,0,(ny-0)%ny])
# pl.title('GmomExt')
# pl.show()

pl.figure()
for i in range(nx):
	pl.errorbar(range(nTau),G4ppQ0[0,0,0,:,i,0],yerr=G4ppQ0Err[0,0,0,:,i,0])
# pl.plot(GmomReverse[:,0,(ny-0)%ny])
pl.title('GmomExt')
pl.show()

# G4red=(-G4ppQ0)/np.diag((GmomExt*GmomReverse).reshape(nTau*nx*ny)).reshape((nTau,nx,ny)*2)

# pl.contourf(range(nx),range(ny),GmomExt[nTau//2])


G4red=(-G4ppQ0)

G4red=G4red/GmomExt[np.newaxis,np.newaxis,np.newaxis,:,:,:]/GmomReverse[np.newaxis,np.newaxis,np.newaxis,:,:,:]

pl.figure()
for i in range(nx):
	pl.plot(G4red[0,i,0,:,0,0])
# pl.plot(GmomReverse[:,0,(ny-0)%ny])
pl.title('G4red')
pl.show()

M=-G4red.reshape((nTau*nx*ny,)*2)
# pl.figure()
# pl.plot(G4red[nTau//2+7,0,0,:,0,0],'x-')
G4red+=np.identity(nTau*nx*ny).reshape((nTau,nx,ny)*2)
# pl.plot(G4red[nTau//2+7,0,0,:,0,0],'x-')
G4red=G4red/GmomExt[:,:,:,np.newaxis,np.newaxis,np.newaxis]/GmomReverse[:,:,:,np.newaxis,np.newaxis,np.newaxis]
#*(beta*nx*ny) typo in https://sci-hub.tw/https://journals.aps.org/prb/pdf/10.1103/PhysRevB.47.6157 ?
# pl.figure()
# pl.plot(G4red[nTau//2+7,0,0,:,0,0],'x-')

M2=1/(beta*nx*ny)*G4red*GmomExt[np.newaxis,np.newaxis,np.newaxis,:,:,:]*GmomReverse[np.newaxis,np.newaxis,np.newaxis,:,:,:]
M2=np.identity(nTau*nx*ny)-M2.reshape((nTau*nx*ny,)*2)
print("M/M2")
print(M/M2)
# pl.figure()
G4ired=np.transpose(np.linalg.solve(np.transpose(M2),np.transpose(G4red.reshape((nTau*nx*ny,)*2)))).reshape((nTau,nx,ny)*2)

pilist=([[nx//2-i,0+i] for i in range(nx//2)]
		+[[(-i)%nx,nx//2-i] for i in range(nx//2)]
		+[[(nx//2+i)%nx,(-i)%nx] for i in range(nx//2)]
		+[[i%nx,(-nx//2+i)%nx] for i in range(nx//2)])
# pl.plot(*np.transpose(((np.array(pilist)+nx//2-1)%nx-nx//2+1)*2*np.pi/nx),'x-')
pl.figure()
pl.plot([np.real(G4red[nTau//2,p[0],p[1],nTau//2,nx//2,0]) for p in pilist],'x-',label='reducible')
pl.plot([np.real(G4ired[nTau//2,p[0],p[1],nTau//2,nx//2,0]) for p in pilist],'x-',label='ireducible')
pl.legend()
# pl.plot(G4ppQ0[nTau//2+7,0,0,:,0,0],'x-')
# pl.plot(G4red[nTau//2+7,0,0,:,0,0],'x-')
# pl.plot(G4red[nTau//2+3,0,0,:,0,0],'x-')
pl.show()

# exit(0)


# Mpp=-(1/(beta*nx*ny))*G4ppQ0

# # pl.figure()
# # pl.contourf(Mpp)
# # pl.show()
# # exit(0)

# la, phi=np.linalg.eig(Mpp)
# # sortedOrder=sorted(range(len(la)),key=lambda i:abs(1-1/la[i]-1))
# # la=la[sortedOrder]
# # phi=phi[sortedOrder]
# pl.figure()
# for p in phi[:5]:
# 	pl.plot(pl.imag(p.reshape(nTau,nx,ny)[:,0,0]))
# pl.show()
# # exit(0)
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

pl.show()