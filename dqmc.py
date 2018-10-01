import numpy as np
from numpy import transpose, diag
import random
import math
import scipy
from scipy import linalg
from numpy import linalg
from functools import reduce
import multiprocessing
import time

DEBUG=False

def mgs(M):
	# return np.linalg.svd(M)
	#modified gram-schmidt factorization
	# Q - orthogonal, D - diagonal, R - unit upper triangular
	Q,R=np.linalg.qr(M)

	D=np.diagonal(R)
	R=diag(1/D)@R

	if DEBUG:
		diff=np.max(np.abs(Q@diag(D)@R-M))/np.mean(np.abs(M))
		if diff>1e-10:
			print("QR issue1: "+str(diff))

		diff=np.max(np.abs(Q@transpose(Q)-np.eye(len(M)) ))/np.mean(np.abs(Q))
		if diff>1e-10:
			print("QR issue2: "+str(diff))
	return Q,D,R

def triInv(A):
	ret=scipy.linalg.solve_triangular(A, np.identity(A.shape[0]),unit_diagonal=True)
	if DEBUG:
		diff=np.max(np.abs(A@ret-np.eye(len(A))))/np.mean(np.abs(A))
		if diff>1e-9:
			print("inv issue: "+str(diff))
	return ret

def calcB(Kexp,lamb,deltaTau,state,mu):
	return [Kexp@diag([np.exp( lamb*sig*s + deltaTau*mu ) for s in state])  for sig in [-1,1]]

def calciB(iKexp,lamb,deltaTau,state,mu):
	return [diag([np.exp( -lamb*sig*s - deltaTau*mu ) for s in state])@iKexp  for sig in [-1,1]]

def calcBs(Kexp,lamb,deltaTau,state,mu,N,ntau):
	return [[Kexp@diag([np.exp( lamb*sig*state[li][i] + deltaTau*mu ) for i in range(N)]) for li in range(ntau)] for sig in [-1,1]]

def calciBs(iKexp,lamb,deltaTau,state,mu,N,ntau):
	return [[diag([np.exp( -lamb*sig*state[li][i] - deltaTau*mu ) for i in range(N)])@iKexp for li in range(ntau)] for sig in [-1,1]]

def getNTau(beta, nTauPerBeta, m):
	minN=math.ceil(beta*nTauPerBeta/m)*m
	# nTau=m
	# while nTau<minN: nTau*=2
	return minN

def timeOrder(g,nTau):
	g2=np.zeros((2,nTau,nx*ny,nTau,nx*ny))
	for taui1 in range(nTau):
		for taui2 in range(nTau):
			g2[:,taui1,:,taui2]=g[:,taui1,:,taui2]*(1 if taui1<taui2 else -1)
	return g2

def symmetrizeG(g,nTau,nx,ny):
	gsym=np.zeros((nTau,nx*ny))
	for dtau in range(nTau):
		tmp=np.sum(np.trace(np.roll(g,-dtau,axis=2),axis1=1,axis2=2),axis=0)#why trace? loses info
		for x in range(nx):
			for y in range(ny):
				for dx in range(nx):
					for dy in range(ny):
						i1=x+nx*y
						i2=(x+dx)%nx+nx*((y+dy)%ny)
						i3=dx+nx*dy
						gsym[dtau,i3]+=tmp[i1,i2]
						# gsym[:,i3]+=np.roll(g[sigi,tau,:,i1,i2],tau)
	return gsym/(2*nTau*nx*ny)

def momentumG(g,nTau,nx,ny,beta):
	gxy=np.zeros((2,nTau,nx,ny,nTau,nx,ny))

	for x1 in range(nx):
		for y1 in range(ny):
			# gxy[:,:,x1,y1,:,:,:]=g[:,:,:,x1+nx*y1].reshape(2,nTau,nTau,nx,ny)
			for x2 in range(nx):
				for y2 in range(ny):
					gxy[:,:,x1,y1,:,x2,y2]=g[:,:,:,x1+nx*y1,x2+nx*y2]

	for t1 in range(nTau):
		for t2 in range(nTau):
			if t2<t1:
				gxy[:,t1,:,:,t2,:,:]=-gxy[:,t1,:,:,t2,:,:]
			if t1==t2:
				gxy[:,t1,:,:,t2,:,:]-=[.5*np.identity(nx*ny).reshape((nx,ny,nx,ny))]*2
	# print(gxy[0,:,0,0,:,0,0])
	# exit(0)
	#gxy=np.concatenate((gxy,-gxy),axis=1)
	#gxy=np.concatenate((gxy,-gxy),axis=4)
	fermFactor=np.exp(-1j*np.pi*np.arange(nTau)/nTau)
	# # use reshape?
	ret=(beta/(nTau*nx*ny))**2*np.fft.fftn(gxy
	 	*fermFactor[np.newaxis,:,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis]
	 	*fermFactor[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:,np.newaxis,np.newaxis],axes=(1,2,3,4,5,6))
	#ret=(beta/(nTau*nx*ny))**2*np.fft.fftn(gxy,axes=(2,3,5,6))#[:,1::2,:,:,1::2,:,:]

	#reverse second part per convention
	ret=np.roll(np.flip(ret,(1,5,6)),(1,1),axis=(5,6))
	return ret
# def updateBtree(tree,m,taui,newB):
# 	Bs=tree[0]
# 	Bs[taui]=newB
# 	for i in range(1,len(tree)):
# 		tree[i][taui//2**i]=BTree[1]

# def getBProd(Btree,start,end): #not including end
	
#Z. Bai et al. / Linear Algebra and its Applications 435 (2011) 659–673
def calcGFromScratch(Kexp,lamb,state,taui,deltaTau,mu,N,m,ntauOverm,B,stabilize=True,useSVD=True):
	ntau=m*ntauOverm
	if stabilize:
		mB=[[reduce(lambda a,b:a@b,(B[sig][(taui+j*m+i+1)%ntau] for i in range(m))) for j in range(ntauOverm)] for sig in range(2)]
		ret=[0,0]
		for sigi in range(2):
			for i in range(len(mB[sigi])-1,-1,-1):
				if i==len(mB[sigi])-1:
					M=mB[sigi][-1]
				else:
					M=(mB[sigi][i]@U)@diag(D)
				if useSVD:
					U,D,Vp=scipy.linalg.svd(M,lapack_driver='gesvd',overwrite_a=True) #default gesdd is not accurate enough, mgs is even better..
				else:
					U,D,Vp=mgs(M)
				if i!=len(mB[sigi])-1:
					V=Vp@V
				else:
					V=Vp
			Ds=[d if abs(d)<1 else 1 for d in D]
			Dbi=[1/d if abs(d)>1 else 1 for d in D]
			H=diag(Dbi)@transpose(U)+diag(Ds)@V
			LU,P=scipy.linalg.lu_factor(H)

			ret[sigi]=scipy.linalg.lu_solve((LU, P), diag(Dbi)@transpose(U))

		return ret
	else:
		Ml=[B[sigi][(taui+1)%ntau] for sigi in range(2)]
		for il in range(1,ntau):
			for sigi in range(2):
				Ml[sigi]=Ml[sigi] @ B[sigi][(taui+1+il)%ntau]

		return [np.linalg.inv(np.eye(N,N)+Ml[sigi]) for sigi in range(2)]

def stableProd(Bs,N,m):
	if len(Bs)==0:
		return np.eye(N,N),np.eye(N,N),np.eye(N,N)
	mB=[reduce(lambda a,b:a@b, Bs[m*i:min(m*i+m,len(Bs))]) for i in range((len(Bs)+m-1)//m)]
	# U,D,V=np.linalg.svd(mB[0])
	# for i in range(1,len(mB)):
	# 	Up,D,V=np.linalg.svd(diag(D)@(V@mB[i]))
	# 	U=U@Up
	# return U,D,V

	U,D,V=np.linalg.svd(mB[-1])
	for i in range(len(mB)-2,-1,-1):
		U,D,Vp=np.linalg.svd((mB[i]@U)@diag(D))
		V=Vp@V
	return U,D,V

def splitDiag(D):
	return [d if abs(d)<1 else 1 for d in D],[1/d if abs(d)>1 else 1 for d in D]

def dqmc(nx,ny,tx,ty,tne,tnw,U,mu,beta,ntauOverm,m,seed,nSweeps,stabilize,
	observables,observablesTD=[],stabEps=1e-4,autoCorrN=0,profile=False,returnState=False,startState=None,measurePeriod=1,saveSamples=None,nWarmupSweeps=0,showProgress=True,progressStart=0,progressFinish=1):
	rand=random.Random()
	rand.seed(seed)

	ntau=ntauOverm*m
	deltaTau=beta/ntau

	lamb=np.arccosh(np.exp(U*deltaTau/2))
	N=nx*ny
	K=np.zeros((N,N))
	sigs=[-1,1]
	if autoCorrN>0:
		old=[None for i in range(autoCorrN)]
		autoCorr=[0 for i in range(autoCorrN)]
		autoCorrSamples=0
		avCorrVal=0

	for x in range(nx):
		for y in range(ny):
			xp=(x-1)%nx
			xn=(x+1)%nx
			yp=(y-1)%ny
			yn=(y+1)%ny
			here=x+nx*y
			K[here, xp%nx+nx*y]-=tx
			K[here, xn%nx+nx*y]-=tx
			# if ny>1:
			K[here, x%nx+nx*yp]-=ty
			K[here, x%nx+nx*yn]-=ty

			K[here, xn%nx+nx*yn]-=tne
			K[here, xp%nx+nx*yp]-=tne

			K[here, xp%nx+nx*yn]-=tnw
			K[here, xn%nx+nx*yp]-=tnw	
	
	Kexp=scipy.linalg.expm(-deltaTau*K)
	iKexp=scipy.linalg.expm(deltaTau*K)
	
	if startState==None:
		state=[np.array([rand.choice([-1,1]) for i in range(N)]) for tau in range(ntau)]
	else:
		state=startState

	res=[0 for o in observables]
	resTD=[0 for o in observablesTD]

	B=calcBs(Kexp,lamb,deltaTau,state,mu,N,ntau)
	iB=calciBs(iKexp,lamb,deltaTau,state,mu,N,ntau)
	g=calcGFromScratch(Kexp,lamb,state,ntau-1,deltaTau,mu,N,m,ntauOverm,B)
	attempts=0
	accepted=0
	nMeasures=0
	sign=0
	
	gTimeDep=np.zeros((2,ntau,nx*ny,nx*ny))
	
	if profile: startTime=time.clock()

	for sweep in range(nWarmupSweeps+nSweeps):
		
		# mB=[[reduce(lambda a,b:a@b,(B[sigi][(j*m+i+1)%ntau] for i in range(m))) for sigi in range(2)] for j in range(ntauOverm)]
		mB=[[reduce(lambda a,b:a@b,(B[sigi][(j*m+i)%ntau] for i in range(m))) for sigi in range(2)] for j in range(ntauOverm)]

		Brights=[([],np.eye(nx*ny),[1 for i in range(nx*ny)],np.eye(nx*ny)) for sigi in range(2)]
		Bleft=[[([],np.eye(nx*ny),[1 for i in range(nx*ny)],np.eye(nx*ny)) ] for sigi in range(2)]
		for sigi in range(2):
			for i in range(1,ntau//m+1):
				n,U,D,V=Bleft[sigi][i-1]
				Up,Dp,Vp=scipy.linalg.svd(diag(D)@(V@mB[i-1][sigi]),lapack_driver='gesvd',overwrite_a=True)
				Up=U@Up
				Bleft[sigi].append((n+[((i-1)*m+ii)%ntau for ii in range(m)],Up,Dp,Vp))
		for taui in range(ntau-1,-1,-1):
			if showProgress:
				print("Progress {:2.1%}".format(progressStart+(progressFinish-progressStart)*(sweep*ntau+(ntau-1-taui))/ntau/(nWarmupSweeps+nSweeps)), end="\r")
			for u in range(nx*ny):
				# x=rand.randrange(nx)
				# y=rand.randrange(ny)
				# ii=x+nx*y
				ii=u%(nx*ny)
				delta=[np.exp(-2*sigs[sigi]*lamb*state[taui][ii])-1 for sigi in range(2)]
				R=[1+(1-g[sigi][ii,ii])*delta[sigi] for sigi in range(2)]
				Rprod=R[0]*R[1]
				ps=Rprod/(1+Rprod) #heatbath, alt: ps=Rprod
				sign+=np.sign(ps)
				p=abs(ps)
				attempts+=1
				if p>1 or rand.random()<p:
					accepted+=1
					state[taui][ii]*=-1
					for sigi in range(2):
						g[sigi]-=np.outer([1 if ii==i else 0 for i in range(N)]-g[sigi][::,ii],g[sigi][ii,::]*(delta[sigi]/R[sigi]))
			
			tB=calcB(Kexp,lamb,deltaTau,state[taui],mu)
			for sigi in range(2):
				B[sigi][taui]=tB[sigi]
			
			tiB=calciB(iKexp,lamb,deltaTau,state[taui],mu)
			for sigi in range(2):
				iB[sigi][taui]=tiB[sigi]

			# measure ?
			if sweep>=nWarmupSweeps:
				if autoCorrN>0:
					corrVal=np.array(state)
					corrVal=g[0]
					old=[corrVal if i==0 else old[i-1] for i in range(autoCorrN)]
					if sweep-nWarmupSweeps>=autoCorrN-1:
						avCorrVal+=corrVal
						autoCorrSamples+=1
						for i in range(autoCorrN):
							autoCorr[i]+=corrVal*old[i]
				if profile and taui==0 and sweep==nWarmupSweeps:
						startMeasTime=time.clock()
				if (sweep-nWarmupSweeps+1)%measurePeriod==0:
					nMeasures+=1
					for oi in range(len(observables)):
						res[oi]+=observables[oi](g)
					if len(observablesTD)!=0:
						for sigi in range(2):
							# https://link.springer.com/content/pdf/10.1007%2F978-1-4613-0565-1.pdf
							if stabilize:
								if False:
									gTimeDep[sigi][0]=g[sigi]
									U,D,V=np.linalg.svd(gTimeDep[sigi][0])
									for i in range(1,ntau):
										Up,D,V=np.linalg.svd(diag(D)@(V@B[sigi][(taui+i)%ntau]))
										U=U@Up
										gTimeDep[sigi][i]=U@diag(D)@V
								else:
									for i in range(0,ntau,1):
										if i==0:
											gTimeDep[sigi][i]=g[sigi]
										else:
											Ul,Dl,Vl=stableProd([B[sigi][(taui+i-j)%ntau] for j in range(i)],N,m)
											Ur,Dr,Vr=stableProd([B[sigi][(taui+ntau-j)%ntau] for j in range(ntau-i)],N,m)
											Up,Dp,Vp=np.linalg.svd(diag(1/Dl)@(transpose(Ul)@transpose(Vr))+(Vl@Ur)@diag(Dr))
											# c=min(Dp)/max(Dp)
											# if abs(c)<1e-7:
											# 	print(str(i/(ntau-1))+", "+str(c))
											gTimeDep[sigi][i]=(transpose(Vr)@transpose(Vp))@diag(1/Dp)@(transpose(Up)@Vl)
							else:
								gTimeDep[sigi][0]=g[sigi]
								for i in range(1,ntau):
									gTimeDep[sigi][i]=gTimeDep[sigi][i-1]@B[sigi][(taui+i)%ntau]
						for oi in range(len(observablesTD)):
							resTD[oi]+=observablesTD[oi](gTimeDep)

			#prepare for next slice

			# g=[ iB[sigi][(taui+1)%ntau]@g[sigi]@B[sigi][(taui+1)%ntau] for sigi in range(2)] #tau->tau+1
			g=[ B[sigi][taui]@g[sigi]@iB[sigi][taui] for sigi in range(2)] #tau->tau-1
			
			if stabilize and (taui)%m==0:
				gFromScratch=[0,0]
				for sigi in range(2):
					mB=reduce(lambda a,b:a@b,(B[sigi][(taui+i)%ntau] for i in range(m)))
					n,U,D,V=Brights[sigi]
					Up,Dp,Vp=scipy.linalg.svd((mB@U)@diag(D),lapack_driver='gesvd',overwrite_a=True)
					Brights[sigi]=([(taui+i)%ntau for i in range(m)]+n,Up,Dp,Vp@V)

					nl,Uleft,Dleft,Vleft=Brights[sigi]
					nr,Uright,Dright,Vright=Bleft[sigi][(taui+1)//m]
					# print()
					# print("next taui: {i}".format(i=(taui-1)%ntau))
					# print("nl={nl}, nr={nr}".format(nl=nl,nr=nr))
					# print("{nl}, {nr}".format(nl=nl,nr=nr))
					# print(Uright@diag(Dright)@Vright@Uleft@diag(Dleft)@Vleft)

					Dlefts,Dleftbi=splitDiag(Dleft)
					Drights,Drightbi=splitDiag(Dright)

					M=diag(Dleftbi)@(transpose(Uleft)@transpose(Vright))@diag(Drightbi)+diag(Dlefts)@(Vleft@Uright)@diag(Drights)
					LU,P=scipy.linalg.lu_factor(M)

					# gFromScratch[sigi]=transpose(Vright)@(diag(Drightbi)@scipy.linalg.lu_solve((LU, P), diag(Dleftbi)@transpose(Uleft)))
					gFromScratch[sigi]=transpose(Vright)@(diag(Drightbi)@scipy.linalg.lu_solve((LU, P), np.eye(nx*ny))@diag(Dleftbi))@transpose(Uleft)
					# gFromScratch[sigi]=np.linalg.inv(1+Uleft@(diag(Dleft)@(Vleft@Uright)@diag(Dright))@Vright)
					# gFromScratch=calcGFromScratch(Kexp,lamb,state,(taui+1)%ntau,deltaTau,mu,N,m,ntauOverm,B)
					
					mean=np.mean(np.abs(gFromScratch[sigi]))
					maxError=np.max(np.abs(g[sigi]-gFromScratch[sigi])/mean)
					if maxError>stabEps:
						print("WARNING3: numerical instability, decrease m. Max relative error of size {e:.3e}".format(e=maxError))

				g=gFromScratch

		#measure more
		if sweep%measurePeriod==0 and saveSamples!=None:
			# mB=[[reduce(lambda a,b:a@b,(B[sigi][(j*m+i)%ntau] for i in range(m))) for sigi in range(2)] for j in range(ntauOverm)]
			# mBprods=[mB]
			# for l in range(ntau/m-1):
			# 	mBprods.append([])
			# 	for taui in range(ntau/m):
			# 		mBprods[-1].append([])
			# 		for sigi in range(2):
			# 			U,D,V=mBprods[l]
			# 			Up,Dp,Vp=scipy.linalg.svd(diag(D)@(V@mB[taui+l*m+1][sigi]),lapack_driver='gesvd',overwrite_a=True)
			# 			mBprods[-1].append((U@Up,Dp,Vp))
			
			Bprods=[]
			for l in range(0,ntau+1):
				Bprods.append([])
				for taui in range(ntau):
					if l==0:
						Bprods[l].append([scipy.linalg.svd(np.eye(nx*ny),lapack_driver='gesvd',overwrite_a=True) for sigi in range(2)])
					else:
						Bprod=[]
						for sigi in range(2):
							U,D,V=Bprods[l-1][taui][sigi]
							Up,Dp,Vp=scipy.linalg.svd(diag(D)@(V@B[sigi][(taui-l)%ntau]),lapack_driver='gesvd',overwrite_a=True)
							Bprod.append((U@Up,Dp,Vp))
						Bprods[l].append(Bprod)
			# Ul,Dl,Vl=Bprods[3][7][0]
			# Ur,Dr,Vr=Bprods[2][4][0]
			# print(Ul@diag(Dl)@Vl@Ur@diag(Dr)@Vr)
			# Ul,Dl,Vl=Bprods[5][7][0]
			# print(Ul@diag(Dl)@Vl)
			g2TimeDep=np.zeros((2,ntau,ntau,nx*ny,nx*ny))
			for taui in range(ntau):
				for i in range(ntau):
					for sigi in range(2):
						# Ul,Dl,Vl=stableProd([B[sigi][(taui+i-j)%ntau] for j in range(i)],N,m)
						Ul,Dl,Vl=Bprods[i][(taui+i)%ntau][sigi]
						# Ur,Dr,Vr=stableProd([B[sigi][(taui+ntau-j)%ntau] for j in range(ntau-i)],N,m)
						Ur,Dr,Vr=Bprods[ntau-i][taui][sigi]
						Dls,Dlbi=splitDiag(Dl)
						Drs,Drbi=splitDiag(Dr)

						M=diag(Dlbi)@transpose(Ul)@transpose(Vr)@diag(Drbi)+diag(Dls)@Vl@Ur@diag(Drs)

						LU,P=scipy.linalg.lu_factor(M)

						g2TimeDep[sigi][taui][(taui+i)%ntau]=transpose(Vr)@(diag(Drbi)@scipy.linalg.lu_solve((LU, P), np.eye(nx*ny))@diag(Dls))@Vl
						# Up,Dp,Vp=scipy.linalg.svd(diag(1/Dl)@(transpose(Ul)@transpose(Vr))+(Vl@Ur)@diag(Dr),lapack_driver='gesvd',overwrite_a=True)
						# g2TimeDep[sigi,taui,i]=(transpose(Vr)@transpose(Vp))@diag(1/Dp)@(transpose(Up)@Vl)
						#TODO: split
			
			saveSamples(sweep, np.sign(ps), g2TimeDep)

		if False:
			for y in range(ny):
				print(reduce(lambda a,b:a+b,('O' if state[0][x+nx*y]==-1 else '*' for x in range(nx))))
			print()
			for y in range(ntau):
				print(reduce(lambda a,b:a+b,('O' if state[y][x+nx*0]==-1 else '*' for x in range(nx))))
			print()

	# print("Acceptance rate: {a}".format(a=accepted/attempts))
	# print("Average sign: {s}".format(s=sign/(nWarmupSweeps+nSweeps)/ntau/nx/ny))
	ret=([r/nMeasures for r in res],)
	if len(observablesTD)!=0:
		ret=ret+([r/nMeasures for r in resTD],)
	if profile:
		if nSweeps==0:
			startMeasTime=time.clock()
		ret=ret+(startMeasTime-startTime,time.clock()-startMeasTime)
	if autoCorrN>0:
		ret=ret+([a/autoCorrSamples-avCorrVal*avCorrVal/autoCorrSamples**2 for a in autoCorr],)
	if returnState:
		ret=ret+(state,)
	return ret

def optimizeRun(nx,ny,tx,ty,tnw,tne,U,mu,beta,m,tausPerBeta,nWarmupSweeps,tdops,stabilize=True):
	ntau=math.ceil(beta*tausPerBeta/m)*m

	warmupMargin=3
	autoCorrSamples=50
	print("Measuring metropolis correlation length...")
	_,autocorr,state=dqmc(nx,ny,tx,ty,tnw,tne,U,mu,beta,ntau//m,m,0,nWarmupSweeps,
		nWarmupSweeps*autoCorrSamples//warmupMargin,stabilize,[],autoCorrN=nWarmupSweeps//warmupMargin,returnState=True)
	
	autocorrNorm=np.array([np.mean(a) for a in autocorr])
	autocorrNorm/=autocorrNorm[0]
	for i in range(len(autocorrNorm)):
		if autocorrNorm[i]<.05:
			autocorrNorm=autocorrNorm[:i]

	c,lamb=np.linalg.lstsq([[1,-x] for x in range(len(autocorrNorm))], np.log(autocorrNorm),rcond=None)[0]
	print("Metropolis correlation length: {l:.1f} sweeps".format(l=1/lamb))
	
	# import pylab as pl
	# pl.plot(autocorrNorm)
	# pl.plot([np.exp(c-lamb*x) for x in range(len(autocorrNorm))],linestyle="--")
	# pl.show()
	
	if warmupMargin/lamb>nWarmupSweeps:
		print("WARNING: Not long enough warmup ({a:.1f} < {m} correlation lengths)".format(a=nWarmupSweeps*lamb,m=warmupMargin))
	else:
		print("Warmup sufficient ({a:.1f} > {m} correlation lengths)".format(a=nWarmupSweeps*lamb,m=warmupMargin))

	nTimeMeasureSweeps=5
	_,_,warmUpT,measT=dqmc(nx,ny,tx,ty,tnw,tne,U,mu,beta,ntau//m,m,0,
		nTimeMeasureSweeps,nTimeMeasureSweeps,stabilize,[],observablesTD=tdops,profile=True,startState=state)
	
	rm=(measT-warmUpT)/warmUpT
	print("Time per update sweep: {t:.1f} ms".format(t=1e3*warmUpT/nTimeMeasureSweeps))
	print("Time per update and measure sweep: {t:.1f} ms".format(t=1e3*measT/nTimeMeasureSweeps))

	measurePeriod=1
	var=math.inf
	while True:
		r=1/measurePeriod
		v=(np.exp(lamb/r)+1)*(1+r*rm)/(np.exp(lamb/r)-1)/r
		if v>var:
			break
			measurePeriod-=1
		measurePeriod+=1
		var=v
	print("Optimal number of sweeps between measurements: {per}".format(per=measurePeriod))
	return 1/lamb,measurePeriod

def getDirName(nTau,nx,ny,tx,ty,tnw,tne,U,mu,beta,m):
	return ("samples/nx={nx},ny={ny},ntau={nTau},tx={tx},tx={ty},tnw={tnw},"+
			"tne={tne},U={U},mu={mu},beta={beta},m={m}").format(
				nx=nx,ny=ny,nTau=nTau,tx=tx,ty=ty,tnw=tnw,tne=tne,U=U,mu=mu,beta=beta,m=m)

def genGSamples(nx,ny,tx,ty,tnw,tne,U,mu,beta,m,tausPerBeta,nSamplesPerRun,measurePeriod=-1,nRuns=1,nThreads=1,startSeed=0):
	ntau=getNTau(beta, tausPerBeta, m)
	if measurePeriod==-1:
		measurePeriod=ntau #ensures similar complexity for measurements and updates

	dirname=getDirName(ntau,nx,ny,tx,ty,tnw,tne,U,mu,beta,m)

	import os
	if not os.path.exists(dirname):
		os.makedirs(dirname)
	for run in range(nRuns):
		sweeps=[]
		sgns=[]
		gs=[]
		def save(sweep, sgn, g):
			sweeps.append(sweep)
			sgns.append(sgn)
			gs.append(g)
		seed=startSeed+run
		fn=str(seed)+".npz"
		file=os.path.join(dirname, fn)
		if os.path.exists(file):
			print("Skipping, run already exists: "+fn)
		else:
			dqmc(nx,ny,tx,ty,tne,tnw,U,mu,beta,ntau//m,m,seed,(nSamplesPerRun-1)*measurePeriod+1,True,
				[],measurePeriod=measurePeriod,saveSamples=save,progressStart=run/nRuns,progressFinish=(run+1)/nRuns)
			np.savez_compressed(file, sweeps=np.array(sweeps), sgns=np.array(sgns), gs=np.array(gs))

def loadRuns(ntau,nx,ny,tx,ty,tnw,tne,U,mu,beta,m,maxN=-1):
	import os
	from os.path import isfile, join
	path=getDirName(ntau,nx,ny,tx,ty,tnw,tne,U,mu,beta,m)
	files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith('.npz')]
	print("Found {n} previous runs.".format(n=len(files)))
	runs=[]
	for i in range(len(files)):
		if i==maxN:
			break
		loaded = np.load(os.path.join(path, files[i]))
		runs.append(list(zip(loaded['sweeps'],loaded['sgns'],loaded['gs'])))
		print('\tRun {i}: {n} configurations'.format(i=i,n=len(runs[-1])))

	return runs

def getAutocorrelator(run, length):
	assert(0<len(run)-length)
	return [sum(run[i][2]*run[i+l][2] for i in range(len(run)-length))/(len(run)-length) for l in range(length)]

def measure(ntau,nx,ny,tx,ty,tnw,tne,U,mu,beta,m,op,warmUp,showProgress=False,limitNumRuns=-1,nThreads=1, nBins=1):
	import os
	from os.path import isfile, join
	path=getDirName(ntau,nx,ny,tx,ty,tnw,tne,U,mu,beta,m)
	files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith('.npz')]
	print("Found {n} previous runs.".format(n=len(files)))
	opRes=[]
	nConfs=0
		
	if limitNumRuns==-1: limitNumRuns=len(files)	
	binSize=limitNumRuns//nBins
	nFiles=binSize*nBins
	if nFiles!=limitNumRuns:
		print("Skipping {n}/{tot} runs to get equally sized bins.".format(n=limitNumRuns-nFiles,tot=limitNumRuns))
	print("Loading first run..")
	loaded = np.load(os.path.join(path, files[0]))
	nextSamples=list(zip(loaded['sweeps'],loaded['sgns'],loaded['gs']))
	manager = multiprocessing.Manager()
	return_dict = manager.dict()
	bins=[]
	for i in range(nFiles):
		print("Progress {:2.1%}".format(i/limitNumRuns))
		runs=nextSamples#list(zip(loaded['sweeps'],loaded['sgns'],loaded['gs']))
		def work(wi,start,end,return_dict):
			return_dict[wi] =sum(op(runs[r][2])*runs[r][1] for r in range(start,end))
	
		print("Setting up processes..")
		bs=math.ceil(len(runs)/nThreads)
		job=[multiprocessing.Process(target=work, args=(t, t*bs,min((t+1)*bs,len(runs)),return_dict)) for t in range(nThreads)]
		print("Starting processes..")
		for t in job: t.start()
		if i+1<nFiles:
			print("Preloading next run..")
			loaded = np.load(os.path.join(path, files[i+1]))
			nextSamples=list(zip(loaded['sweeps'],loaded['sgns'],loaded['gs']))

		print("Joining processes..")
		for t in job: t.join()
		
		opSum=sum(return_dict.values())
		sgnSum=sum(r[1] for r in runs)
		oneSum=len(runs)
		if i%binSize==0:
			print("New bin..")
			bins.append([opSum,sgnSum,oneSum])
		else:
			bins[-1][0]+=opSum
			bins[-1][1]+=sgnSum
			bins[-1][2]+=oneSum
	#print(opRes)
	#print(sgnRes)
	return bins#np.array(b[0] for b in bins)/,np.array(sgnRes)

def averageOverGFiles(ntau,nx,ny,tx,ty,tnw,tne,U,mu,beta,m,op,warmUp,showProgress=False,limitNumRuns=-1,nThreads=1, nBins=-1):
	import os
	from os.path import isfile, join
	path=getDirName(ntau,nx,ny,tx,ty,tnw,tne,U,mu,beta,m)
	files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith('.npz')]
	print("Found {n} previous runs.".format(n=len(files)))
	opRes=[]
	nConfs=0
	if nThreads>1:
		import multiprocessing
		
	if limitNumRuns==-1: limitNumRuns=len(files)	
	print("Loading first run..")
	loaded = np.load(os.path.join(path, files[0]))
	for i in range(limitNumRuns):
		runs=list(zip(loaded['sweeps'],loaded['sgns'],loaded['gs']))
		opAcc=0
		sgnAcc=0

		def work(start,end,return_dict):
			return_dict[i] = sum(op(runs[r][2])*runs[r][1] for r in range(start,end))
	
		manager = multiprocessing.Manager()
		return_dict = manager.dict()
		bs=len(runs)//nThreads
		job=[multiprocessing.Process(target=work, args=(t*bs,min((t+1)*bs,len(runs)),return_dict)) for t in range(nThreads)]
		print("Progress {:2.1%}".format(i/limitNumRuns))
		print("Starting processes..")
		for t in job: t.start()
		if i+1<limitNumRuns:
			print("Preloading next run..")
			loaded = np.load(os.path.join(path, files[i+1]))
		print("Joining processes..")
		for t in job: t.join()
		res=return_dict.values()
		for r in res:
			opAcc+=r
		for r in runs:
			sgnAcc+=r[1]
		'''
		for j in range((len(run)-warmUp)//nThreads+1):
			start=warmUp+j*nThreads
			end=min(start+nThreads,len(run))
			if showProgress:
				print("Progress {:2.1%}".format((i*(len(run)-warmUp)+j*nThreads)/((len(run)-warmUp)*limitNumRuns)), end="\r")
			
			def work(i,return_dict):
				return_dict[i] = op(run[start+i][2])*run[start+i][1]
		
			manager = multiprocessing.Manager()
			return_dict = manager.dict()
			job=[multiprocessing.Process(target=work, args=(t,return_dict)) for t in range(end-start)]
			for t in job: t.start()
			for t in job: t.join()
			res=return_dict.values()
			for r in res:
				opAcc+=r
			#if nThreads>1:
			#	pool.map(op, run[start:end][2])
			#else:
			#	opAcc+=op(run[start][2])
			for t in range(end-start):
				sgnAcc+=run[start+t][1]'''
		weight=len(runs)-warmUp
		opRes.append(opAcc/sgnAcc)
		nConfs+=weight
	return (np.mean(opRes,axis=0),np.std(opRes,axis=0)/np.sqrt(len(opRes)))

def averageOverG(runs,op,warmUp,showProgress=False):
	opRes=[]
	nConfs=0
	for i in range(len(runs)):
		opAcc=0
		sgnAcc=0
		for j in range(warmUp,len(runs[i])):
			if showProgress:
				print("Progress {:2.1%}".format((i*(len(runs[i])-warmUp)+j-warmUp)/((len(runs[i])-warmUp)*len(runs))), end="\r")
			opAcc+=op(runs[i][j][2])
			sgnAcc+=runs[i][j][1]
		weight=len(runs[i])-warmUp
		opRes.append(opAcc/sgnAcc)
		nConfs+=weight

	return (np.mean(opRes,axis=0),np.std(opRes,axis=0)/np.sqrt(len(opRes)))

def getG(nx,ny,tx,ty,tnw,tne,U,mu,beta,m,tausPerBeta,nThreads,nWarmupSweeps,nSweepsPerThread,stabilize=True,nSamples=1):
	
	ntau=math.ceil(beta*tausPerBeta/m)*m
	
	def averageG(g):
		ag=np.zeros((ntau,nx,ny))
		for sigi in range(2):
			for x in range(nx):
				for y in range(ny):
					for dx in range(nx):
						for dy in range(ny):
							ag[:,x,y]+=g[sigi][:,dx+dy*nx,(dx+x)%nx+((dy+y)%ny)*nx]
		return ag/(2*nx*ny)
	
	_,measurePeriod=optimizeRun(nx,ny,tx,ty,tnw,tne,U,mu,beta,m,tausPerBeta,nWarmupSweeps,[averageG],stabilize)
	
	def work(seed,return_dict):
		ret=dqmc(nx,ny,tx,ty,tne,tnw,U,mu,beta,ntau//m,m,seed,nWarmupSweeps,nSweepsPerThread,
			stabilize,[],observablesTD=[averageG],measurePeriod=measurePeriod)
		return_dict[seed] = ret
	
	print("Running DQMC...")
	if nThreads==1:
		r=[]
		for i in range(nSamples):
			print("Progress {:2.1%}".format(i/nSamples), end="\r")
			ret=dqmc(nx,ny,tx,ty,tnw,tne,U,mu,beta,ntau//m,m,1+i,nWarmupSweeps,nSweepsPerThread,stabilize,
				[],observablesTD=[averageG],measurePeriod=measurePeriod)
			r.append(ret[1][0])
		return np.mean(r,axis=0),np.std(r,axis=0)/np.sqrt(nSamples)
	else:
		manager = multiprocessing.Manager()
		return_dict = manager.dict()
		job=[multiprocessing.Process(target=work, args=(i,return_dict)) for i in range(nThreads)]
		startTime=time.time()
		print("Starting {n} jobs...".format(n=nThreads))
		for t in job: t.start()
		print("Waiting for {n} jobs...".format(n=nThreads))
		for t in job: t.join()
		print("Time per sweep: {time:.2f} ms".format(time=1000*(time.time()-startTime)/(nSweepsPerThread+nWarmupSweeps)/nThreads))
		res=return_dict.values()
		r=[res[t][1][0] for t in range(nThreads)]
		return np.mean(r,axis=0),np.std(r,axis=0)/np.sqrt(nThreads)


def main():
	nx=2
	ny=2
	U=2
	tx=1
	ty=1
	beta=2/tx #2/tx
	m=8
	tausPerBeta=8 #8 recommended in literature
	mu=0

	ntau=(int(beta*tausPerBeta)//m)*m

	nThreads=8
	nWarmupSweeps=200
	nSweepsPerThread=2500

	# ops=[(lambda g:g[0]),(lambda g:g[1]),(lambda g:2-g[0][0,0]-g[1][0,0])]
	opnames=["<n>","<nn>"]
	ops=[(lambda g:2-g[0][0,0]-g[1][0,0]),(lambda g:1-g[0][0,0]-g[1][0,0]+g[0][0,0]*g[1][0,0])]

	opnames=["<g{i}>".format(i=i) for i in range(nx)]
	ops=[lambda g,i=i:g[0][0,i] for i in range(nx)]

	opnames=["g"]
	def averageG(g):
		ag=np.zeros((ntau,nx,ny))
		for sigi in range(2):
				for x in range(nx):
					for y in range(ny):
						for dx in range(nx):
							for dy in range(ny):
								ag[:,x,y]+=g[sigi][:,dx+dy*nx,(dx+x)%nx+((dy+y)%ny)*nx]
		return ag/(2*nx*ny)

	ops=[lambda g:(g[0]+g[1])*.5]
	ops=[averageG]
	# ops=[lambda g:transpose(np.reshape(g[0][0,:],(ny,nx)))]

	
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
		print
		if opnames[i]=="g":
			# g=np.zeros((ntau,nx,ny))
			# for tau in range(ntau):
			# 	for x in range(nx):
			# 		for y in range(ny):
			# 			for dx in range(nx):
			# 				for dy in range(ny):
			# 					g[tau,x,y]+=mean[tau,dx+dy*nx,(dx+x)%nx+((dy+y)%ny)*nx]/(nx*ny)

			# print(g)
			print("{name} = {mean}\n±\n{std}".format(name=opnames[i],mean=mean,std=std))
		else:
			ri=[res[t][i] for t in range(nThreads)]
			mean=np.mean(ri,axis=0)
			std=np.std(ri,axis=0,ddof=1)/np.sqrt(nThreads)
			print("{name} = {mean} ± {std}".format(name=opnames[i],mean=mean,std=std))


if __name__ == "__main__":
	main()
