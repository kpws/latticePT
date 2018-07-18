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

def proj(x,u):
    ## Can't hurt
    u = unit_vec(u)
    return np.dot(x,u) * u

def unit_vec(x):
    """Get unit vector of x. Same direction, norm==1"""
    return x/np.linalg.norm(x)

def modifiedGramSchmidt(vectors):
    """ _correct_ recursive implementation of Gram Schmidt algo that is not subject to 
    rounding erros that the original formulation is. 
    Function signature and usage is the same as gramSchmidt()
    """
    ###### Ensure the input is a 2d array (or can be treated like one)
    vectors = np.atleast_2d(vectors)

    ###### Handle End Conditions
    if len(vectors) == 0:
        return []

    ## Always just take unit vector of first vector for the start of the basis
    u1 = unit_vec(vectors[0])

    if len(vectors) == 1:
        return u1

    ###### Orthonormalize the rest of the vectors
    #           | easy row stacking
    #           |                                            | Get the orthagonal projection of each subsequent vector onto u1 (ensures whole space is now orthagonal to u1)                  
    #                       | Recurse on the projections     |
    basis = np.vstack( (u1, modifiedGramSchmidt( list(map(lambda v: v - proj(v,u1), vectors[1:])))) ) # not explicit list(map) conversion, need for python3+

    return np.array(basis)


def mgs(M):
	# return np.linalg.svd(M)
	#modified gram-schmidt factorization
	# Q - orthogonal, D - diagonal, R - unit upper triangular
	Q,R=np.linalg.qr(M)
	#Q=transpose(modifiedGramSchmidt(transpose(M)))
	#R=transpose(Q)@M
	
	
	
	D=np.diagonal(R)
	R=diag(1/D)@R
	diff=np.max(np.abs(Q@diag(D)@R-M))/np.mean(np.abs(M))
	if diff>1e-10:
		print("QR issue1: "+str(diff))

	diff=np.max(np.abs(Q@transpose(Q)-np.eye(len(M)) ))/np.mean(np.abs(Q))
	if diff>1e-10:
		print("QR issue2: "+str(diff))
	return Q,D,R

# K D1 K D2 K D3
# K=U D Uinv


def triInv(A):
	# return transpose(A)
	ret=scipy.linalg.solve_triangular(A, np.identity(A.shape[0]),unit_diagonal=True)
	diff=np.max(np.abs(A@ret-np.eye(len(A))))/np.mean(np.abs(A))
	if diff>1e-9:
		print("inv issue: "+str(diff))
	return ret
	# return scipy.linalg.inv(A)

def calcB(Kexp,lamb,deltaTau,state,mu):
	# w,v=np.linalg.eig(Kexp@diag([np.exp( lamb*1*s + deltaTau*mu ) for s in state]))
	# print(w)
	return [Kexp@diag([np.exp( lamb*sig*s + deltaTau*mu ) for s in state])  for sig in [-1,1]]

def calciB(iKexp,lamb,deltaTau,state,mu):
	return [diag([np.exp( -lamb*sig*s - deltaTau*mu ) for s in state])@iKexp  for sig in [-1,1]]

def calcBs(Kexp,lamb,deltaTau,state,mu,N,ntau):
	return [[Kexp@diag([np.exp( lamb*sig*state[li][i] + deltaTau*mu ) for i in range(N)]) for li in range(ntau)] for sig in [-1,1]]

def calciBs(iKexp,lamb,deltaTau,state,mu,N,ntau):
	return [[diag([np.exp( -lamb*sig*state[li][i] - deltaTau*mu ) for i in range(N)])@iKexp for li in range(ntau)] for sig in [-1,1]]
#Z. Bai et al. / Linear Algebra and its Applications 435 (2011) 659–673
def calcGFromScratch(Kexp,lamb,state,taui,deltaTau,mu,N,m,ntauOverm,B,stabilize,special=False,Bi=None,precursor=False):
	ntau=m*ntauOverm
	if special:
		mB=[[reduce(lambda a,b:a@b,(Bi[sig][(taui+j*m+(ntau-1-i)+1)%ntau] for i in range(m))) for j in range(ntauOverm)] for sig in range(2)]
		# print("eig:")
		# print(np.linalg.eig(mB[0][0]))
		ret=[0,0]
		for sigi in range(2):
			# U,D,V=np.linalg.svd(mB[sigi][-1])
			U,D,V=mgs(mB[sigi][-1])
			for i in range(len(mB[sigi])-2,-1,-1):
				M=(mB[sigi][i]@U)@diag(D)
				# U,D,Vp=np.linalg.svd(M)
				U,D,Vp=mgs(M)
				V=Vp@V
			
			M=U@diag(D)@V
			v,w=np.linalg.eig(M)
			print("id:")
			print(D)
			print(v)
			# print(np.real(w@np.diag(v)@np.linalg.inv(w)))
			print(M@np.linalg.inv((np.eye(len(M))+M)))
			ret[sigi]=np.real(w@np.diag(v/(1+v))@np.linalg.inv(w))
			print(ret[sigi])


		ntauE=1
		while 2**ntauE!=ntau:
			ntauE+=1
		ret=[]
		for sigi in range(2):
			# old=[mgs(B[sigi][(taui+i+1)%ntau]) for i in range(ntau)]
			old=[mgs(np.linalg.inv(B[sigi][(taui+(ntau-i-1)+1)%ntau])) for i in range(ntau)]
			
			for l in range(ntauE):
				new=[]
				for i in range(len(old)//2):
					U1,D1,V1=old[2*i]
					U2,D2,V2=old[2*i+1]
					U,D,V=mgs(diag(D1)@(V1@U2)@diag(D2))
					V=V@V2
					U=U1@U
					new.append((U,D,V))
				old=new
			# U,D,V=new[0]
			# Vi=triInv(V)
			# Up,Dp,Vp=mgs(transpose(U)@Vi+diag(D))
			# print("min: {min}, max: {max}".format(min=min(np.abs(Dp)),max=max(np.abs(Dp))))
			# ret.append(triInv(Vp@V)@diag(1/Dp)@(transpose(Up)@transpose(U)))

			# U,D,V=new[0]
			# Vi=triInv(V)
			# Up,Dp,Vp=mgs(transpose(U)@Vi+diag(D))
			# print("min: {min}, max: {max}".format(min=min(np.abs(D)),max=max(np.abs(D))))
			# ret.append(triInv(Vp@V)@diag(1/Dp)@(transpose(Up)@diag(D)@transpose(V)))
			M=new[0][0]@diag(new[0][1])@new[0][2]
			v,w=np.linalg.eig(M)
			# print(w@np.diag(v)@np.linalg.inv(w)/M)
			# print(np.diag(v/(1+v)))
			# print("eigs: "+str(v))
			# print("orig:")
			# print(new[0][0]@diag(new[0][1])@new[0][2])
			# print("PDPi:")
			# print(w@np.diag(v/(1+v))@np.linalg.inv(w))
			# print(v/(1+v))
			ret.append(np.real(w@np.diag(v/(1+v))@np.linalg.inv(w)))
			# ret.append(np.linalg.inv(np.eye(len(U))+U@diag(D)@V))
		return ret

	if stabilize:
		mB=[[reduce(lambda a,b:a@b,(B[sig][(taui+j*m+i+1)%ntau] for i in range(m))) for j in range(ntauOverm)] for sig in range(2)]

		ret=[0,0]
		pre=[0,0]
		useSVD=True
		for sigi in range(2):
			# U,D,V=np.linalg.svd(mB[sigi][-1])
			U,D,V=mgs(mB[sigi][-1])
			for i in range(len(mB[sigi])-2,-1,-1):
				M=(mB[sigi][i]@U)@diag(D)
				# U,D,Vp=np.linalg.svd(M)
				if useSVD:
					U,D,Vp=scipy.linalg.svd(M,lapack_driver='gesvd') #default gesdd is not accurate enough, mgs is even better..
				else:
					U,D,Vp=mgs(M)
				# print("Check:")
				# print(min(np.abs(D)))
				# print(max(D))
				# print(np.max(np.abs(U@diag(D)@Vp-M)))
				V=Vp@V
			# Ml[sigi]=U@diag(D)@V
			# Up,Dp,Vp=np.linalg.svd(transpose(U)@triInv(V)+diag(D))
			Ds=[d if abs(d)<1 else 1 for d in D]
			Dbi=[1/d if abs(d)>1 else 1 for d in D]
			
			# print(np.linalg.svd(V)[1])
			# Up,Dp,Vp=mgs(diag(Dbi)@transpose(U)+diag(Ds)@V)
			# Up,Dp,Vp=np.linalg.svd(diag(Dbi)@transpose(U)+diag(Ds)@V)
			# ret[sigi]=triInv(Vp)@diag(1/Dp)@transpose(Up)@diag(Dbi)@transpose(U)
			# print(np.linalg.svd(diag(Dbi)@transpose(U)+diag(Ds)@V)[1])

			# ret[sigi]=np.linalg.inv(diag(Dbi)@transpose(U)+diag(Ds)@V)@(diag(Dbi)@transpose(U))
			H=diag(Dbi)@transpose(U)+diag(Ds)@V
			# print(np.linalg.svd(H)[1])
			# print(V)
			LU,P=scipy.linalg.lu_factor(H)
			# ret[sigi]=np.linalg.solve(H, diag(Dbi)@transpose(U))
			ret[sigi]=scipy.linalg.lu_solve((LU, P), diag(Dbi)@transpose(U))
			pre[sigi]=(H,Dbi,transpose(U))
			# ret[sigi]=transpose(Vp)@(diag(1/Dp)@transpose(Up)@diag(Dbi))@transpose(U)

			# Up,Dp,Vp=mgs(transpose(U)@triInv(V)+diag(D))
			# print("Dp:"+str(D))
			# ret[sigi]=(triInv(V)@triInv(Vp))@diag(1/Dp)@(transpose(Up)@transpose(U))
			# ret[sigi]=(triInv(Vp@V))@diag(1/Dp)@(transpose(Up)@transpose(U))
		# for mb in mB:
			# U,D,V=numpy.linalg.svd(mb)
		
		# Ml=[mB[sigi][0] for sigi in range(2)]
		# for il in range(1,ntauOverm):
		# 	for sigi in range(2):
		# 		Ml[sigi]=Ml[sigi] @ mB[sigi][il]
		if precursor:
			return ret,pre
		else:
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


def dqmc(nx,ny,tx,ty,tne,tnw,U,mu,beta,ntauOverm,m,seed,nWarmupSweeps,nSweeps,stabilize,
	observables,observablesTD=[],stabEps=1e-4,autoCorrN=0,profile=False,returnState=False,startState=None,measurePeriod=1,saveSamples=None):
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
	
	# w,v=np.linalg.eig(Kexp)
	# print(w)
	# _,D,_=np.linalg.svd(Kexp)
	# print(D)

	if startState==None:
		state=[np.array([rand.choice([-1,1]) for i in range(N)]) for tau in range(ntau)]
	else:
		state=startState

	res=[0 for o in observables]
	resTD=[0 for o in observablesTD]

	B=calcBs(Kexp,lamb,deltaTau,state,mu,N,ntau)
	iB=calciBs(iKexp,lamb,deltaTau,state,mu,N,ntau)
	g=calcGFromScratch(Kexp,lamb,state,0,deltaTau,mu,N,m,ntauOverm,B,stabilize)
	attempts=0
	accepted=0
	nMeasures=0
	sign=0
	
	gTimeDep=np.zeros((2,ntau,nx*ny,nx*ny))
	g2TimeDep=np.zeros((2,ntau,ntau,nx*ny,nx*ny))
	if profile: startTime=time.clock()
	for sweep in range(nWarmupSweeps+nSweeps):
		if seed==0:
			print("Progress {:2.1%}".format(sweep/(nWarmupSweeps+nSweeps)), end="\r")
		
		for taui in range(ntau):
			for u in range(nx*ny):
				# x=rand.randrange(nx)
				# y=rand.randrange(ny)
				# ii=x+nx*y
				ii=u
				delta=[np.exp(-2*sigs[sigi]*lamb*state[taui][ii])-1 for sigi in range(2)]
				R=[1+(1-g[sigi][ii,ii])*delta[sigi] for sigi in range(2)]
				Rprod=R[0]*R[1]
				ps=Rprod/(1+Rprod) #heatbath
				sign+=np.sign(ps)
				p=abs(ps)
				attempts+=1
				if p>1 or rand.random()<p:
					accepted+=1
					state[taui][ii]*=-1
					for sigi in range(2):
						# print("g")
						# print(g[sigi])
						# print("dg")
						# print(np.outer([1 if ii==i else 0 for i in range(N)]-g[sigi][::,ii],g[sigi][ii,::]*(delta[sigi]/R[sigi])))
						g[sigi]-=np.outer([1 if ii==i else 0 for i in range(N)]-g[sigi][::,ii],g[sigi][ii,::]*(delta[sigi]/R[sigi]))
			
			tB=calcB(Kexp,lamb,deltaTau,state[taui],mu)
			for sigi in range(2):
				B[sigi][taui]=tB[sigi]
			tiB=calciB(iKexp,lamb,deltaTau,state[taui],mu)
			for sigi in range(2):
				iB[sigi][taui]=tiB[sigi]

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
				if (sweep-nWarmupSweeps)%measurePeriod==0:
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
			# print(np.linalg.inv(B[sigi][(taui+1)%ntau])-iB[sigi][(taui+1)%ntau])
			# gFromScratch=calcGFromScratch(Kexp,lamb,state,(taui)%ntau,deltaTau,mu,N,m,ntauOverm,B,stabilize,special=False)
			# mean=np.mean(np.abs(gFromScratch[sigi]))
			# errors=np.abs(g[sigi]-gFromScratch[sigi])/mean
			
			# maxError=np.max(errors)
			# # print(gFromScratch[sigi]/g[sigi])
			# # print(np.array(g[sigi]))
			
			# if maxError>stabEps:
			# 	print("WARNING1: numerical instability, decrease m. Max relative error of size {e}".format(e=maxError))
			# g=gFromScratch
			
			# gFromScratch,HB=calcGFromScratch(Kexp,lamb,state,(taui)%ntau,deltaTau,mu,N,m,ntauOverm,B,stabilize,special=False,Bi=iB,precursor=True)
			# for sigi in range(2):
			# 	mean=np.mean(np.abs(gFromScratch[sigi]))
			# 	errors=np.abs(g[sigi]-gFromScratch[sigi])/mean
			
			# 	maxError=np.max(errors)
		
			
			# 	if maxError>stabEps:
			# 		print("WARNING: g from updates numerical instability. Max relative error of size {e}".format(e=maxError))

			# g=gFromScratch

			g=[ iB[sigi][(taui+1)%ntau]@g[sigi]@B[sigi][(taui+1)%ntau] for sigi in range(2)]	
			# g=[ iB[sigi][(taui+1)%ntau]@np.linalg.solve( HB[sigi][0], diag(HB[sigi][1])@(HB[sigi][2]@B[sigi][(taui+1)%ntau])  ) for sigi in range(2)]
		
			if stabilize and (taui+1)%m==0:

				gFromScratch=calcGFromScratch(Kexp,lamb,state,(taui+1)%ntau,deltaTau,mu,N,m,ntauOverm,B,stabilize,special=False,Bi=iB)
				for sigi in range(2):
					mean=np.mean(np.abs(gFromScratch[sigi]))
					errors=np.abs(g[sigi]-gFromScratch[sigi])/mean
					
					maxError=np.max(errors)
					# print(np.array(g[sigi]))
					
					if maxError>stabEps:
						print("WARNING3: numerical instability, decrease m. Max relative error of size {e}".format(e=maxError))
					#print(mean)
				g=gFromScratch
				# TODO: warn "m too large" if propagated g differs too much from this
		if (sweep-nWarmupSweeps)%measurePeriod==0 and saveSamples!=None:
			for taui in range(0,ntau):
				for i in range(0,ntau):
					if i==0:
						g2TimeDep[sigi,taui,i]=g[sigi]
					else:
						Ul,Dl,Vl=stableProd([B[sigi][(taui+i-j)%ntau] for j in range(i)],N,m)
						Ur,Dr,Vr=stableProd([B[sigi][(taui+ntau-j)%ntau] for j in range(ntau-i)],N,m)
						Up,Dp,Vp=np.linalg.svd(diag(1/Dl)@(transpose(Ul)@transpose(Vr))+(Vl@Ur)@diag(Dr))
						g2TimeDep[sigi,taui,i]=(transpose(Vr)@transpose(Vp))@diag(1/Dp)@(transpose(Up)@Vl)
			saveSamples(sweep, np.sign(ps), g2TimeDep)
		if False:
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

def getDirName(nx,ny,nTau,tx,ty,tnw,tne,U,mu,beta,m,measurePeriod):
	return ("samples/nx={nx},ny={ny},ntau={nTau},tx={tx},tx={ty},tnw={tnw},"+
			"tne={tne},U={U},mu={mu},beta={beta},m={m},period={measurePeriod}").format(
				nx=nx,ny=ny,nTau=nTau,tx=tx,ty=ty,tnw=tnw,tne=tne,U=U,mu=mu,beta=beta,m=m,measurePeriod=measurePeriod)

def genGSamples(nx,ny,tx,ty,tnw,tne,U,mu,beta,m,tausPerBeta,nSweepsPerRun,measurePeriod,nRuns=1,nThreads=1,startSeed=0):
	ntau=math.ceil(beta*tausPerBeta/m)*m

	dirname=getDirName(nx,ny,ntau,tx,ty,tnw,tne,U,mu,beta,m,measurePeriod)

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
		file=os.path.join(dirname, str(seed)+".npz")
		if os.path.exists(file):
			print("ERROR: run already exists, "+file)
		else:
			dqmc(nx,ny,tx,ty,tne,tnw,U,mu,beta,ntau//m,m,seed,0,nSweepsPerRun,True,
				[],measurePeriod=measurePeriod,saveSamples=save)
			np.savez_compressed(file, sweeps=sweeps, sgns=sgns, gs=gs)

def loadRuns(nx,ny,ntau,tx,ty,tnw,tne,U,mu,beta,m,measurePeriod):
	import os
	from os.path import isfile, join
	path=getDirName(nx,ny,ntau,tx,ty,tnw,tne,U,mu,beta,m,measurePeriod)
	files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
	print("Found {n} previous runs.".format(n=len(files)))
	runs=[]
	for f in files:
		loaded = np.load(os.path.join(path, f))
		runs.append((loaded['sweeps'],loaded['sgns'],loaded['gs']))

	return runs

def averageOverG(runs,op):
	opRes=[]
	sgnRes=[]
	for run in runs:
		opAcc=0
		sgnAcc=0
		for g in run[2]:
			opAcc+=op(g)
		for sign in run[1]:
			sgnAcc+=sign
		opRes.append(opAcc/len(run))
		sgnRes.append(sgnAcc/len(run))
	opVar=np.std(opRes,axis=0)**2/len(runs)
	sgnVar=np.std(sgnRes,axis=0)**2/len(runs)
	# take op-sgn covariance into account too...
	return (np.mean(opRes,axis=0)/p.mean(sgnRes,axis=0),-1)

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