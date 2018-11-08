import numpy as np
from scipy.signal import convolve2d
from scipy.sparse.linalg import eigs, LinearOperator
from greensFunction import *

def getLambdas(Nc,Nx,Ny,tx,ty,t2,filling,T,linearized=False,U=None,St=None,returnU=False):
	kxs=np.array([[-np.pi+(i+1)*2*np.pi/Nx for i in range(Nx)],]*Ny).transpose()
	kys=np.array([[-np.pi+(i+1)*2*np.pi/Ny for i in range(Ny)],]*Nx)
	wsf=[greensFunction.getw(Nc,-1,T,i) for i in range(2*Nc)]
	wsb=[greensFunction.getw(Nc,1,T,i) for i in range(2*Nc-1)]
	if linearized: #half filling, low energy limit
		eps=-2*tx*(np.abs(kxs)-np.pi/2)
	else:
		mu=0
		muu=2*tx+2*ty+2*t2
		mul=-muu
		ok=False
		n=400
		kxsf=np.array([[-np.pi+(i+1)*2*np.pi/n for i in range(n)],]*n).transpose()
		kysf=np.array([[-np.pi+(i+1)*2*np.pi/n for i in range(n)],]*n)
	
		Npgoal=int(filling*n*n)
		while not ok:
			eps=-2*tx*np.cos(kxsf)-2*ty*np.cos(kysf)-2*t2*np.cos(kxsf+kysf)-mu
			Np=(0 >= eps).sum()
			Nh=(0 < eps).sum()
			assert Np+Nh==n*n
			if Npgoal==Np or abs(muu-mul)<1e-9:
				ok=True
			if Np<Npgoal:
				mul=mu
				mu=(mu+muu)/2
			if Np>Npgoal:
				muu=mu
				mu=(mu+mul)/2

		eps=-2*tx*np.cos(kxs)-2*ty*np.cos(kys)-2*t2*np.cos(kxs+kys)-mu
	mu=0
	# pl.figure(0)
	# plotBZ(eps)
	# pl.draw()

	G=greensFunction(-1,T,v=[1/(1j*w-eps) for w in wsf])
	chi0=-G*(G.reverse())


	if St!=None:
		chi0max=chi0.v.max()
		U=St/chi0max.real
		#print('U=%s'%(U/tx))

	chis=chi0/(1-U*chi0)
	chic=chi0/(1+U*chi0)
	Vsa=U+(3./2)*(U**2)*chis-(1./2)*(U**2)*chic
	Vta=-1./2*(U**2)*chis-1./2*(U**2)*chic

	v=greensFunction(-1,T,v=np.random.rand(2*Nc,Nx,Ny))

	G2=G**(G.reverse())

	Nvals=8
	Ntf=2*Nc
	lambdas=[[[],[]],[[],[]]]
	for S in [0,1]:
		print(S)
		def linOp(x):
			return (-(Vta if S==1 else Vsa)*(G2**greensFunction(-1,T,v=x.reshape(Ntf,Nx,Ny)))).v.flatten()		
		A=LinearOperator((Ntf*Nx*Ny,Ntf*Nx*Ny),linOp)
		vals, vecs = eigs(A, k=Nvals,tol=1e-6,which='LR')
		inds = (-vals.real).argsort()

		for j in range(Nvals):
			i=inds[j]
			vi=vecs[:,i].reshape(Ntf,Nx,Ny)
			Tsym=vi[Ntf*3//4-1,Nx*3//4-1,Ny*3//4-1]/vi[Ntf*1//4-1,Nx*3//4-1,Ny*3//4-1]
			Psym=vi[Ntf*3//4-1,Nx*3//4-1,Ny*3//4-1]/vi[Ntf*3//4-1,Nx*1//4-1,Ny*1//4-1]
			if Tsym*Psym*(-1 if S==0 else 1)>0: continue
			lambdas[S][0 if Tsym>0 else 1].append(vals[j])
		for tsym in [0,1]:
			if len(lambdas[S][tsym])==0:
				lambdas[S][tsym].append(0)
	if returnU:
		return lambdas, U
	else:
		return lambdas

def gl(a):
	Nc=a[0]
	Nx=a[1]
	Ny=a[2]
	tx=a[3]
	x=a[4]
	filling=a[5]
	T=a[6]
	U=a[7]
	St=a[8]
	linearized=a[9]
	return getLambdas(Nc,Nx,Ny,tx,tx*x,tx*x,filling,T,linearized=linearized,U=U,St=St)

def main2():
	Nc=256
	Nx=128
	Ny=1
	tx=1
	filling=.5
	# U=1.6*tx
	U=2.5
	St=None
	Ts=np.linspace(0.01,.2,10)
	# G=greensFunction(-1,T,v=[1/(1j*w-eps) for w in wsf])
	# chi0=-G*(G.reverse())


	# if St!=None:
	# 	chi0max=chi0.v.max()
	# 	U=St/chi0max.real

	# print('mu='+str(mu))
	# print('U='+str(U))
	# return

	from multiprocessing import Pool
	p=Pool(processes=len(Ts)+1)
	# lambdas=p.map(gl,[(Nc,Nx,Ny,tx,x,filling,T,U,St) for x in xs])
	lambdasDisp=p.map(gl,[(Nc,Nx,Ny,tx,0,filling,T,U,None,False) for T in Ts])
	lambdasLin=p.map(gl,[(Nc,Nx,Ny,tx,0,filling,T,U,None,True) for T in Ts])
	
	p.terminate()
	import matplotlib
	# matplotlib.use('Agg')
	from matplotlib import rc
	rc('text', usetex=True)
	import pylab as pl
	for S in [0,1]:
		for Tsym in [0,1]:
			pl.plot(Ts,[np.real(l[S][Tsym][0]) for l in lambdasDisp],marker='.',label='$\\mathrm{Full\\ '+('E' if (S-.5)*(.5-Tsym)<0 else 'O')+('T' if S==1 else 'S')+('O' if Tsym==1 else 'E')+'}$')
			pl.plot(Ts,[np.real(l[S][Tsym][0]) for l in lambdasLin],marker='.',label='$\\mathrm{Linearized\\ '+('E' if (S-.5)*(.5-Tsym)<0 else 'O')+('T' if S==1 else 'S')+('O' if Tsym==1 else 'E')+'}$')
	pl.legend()
	pl.xlim([Ts[0],Ts[-1]])
	pl.ylim([0,1])
	# pl.xlabel('$t_y/t_x=t_2/t_x$')
	pl.xlabel('$T/t_x$')
	pl.ylabel('$\lambda$')
	if St==None:
		title='N_c={0},\\ N_x={1},\\ N_y={2},\\ \\nu={4},\\ U={5}t_x'.format(Nc,Nx,Ny,0,filling,U/tx)
	else:
		title='N_c={0},\\ N_x={1},\\ N_y={2},\\ \\nu={4},\\ S_t={5}'.format(Nc,Nx,Ny,0,filling,St)
	pl.title('$'+title+'$')
	pl.savefig('lambdas_'+title.replace('\\','')+'.pdf', bbox_inches='tight')
	pl.show()


def main3():
	# Nc=128
	# Nx=128
	# Ny=64

	Nc=512
	Nx=128
	Ny=1
	tx=1
	filling=.5
	# U=1.6*tx
	U=None
	St=.97
	T=0.005*tx
	

	# #test if small T gives small U
	# ty=0.05
	# t2=0.05
	# mu=0
	# muu=2*tx+2*ty+2*t2
	# mul=-muu
	# ok=False
	# Npgoal=int(filling*Nx*Ny)
	# kxs=np.array([[-np.pi+(i+1)*2*np.pi/Nx for i in range(Nx)],]*Ny).transpose()
	# kys=np.array([[-np.pi+(i+1)*2*np.pi/Ny for i in range(Ny)],]*Nx)
	# wsf=[greensFunction.getw(Nc,-1,T,i) for i in range(2*Nc)]
	# wsb=[greensFunction.getw(Nc,1,T,i) for i in range(2*Nc-1)]
	# while not ok:
	# 	eps=-2*tx*np.cos(kxs)-2*ty*np.cos(kys)-2*t2*np.cos(kxs+kys)-mu
	# 	Np=(0 >= eps).sum()
	# 	Nh=(0 < eps).sum()
	# 	assert Np+Nh==Nx*Ny
	# 	if Npgoal==Np or abs(muu-mul)<1e-9:
	# 		ok=True
	# 	if Np<Npgoal:
	# 		mul=mu
	# 		mu=(mu+muu)/2
	# 	if Np>Npgoal:
	# 		muu=mu
	# 		mu=(mu+mul)/2
	

	# pl.figure(0)
	# plotBZ(eps)
	# pl.draw()

	G=greensFunction(-1,T,v=[1/(1j*w-eps) for w in wsf])
	chi0=-G*(G.reverse())


	if St!=None:
		chi0max=chi0.v.max()
		U=St/chi0max.real

	print('mu='+str(mu))
	print('U='+str(U))
	return

	xs=np.linspace(0,.1,2)
	from multiprocessing import Pool
	p=Pool(processes=len(xs)+1)
	# lambdas=p.map(gl,[(Nc,Nx,Ny,tx,x,filling,T,U,St) for x in xs])
	lambdas=p.map(gl,[(Nc,Nx,Ny,tx,0,filling,T,U,St,l) for l in [False,True]])
	xs=[0,1]
	p.terminate()
	import matplotlib
	# matplotlib.use('Agg')
	from matplotlib import rc
	rc('text', usetex=True)
	import pylab as pl
	for S in [0,1]:
		for Tsym in [0,1]:
			print([np.real(l[S][Tsym][0]) for l in lambdas])
			pl.plot(xs,[np.real(l[S][Tsym][0]) for l in lambdas],marker='.',label='$\\mathrm{'+('E' if (S-.5)*(.5-Tsym)<0 else 'O')+('T' if S==1 else 'S')+('O' if Tsym==1 else 'E')+'}$')
	pl.legend()
	pl.xlim([xs[0],xs[-1]])
	pl.ylim([0,1])
	# pl.xlabel('$t_y/t_x=t_2/t_x$')
	pl.xlabel('[Full dispersion, Linearized]')
	pl.ylabel('$\lambda$')
	if St==None:
		title='N_c={0},\\ N_x={1},\\ N_y={2},\\ T={3}t_x,\\ \\nu={4},\\ U={5}t_x'.format(Nc,Nx,Ny,T/tx,filling,U/tx)
	else:
		title='N_c={0},\\ N_x={1},\\ N_y={2},\\ T={3}t_x,\\ \\nu={4},\\ S_t={5}'.format(Nc,Nx,Ny,T/tx,filling,St)
	pl.title('$'+title+'$')
	pl.savefig('lambdas_'+title.replace('\\','')+'.pdf', bbox_inches='tight')
	pl.show()


def main():
	# Nc=2048
	# Nx=128
	# Ny=64
	Nc=2048
	Nx=512
	Ny=64
	tx=1

	# filling=.5
	# x=.4
	# T=0.04*tx
	# St=.97
	# ty=x*tx
	# t2=x*tx

	# filling=.5
	# x=.1
	# T=0.05*tx
	# St=.95
	# ty=x*tx
	# t2=x*tx

	filling=.5
	x=.05
	U=1.6*tx
	T=0.06*tx
	ty=x*tx
	t2=x*tx
	St=None


	kxs=np.array([[-np.pi+(i+1)*2*np.pi/Nx for i in range(Nx)],]*Ny).transpose()
	kys=np.array([[-np.pi+(i+1)*2*np.pi/Ny for i in range(Ny)],]*Nx)
	wsf=[greensFunction.getw(Nc,-1,T,i) for i in range(2*Nc)]
	wsb=[greensFunction.getw(Nc,1,T,i) for i in range(2*Nc-1)]

	mu=0
	muu=2*tx+2*ty+2*t2
	mul=-muu
	ok=False
	Npgoal=int(filling*Nx*Ny)
	print("Tuning filling fraction... (nu=%s => Np=%s)"%(filling, Npgoal))
	while not ok:
		eps=-2*tx*np.cos(kxs)-2*ty*np.cos(kys)-2*t2*np.cos(kxs+kys)-mu
		Np=(0 > eps).sum()
		Nh=(0 < eps).sum()
		assert Np+Nh==Nx*Ny
		print("%s<mu<%s,  %s"%(mul,muu,Np))
		if Npgoal==Np:
			ok=True
		if Np<Npgoal:
			mul=mu
			mu=(mu+muu)/2
		if Np>Npgoal:
			muu=mu
			mu=(mu+mul)/2

	import pylab as pl
	def plotBZ(z):
		pl.imshow(np.transpose(z), extent=(-np.pi, np.pi, -np.pi, np.pi),origin='lower')
		CS =pl.contour(kxs,kys,z,colors='k')
		pl.clabel(CS, fontsize=9, inline=1)

	# pl.figure(0)
	# plotBZ(eps)
	# pl.draw()

	print("Generating G...")
	G=greensFunction(-1,T,v=[1/(1j*w-eps) for w in wsf])
	print("Calculating chi...")
	chi0=-G*(G.reverse())


	if St!=None:
		chi0max=chi0.v.max()
		print("chi0 max: "+str(chi0max))
		U=St/chi0max.real
		print("St=%s => U= "%St+str(U/tx)+' tx')

	print('Nc=%s'%Nc)
	print('Nx=%s'%Nx)
	print('Ny=%s'%Ny)
	print('T=%stx'%(T/tx))
	print('U=%stx'%(U/tx))
	print('ty=%stx'%(ty/tx))
	print('t2=%stx'%(t2/tx))

	chis=chi0/(1-U*chi0)
	chic=chi0/(1+U*chi0)
	Vsa=U+(3./2)*(U**2)*chis-(1./2)*(U**2)*chic
	Vta=-1./2*(U**2)*chis-1./2*(U**2)*chic

	print("Finding linearized gap equation solutions...")
	v=greensFunction(-1,T,v=np.random.rand(2*Nc,Nx,Ny))

	G2=G**(G.reverse())

	Nvals=8
	Ntf=2*Nc
	for S in [0,1]:
		if S==0:
			print('Singlet:')
		else:
			print('Triplet:')
		i=0
		def linOp(x):
			print('{0}\r'.format(i), end=' ')
			# i=i+1
			return (-(Vta if S==1 else Vsa)*(G2**greensFunction(-1,T,v=x.reshape(Ntf,Nx,Ny)))).v.flatten()		
		A=LinearOperator((Ntf*Nx*Ny,Ntf*Nx*Ny),linOp)
		vals, vecs = eigs(A, k=Nvals,tol=1e-6,which='LR')
		inds = (-vals.real).argsort()

		for j in range(Nvals):
			i=inds[j]
			vi=vecs[:,i].reshape(Ntf,Nx,Ny)
			Tsym=vi[Ntf*3/4-1,Nx*3/4-1,Ny*3/4-1]/vi[Ntf*1/4-1,Nx*3/4-1,Ny*3/4-1]
			Psym=vi[Ntf*3/4-1,Nx*3/4-1,Ny*3/4-1]/vi[Ntf*3/4-1,Nx*1/4-1,Ny*1/4-1]
			if Tsym*Psym*(2*S-1)<0:
				print(' '+('O' if Tsym.real<0 else 'E')+('T' if S==1 else 'S')+('O' if Psym.real<0 else 'E')+' lambda = %s'%vals[i])
			else:
				print('('+('O' if Tsym.real<0 else 'E')+('T' if S==1 else 'S')+('O' if Psym.real<0 else 'E')+')')
			# pl.figure(2*i+1)
			# plotBZ(Nx*Ny*vi[Nc].real)
			# pl.figure(2*i+2)
			# plotBZ(Nx*Ny*vi[Nc-1].real)
		pl.show()
	return


if __name__ == "__main__":
	test()
	main2()

