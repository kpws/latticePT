import numpy as np
from scipy.signal import convolve2d
from scipy.sparse.linalg import eigs, LinearOperator

class greensFunction:
	def __init__(self, stat, T,v=None):
		self.stat=stat
		self.T=T
		if v is not None:
			self.v=np.array(v)
			assert len(v)%2==(stat+1)//2
			self.Nc=(len(v)+(1+self.stat)//2)//2
			if len(self.v.shape)>1: self.Nx=self.v.shape[1]
			if len(self.v.shape)>2: self.Ny=self.v.shape[2]

	def fromFun(self,Nc,Nx,Ny,f):
		self.Nc=Nc
		self.Nx=Nx
		self.Ny=Ny
		ws=[self.getw(i) for i in range(self.getNt())]
		kxs=[self.getk(Nx,i) for i in range(Nx)]
		kys=[self.getk(Ny,i) for i in range(Ny)]
		self.v=np.array([[[f(w,kx,ky) for ky in kys] for kx in kxs] for w in ws])
		return self

	def getw(self,n):
		return greensFunction.getw(self.Nc,self.stat,self.T,n)

	@staticmethod
	def getw(Nc,stat,T,n):
		return (-2*Nc+(1-stat)//2+2*n)*np.pi*T

	# Nx,Ny are even, i=Nx/2-1 is k=0. Eg. Nx=4: k=pi*(-.5, 0, .5, 1)
	def getk(self,Nk,n):
		return -np.pi+2*np.pi*(n+1)//Nk

	def getNt(self):
		return 2*self.Nc-(1+self.stat)//2

	def getMinInHalfStep(self,stat):
		#number of half discrete energy steps the min frequency is below 0
		return 2*self.Nc-2 if stat==1 else 2*self.Nc-1

	def __mul__(self,g2,fft=True):
		#convolves such that momentum is conserved, f(q)=sum_k g(k)*h(q-k)
		assert self.Nc==g2.Nc and self.Nx==g2.Nx and self.Ny==g2.Ny and self.T==g2.T
		stat=self.stat*g2.stat
		Nt=2*self.Nc-(1+stat)//2
		if fft:
			z1=np.zeros([self.getNt()+self.stat]+list(self.v.shape[1:]))
			z2=np.zeros([g2.getNt()+g2.stat]+list(self.v.shape[1:]))
			trim=self.getNt()+len(z1)-Nt
			if self.stat==-1 and g2.stat==-1:
				trim1=(trim)//2
				trim2=(trim)//2
			if self.stat==1 and g2.stat==1:
				trim1=(trim)//2-1
				trim2=(trim)//2+1
			if self.stat==-1 and g2.stat==1:
				trim1=(trim-1)//2
				trim2=(trim+1)//2
			if self.stat==1 and g2.stat==-1:
				trim1=(trim-1)//2
				trim2=(trim+1)//2
			res=np.fft.ifftn(np.fft.fftn(np.concatenate((self.v,z1)))*np.fft.fftn(np.concatenate((g2.v,z2))))
			v=np.roll(np.roll(res[trim1:len(res)-trim2],-((self.Nx+1*-1)//2),axis=1),-((self.Ny+1*-1)//2),axis=2)
		else:
			if stat==1:
				mt=(Nt-1)//2
			if stat==-1:
				if g2.stat:
					mt=Nt//2-1
				else:
					mt=Nt//2-1
			#consider i,j==0
			Dt=(-self.getMinInHalfStep(stat)+self.getMinInHalfStep(self.stat)+self.getMinInHalfStep(g2.stat))//2
			v=np.array([sum(greensFunction.convolvePeriodic(self.v[j],g2.v[i-j+Dt]) for j in range(max(0,i+Dt-(g2.getNt()-1)),min(self.getNt(),i+Dt+1))) for i in range(Nt)])
		return greensFunction(stat,self.T,v=self.T*v/self.Nx/self.Ny)

	def __rmul__(self,f):
		return greensFunction(self.stat,self.T,v=f*self.v)

	@staticmethod
	def convolvePeriodic(a,b):
		#print (a,b)
		return np.roll(np.roll(convolve2d(a,b,boundary='wrap',mode='same'),-1*0,axis=0),-1*0,axis=1)
		#return a*b

	def dot(self,g2,con=True):
		assert self.compatible(g2)
		if con:
			return np.tensordot(self.v,np.conjugate(g2.v),axes=([0,1,2],[0,1,2]))
		else:
			return np.tensordot(self.v,g2.v,axes=([0,1,2],[0,1,2]))

	def reverse(self):
		#return greensFunction(self.stat,self.T,v=np.flip(np.flip(np.flip(self.v,0),1),2))
		return greensFunction(self.stat,self.T,v=np.roll(np.roll(self.v[::-1,::-1,::-1],-1,axis=1),-1,axis=2))

	def __pow__(self,g2):
		assert self.compatible(g2)
		return greensFunction(self.stat,self.T,v=self.v*g2.v)

	def __truediv__(self,g2):
		if isinstance(g2,greensFunction):
			assert self.compatible(g2)
			return greensFunction(self.stat,self.T,v=self.v/g2.v)
		else:
			return greensFunction(self.stat,self.T,v=self.v/g2)

	def __neg__(self):
		return greensFunction(self.stat,self.T,v=-self.v)

	def __add__(self,g2):
		if isinstance(g2,greensFunction):
			assert self.compatible(g2)
			return greensFunction(self.stat,self.T,v=self.v+g2.v)
		else:
			return greensFunction(self.stat,self.T,v=self.v+g2)
	
	__radd__=__add__

	def __sub__(self,g2):
		if isinstance(g2,greensFunction):
			assert self.compatible(g2)
			return greensFunction(self.stat,self.T,v=self.v-g2.v)
		else:
			return greensFunction(self.stat,self.T,v=self.v-g2)

	def __rsub__(self,g2):
		if isinstance(g2,greensFunction):
			assert self.compatible(g2)
			return greensFunction(self.stat,self.T,v=g2.v-self.v)
		else:
			return greensFunction(self.stat,self.T,v=g2-self.v)

	def __str__(self):
		return str(('bosonic' if self.stat==1 else 'fermionic',self.T,self.v))

	def compatible(self,g2):
		return self.Nc==g2.Nc and self.T==g2.T and self.stat==g2.stat and self.Nx==g2.Nx and self.Ny==g2.Ny

def test():
	Nx=12
	Ny=4
	Ntb=5
	Ntf=Ntb+1
	f1=greensFunction(-1,1.,v=np.random.rand(Ntf,Nx,Ny))
	f2=greensFunction(-1,1.,v=np.random.rand(Ntf,Nx,Ny))
	b1=greensFunction(1,1.,v=np.random.rand(Ntb,Nx,Ny))
	b2=greensFunction(1,1.,v=np.random.rand(Ntb,Nx,Ny))
	for (g,name) in [(f1,'f'), (b1,'b')]:
		print(name+':')
		a1=g+(-1*g)
		assert a1.dot(a1)<1e-15; print(' * passed g+(-1*g) = 0')
		a1=g+(-g)
		assert a1.dot(a1)<1e-15; print(' * passed (g)+(-g) = 0')
		a1=g-g
		assert a1.dot(a1)<1e-15; print(' * passed  g-g = 0')
		a1=g.reverse().reverse()-g
		assert a1.dot(a1)<1e-15; print(' * passed  double reverse')
		if g.stat==1:
			assert (g-g.reverse()).v[(Ntb-1)//2,Nx//2-1,Ny//2-1]==0;
		else:
			assert (g.reverse().v[Ntf//2,Nx//2-1,Ny//2-1]==g.v[Ntf//2-1,Nx//2-1,Ny//2-1]
					and
					g.reverse().v[Ntf//2-1,Nx//2-1,Ny//2-1]==g.v[Ntf//2,Nx//2-1,Ny//2-1]);
		print(' * passed reverse center fixed')
	for (g1,g2,name) in [(f1,f2,'ff'), (b1,b2,'bb'), (f1,b1,'fb'), (b2,f1,'bf')]:
		print(name+':')
		a1=g1.__mul__(g2,fft=False)
		a2=g2.__mul__(g1,fft=False)
		assert (a1-a2).dot((a1-a2))<1e-15; print(' * passed reverse')
		a3=g1.__mul__(g2,fft=True)
		assert (a1-a3).dot((a1-a3))<1e-15; print(' * passed fft')
		a4=g2.__mul__(g1,fft=True)
		assert (a1-a4).dot((a1-a4))<1e-15; print(' * passed fft reverse')
		if a1.stat==1:
			assert abs((g1*g2).v[(Ntb-1)//2,Nx//2-1,Ny//2-1]*Nx*Ny/g1.T-g1.dot(g2.reverse()))<1e-10; print(' * passed dot-reverse/conv comparison')

def getLambdas(Nc,Nx,Ny,tx,ty,t2,filling,T,linearized=False,U=None,St=None):
	kxs=np.array([[-np.pi+(i+1)*2*np.pi/Nx for i in range(Nx)],]*Ny).transpose()
	kys=np.array([[-np.pi+(i+1)*2*np.pi/Ny for i in range(Ny)],]*Nx)
	wsf=[greensFunction.getw(Nc,-1,T,i) for i in range(2*Nc)]
	wsb=[greensFunction.getw(Nc,1,T,i) for i in range(2*Nc-1)]
	print(wsf)
	if linearized: #half filling, low energy limit
		eps=-2*tx*(np.abs(kxs)-np.pi/2)
	else:
		mu=0
		muu=2*tx+2*ty+2*t2
		mul=-muu
		ok=False
		Npgoal=int(filling*Nx*Ny)
		while not ok:
			eps=-2*tx*np.cos(kxs)-2*ty*np.cos(kys)-2*t2*np.cos(kxs+kys)-mu
			Np=(0 >= eps).sum()
			Nh=(0 < eps).sum()
			assert Np+Nh==Nx*Ny
			if Npgoal==Np or abs(muu-mul)<1e-9:
				ok=True
			if Np<Npgoal:
				mul=mu
				mu=(mu+muu)/2
			if Np>Npgoal:
				muu=mu
				mu=(mu+mul)/2

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

