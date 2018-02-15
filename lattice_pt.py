import numpy as np
from scipy.signal import convolve2d
from scipy.sparse.linalg import eigs, LinearOperator

class fermionicFun:
	def __init__(self, Nc, T, f):
		v=f(np.linspace(-(2*Nc-1)*np.pi*T,(2*Nc-1)*np.pi*T,2*Nc))

class greensFunction:
	def __init__(self, stat, T,v=None):
		self.stat=stat
		self.T=T
		if v!=None:
			self.v=np.array(v)
			assert len(v)%2==(stat+1)/2
			self.Nc=(len(v)+(1+self.stat)/2)/2
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

	def getGrid():
		if self.stat==1:
			return np.linspace(-(2*self.Nc)*np.pi*self.T,(2*self.Nc)*np.pi*self.T,2*self.Nc-1)
		if self.stat==-1:
			return np.linspace(-(2*self.Nc-1)*np.pi*self.T,(2*self.Nc-1)*np.pi*self.T,2*self.Nc)

	def getw(self,n):
		return greensFunction.getw(self.Nc,self.stat,self.T,n)

	@staticmethod
	def getw(Nc,stat,T,n):
		return (-2*Nc+(1-stat)/2+2*n)*np.pi*T

	# Nx,Ny are even, i=Nx/2-1 is k=0. Eg. Nx=4: k=pi*(-.5, 0, .5, 1)
	def getk(self,Nk,n):
		return -np.pi+2*np.pi*(n+1)/Nk

	def getNt(self):
		return 2*self.Nc-(1+self.stat)/2

	def getMinInHalfStep(self,stat):
		#number of half discrete energy steps the min frequency is below 0
		return 2*self.Nc-2 if stat==1 else 2*self.Nc-1

	def __mul__(self,g2,fft=True):
		#convolves such that momentum is conserved, f(q)=sum_k g(k)*h(q-k)
		assert self.Nc==g2.Nc and self.Nx==g2.Nx and self.Ny==g2.Ny and self.T==g2.T
		stat=self.stat*g2.stat
		Nt=2*self.Nc-(1+stat)/2
		if fft:
			z1=np.zeros([self.getNt()+self.stat]+list(self.v.shape[1:]))
			z2=np.zeros([g2.getNt()+g2.stat]+list(self.v.shape[1:]))
			trim=self.getNt()+len(z1)-Nt
			if self.stat==-1 and g2.stat==-1:
				trim1=(trim)/2
				trim2=(trim)/2
			if self.stat==1 and g2.stat==1:
				trim1=(trim)/2-1
				trim2=(trim)/2+1
			if self.stat==-1 and g2.stat==1:
				trim1=(trim-1)/2
				trim2=(trim+1)/2
			if self.stat==1 and g2.stat==-1:
				trim1=(trim-1)/2
				trim2=(trim+1)/2
			res=np.fft.ifftn(np.fft.fftn(np.concatenate((self.v,z1)))*np.fft.fftn(np.concatenate((g2.v,z2))))
			v=np.roll(np.roll(res[trim1:len(res)-trim2],-((self.Nx+1*-1)/2),axis=1),-((self.Ny+1*-1)/2),axis=2)
		else:
			if stat==1:
				mt=(Nt-1)/2
			if stat==-1:
				if g2.stat:
					mt=Nt/2-1
				else:
					mt=Nt/2-1
			#consider i,j==0
			Dt=(-self.getMinInHalfStep(stat)+self.getMinInHalfStep(self.stat)+self.getMinInHalfStep(g2.stat))/2
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

	def __div__(self,g2):
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

def eps(kx,ky,couplings):
	tx,ty,t2,U=couplings
	return -2*tx*np.cos(kx)-2*ty*np.cos(ky)-2*t2*np.cos(kx+ky)

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
		print name+':'
		a1=g+(-1*g)
		assert a1.dot(a1)<1e-15; print ' * passed g+(-1*g) = 0'
		a1=g+(-g)
		assert a1.dot(a1)<1e-15; print ' * passed (g)+(-g) = 0'
		a1=g-g
		assert a1.dot(a1)<1e-15; print ' * passed  g-g = 0'
		a1=g.reverse().reverse()-g
		assert a1.dot(a1)<1e-15; print ' * passed  double reverse'
	for (g1,g2,name) in [(f1,f2,'ff'), (b1,b2,'bb'), (f1,b1,'fb'), (b2,f1,'bf')]:
		print name+':'
		a1=g1.__mul__(g2,fft=False)
		a2=g2.__mul__(g1,fft=False)
		assert (a1-a2).dot((a1-a2))<1e-15; print ' * passed reverse'
		a3=g1.__mul__(g2,fft=True)
		assert (a1-a3).dot((a1-a3))<1e-15; print ' * passed fft'
		a4=g2.__mul__(g1,fft=True)
		assert (a1-a4).dot((a1-a4))<1e-15; print ' * passed fft reverse'
		if a1.stat==1:
			assert abs((g1*g2).v[(Ntb-1)/2,Nx/2-1,Ny/2-1]*Nx*Ny/g1.T-g1.dot(g2.reverse()))<1e-10; print ' * passed dot-reverse/conv comparison'

def arnoldi_iteration(A,b,nimp):
	"""
	Input
	A: (nxn matrix)
	b: (initial vector)
	nimp: number of iterations
	"""

	h = np.zeros((nimp+1, nimp),dtype=complex)    # Creates a zero matrix of shape (n+1)x n

	q  = b/np.sqrt(b.dot(b))      # Normilize the intput vector
	Q = [q]                     # Adds q to the first column of Q

	for n in range(nimp):
		print 'Arnoldi n=%s'%n           
		v = A(q)                # A*q_0
		for j in range(n+1):
			h[j, n] = Q[j].dot(v,con=False) #maybe without cc?      
			# print h[j, n]
			v = v - h[j,n]*Q[j]   

		h[n+1, n] = np.sqrt(v.dot(v))
		q = v / h[n+1, n]
		Q.append(q)
	return Q, h

def main():
	# Nc=2048
	# Nx=128
	# Ny=64
	Nc=256
	Nx=128
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
	x=.1
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
	print "Tuning filling fraction... (nu=%s => Np=%s)"%(filling, Npgoal)
	while not ok:
		eps=-2*tx*np.cos(kxs)-2*ty*np.cos(kys)-2*t2*np.cos(kxs+kys)-mu
		Np=(0 > eps).sum()
		Nh=(0 < eps).sum()
		assert Np+Nh==Nx*Ny
		print "%s<mu<%s,  %s"%(mul,muu,Np)
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

	pl.figure(0)
	plotBZ(eps)
	pl.draw()

	print "Generating G..."
	G=greensFunction(-1,T,v=[1/(1j*w-eps) for w in wsf])
	print "Calculating chi..."
	chi0=-G*(G.reverse())


	if St!=None:
		chi0max=chi0.v.max()
		print "chi0 max: "+str(chi0max)
		U=St/chi0max.real
		print " => U= "+str(U/tx)+' tx'

	chis=chi0/(1-U*chi0)
	chic=chi0/(1+U*chi0)
	Vsa=U+(3./2)*(U**2)*chis-(1./2)*(U**2)*chic
	Vta=-1./2*(U**2)*chis-1./2*(U**2)*chic

	# pl.figure(3)
	# CS =pl.contour(kxs,kys,chis.v.real[Nc-1],colors='k')
	# pl.clabel(CS, fontsize=9, inline=1)
	# pl.show()

	# pl.figure(3)
	# #Q=(pi,pi/2)
	# pl.plot(wsb,Vsa.v[:,Nx-1,Ny*3/4-1].real)
	# pl.show()

	print "Finding linearized gap equation solutions..."
	v=greensFunction(-1,T,v=np.random.rand(2*Nc,Nx,Ny))

	G2=G**(G.reverse())

	Nvals=4
	Ntf=2*Nc
	def linOp(x):
		print '*'
		return (-Vsa*(G2**greensFunction(-1,T,v=x.reshape(Ntf,Nx,Ny)))).v.flatten()

	A=LinearOperator((Ntf*Nx*Ny,Ntf*Nx*Ny),linOp)
	vals, vecs = eigs(A, k=Nvals,tol=1e-6,which='LR')
	for i in range(Nvals):
		vi=vecs[:,i].reshape(Ntf,Nx,Ny)
		Tsym=vi[Ntf*3/4,Nx*3/4-1,Ny*3/4-1]/vi[Ntf*1/4,Nx*3/4-1,Ny*3/4-1]
		Psym=vi[Ntf*3/4,Nx*3/4-1,Ny*3/4-1]/vi[Ntf*3/4,Nx*1/4-1,Ny*1/4-1]
		print Tsym
		print Psym
		print ('O' if Tsym.real<0 else 'E')+'S'+('O' if Psym.real<0 else 'E')+' lambda=%s'%vals[i]
		pl.figure(2*i+1)
		plotBZ(Nx*Ny*vi[Nc].real)
		# pl.figure(2*i+2)
		# plotBZ(Nx*Ny*vi[Nc-1].real)
	pl.show()
	return

	pl.ion()
	Q,h=arnoldi_iteration(lambda x:-Vsa*(G2**x),v,36)
	print h
	lam,w  =np.linalg.eig(h[:-1,:])
	print lam

	while True:
		print "Applying operator..."
		v2=-Vsa*(G2**v)+shift*v
		n=np.sqrt(v2.dot(v2))
		print "Largest eigenvalue estimate: %s"%(v.dot(v2)-shift)
		v=v2/n
		pl.figure(0)
		pl.cla()
		CS =pl.contour(kxs,kys,Nx*Ny*v.v[Nc].real,colors='k')
		pl.clabel(CS, fontsize=9, inline=1)
		# pl.figure(1)
		# pl.cla()
		# CS =pl.contour(kxs,kys,Nx*Ny*v.v[Nc-1].real,colors='k')
		# pl.clabel(CS, fontsize=9, inline=1)
		pl.pause(0.1)
	

	return


	Deltas=[]
	for i in range(4):
		Deltas.append(eigen(lambda D:Vsa*((G**G.reverse())**D)-sum(d[1]*D.dot(d[1]) for d in Deltas), G**G))
	print [d[0] for d in Deltas]


	import pylab as pl
	pl.figure()
	ws,kxs,kys=fermionicGrid(model)	
	X, Y = np.meshgrid(kxs, kys)
	CS = pl.contour(X, Y, [[eps(kx,ky,model[2])-mu for kx in kxs] for ky in kys], np.linspace(-3,3,13),colors='k',)
	pl.clabel(CS, fontsize=9, inline=1)


	g0=G0vec(model)
	chi0=Chi0(g0,model)
	
	print chi0[Nc+30,23,3]
	print singleFreeChi0(Nc+30,23,3,model)

	chis=chi0/(1-U*chi0)
	chic=chi0/(1+U*chi0)
	Vsa=U+3./2*U**2*chis-1./2*U**2*chic
	
	pl.figure()
	pl.plot(np.transpose(Vsa[0,0:-1:4].real))

	#pl.show()

if __name__ == "__main__":
	test()
	main()

