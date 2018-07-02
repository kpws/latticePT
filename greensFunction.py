import numpy as np
from scipy.signal import convolve2d

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
		ws=[greensFunction.getw(self.Nc,self.stat,self.T,i) for i in range(self.getNt())]
		kxs=[self.getk(Nx,i) for i in range(Nx)]
		kys=[self.getk(Ny,i) for i in range(Ny)]
		self.v=np.array([[[f(w,kx,ky) for ky in kys] for kx in kxs] for w in ws])
		return self

	@staticmethod
	def getw(Nc,stat,T,n):
		return (-2*Nc+(1-stat)//2+2*n)*np.pi*T

	# Nx,Ny are even, i=Nx/2-1 is k=0. Eg. Nx=4: k=pi*(-.5, 0, .5, 1)
	def getk(self,Nk,n):
		return -np.pi+2*np.pi*(n+1)/Nk

	def getNt(self):
		return 2*self.Nc-(1+self.stat)//2

	def getMinInHalfStep(self,stat):
		#number of half discrete energy steps the min frequency is below 0
		return 2*self.Nc-2 if stat==1 else 2*self.Nc-1

	def real(self):
		# return np.fft.fftn(np.sum(self.v,axis=0))/(np.pi*2)**2
		# print(np.sum(self.v,axis=0))
		# print(np.sum(np.roll(np.roll(self.v,-((self.Nx//2-1)),axis=1),-((self.Ny//2-1)),axis=2),axis=0))
		if self.stat==-1:
			return np.fft.fftn(np.sum(np.roll(np.roll(self.v,-((self.Nx//2-1)),axis=1),-((self.Ny//2-1)),axis=2),axis=0))

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

	def inv(self):
		#return greensFunction(self.stat,self.T,v=np.flip(np.flip(np.flip(self.v,0),1),2))
		return greensFunction(self.stat,self.T,v=np.reciprocal(self.v))

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