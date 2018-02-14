import numpy as np
from scipy.signal import convolve2d

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
		return -(2*self.Nc-(1-self.stat)/2-2*n)*np.pi*self.T

	@staticmethod
	def getw(Nc,stat,T,n):
		return -(2*Nc-(1-stat)/2-2*n)*np.pi*T

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
			v=np.roll(np.roll(res[trim1:len(res)-trim2],-((self.Nx-1)/2),axis=1),-((self.Ny-1)/2),axis=2)
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
		return convolve2d(a,b,boundary='wrap',mode='same')
		#return a*b

	def dot(self,g2):
		assert self.compatible(g2)
		return np.tensordot(self.v,np.conjugate(g2.v),axes=([0,1,2],[0,1,2]))

	def reverse(self):
		#return greensFunction(self.stat,self.T,v=np.flip(np.flip(np.flip(self.v,0),1),2))
		return greensFunction(self.stat,self.T,v=self.v[::-1,::-1,::-1])

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
			return greensFunction(self.stat,self.T,v=g2+self.v)
	
	__radd__=__add__

	def __sub__(self,g2):
		if isinstance(g2,greensFunction):
			assert self.compatible(g2)
			return greensFunction(self.stat,self.T,v=self.v-g2.v)
		else:
			return greensFunction(self.stat,self.T,v=g2-self.v)

	def __rsub__(self,g2):
		if isinstance(g2,greensFunction):
			assert self.compatible(g2)
			return greensFunction(self.stat,self.T,v=g2.v-self.v)
		else:
			return greensFunction(self.stat,self.T,v=self.v-g2)

	def __str__(self):
		return str(('bosonic' if self.stat==1 else 'fermionic',self.T,self.v))

	def compatible(self,g2):
		return self.Nc==g2.Nc and self.T==g2.T and self.stat==g2.stat and self.Nx==g2.Nx and self.Ny==g2.Ny

def eps(kx,ky,couplings):
	tx,ty,t2,U=couplings
	return -2*tx*np.cos(kx)-2*ty*np.cos(ky)-2*t2*np.cos(kx+ky)

def test():
	print convolve2d([[1,2,3,4]],[[1,2,3,4]],boundary='wrap',mode='same')
	exit(0)

	f1=greensFunction(-1,1.,v=[[[1,2],[3,4]],[[5,6],[7,8]]])
	f2=greensFunction(-1,1.,v=[[[9,10],[11,12]],[[5,6+3j],[7,8]]])
	b1=greensFunction(1,1.,v=[[ [1,2],[3,4] ]])
	b2=greensFunction(1,1.,v=[[ [1,2],[7,4] ]])

	Nx=12
	Ny=4
	Ntb=5
	Ntf=Ntb+1
	f1=greensFunction(-1,1.,v=np.random.rand(Ntf,Nx,Ny))
	f2=greensFunction(-1,1.,v=np.random.rand(Ntf,Nx,Ny))
	b1=greensFunction(1,1.,v=np.random.rand(Ntb,Nx,Ny))
	b2=greensFunction(1,1.,v=np.random.rand(Ntb,Nx,Ny))
	for (g1,g2,name) in [(f1,f2,'ff'), (b1,b2,'bb'), (f1,b1,'fb'), (b2,f1,'bf')]:
		print name+':'
		a1=g1.__mul__(g2,fft=False)
		a2=g2.__mul__(g1,fft=False)
		assert (a1-a2).dot((a1-a2))<1e-15; print ' * passed reverse'
		a3=g1.__mul__(g2,fft=True)
		assert (a1-a3).dot((a1-a3))<1e-15; print ' * passed fft'
		a4=g2.__mul__(g1,fft=True)
		assert (a1-a4).dot((a1-a4))<1e-15; print ' * passed fft reverse'

	print (f1*f2).v[(Ntb-1)/2,Nx/2,Ny/2]*f1.Nx*f2.Ny/f1.T
	print f1.dot(f2.reverse())
	print abs((f1*f2).v[(Ntb-1)/2,Nx/2,Ny/2]-f1.dot(f2.reverse()))
	assert abs((f1*f2).v[(Ntb-1)/2,Nx/2,Ny/2]*f1.Nx*f2.Ny/f1.T-f1.dot(f2.reverse()))<1e-10; print ' * passed dot-reverse/conv comparison'

def main():
	# Nc=2048
	# Nx=128
	# Ny=64
	Nc=1024
	Nx=128
	Ny=64

	mu=0
	tx=1
	ty=.1*tx
	t2=.1*tx
	T=0.04*tx
	St=.97
	
	print "Generating G..."
	kxs=np.array([[-np.pi+(i+1)*2*np.pi/Nx for i in range(Nx)],]*Ny).transpose()
	kys=np.array([[-np.pi+(i+1)*2*np.pi/Ny for i in range(Ny)],]*Nx)
	eps=-2*tx*np.cos(kxs)-2*ty*np.cos(kys)-2*t2*np.cos(kxs+kys)-mu
	#G=greensFunction(-1,T).fromFun(Nc,Nx,Ny,lambda w,kx,ky:1/(1j*w-eps(kx,ky,model[2])+mu))

	# import pylab as pl
	# pl.figure()
	# CS =pl.contour(eps,np.linspace(-3,3,13),colors='k')
	# pl.clabel(CS, fontsize=9, inline=1)
	# pl.show()




	G=greensFunction(-1,T,v=[1/(1j*greensFunction.getw(Nc,-1,T,i)-eps) for i in range(2*Nc)])
	print "Calculating Chi..."
	chi0=-G*(G.reverse())
	Chi0max=np.sum(chi0.v,(1,2)).max()
	Chi0max=chi0.v.max()
	print "Chi0 max: "+str(Chi0max)
	U=St/Chi0max.real
	print " => U= "+str(U/tx)+' tx'

	chis=chi0/(1-U*chi0)
	chic=chi0/(1+U*chi0)
	Vsa=U+3./2*(U**2)*chis-1./2*(U**2)*chic
	Vta=-1./2*(U**2)*chis-1./2*(U**2)*chic

	print "Finding linearized gap equation solutions..."
	v=greensFunction(-1,T,v=np.random.rand(2*Nc,Nx,Ny))
	shift=3
	import pylab as pl
	pl.ion()

	while True:
		v2=Vsa*((G**G.reverse())**v)+shift*v
		n=np.sqrt(v2.dot(v2))
		print v.dot(v2)-shift
		v=v2/n
		pl.cla()
		CS =pl.contour(Nx*Ny*v.v[Nc].real,colors='k')
		pl.clabel(CS, fontsize=9, inline=1)
		pl.pause(0.5)
	

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

