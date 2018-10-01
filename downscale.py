import numpy as np
from scipy.signal import convolve2d
from scipy.sparse.linalg import eigs, LinearOperator
from greensFunction import *

from lattice_pt import getLambdas
def gl(a):
	Nc=a[0]
	Nx=a[1]
	Ny=a[2]
	tx=a[3]
	ty=a[4]
	t2=a[5]
	filling=a[6]
	T=a[7]
	U=a[8]
	St=a[9]
	return getLambdas(Nc,Nx,Ny,tx,ty,t2,filling,T,linearized=False,U=U,St=St)

def main():
	nc=256
	nx=32
	ny=16
	nxSmall=32
	nySmall=16

	filling=.5
	tx=1
	T=.05
	U=4
	St=.97

	rs=np.linspace(0,1,5)

	from multiprocessing import Pool
	p=Pool(processes=len(rs)+1)
	# lambdas=p.map(gl,[(Nc,Nx,Ny,tx,x,filling,T,U,St) for x in xs])
	#lambdasOrig=p.map(gl,[(Nc,Nx,Ny,tx,0,filling,T,U,None,False) for T in Ts])
	#lambdasNodiag=p.map(gl,[(Nc,Nx,Ny,tx,0,filling,T,U,None,True) for T in Ts])
	lambdasSmall=p.map(gl,[(nc,nxSmall,nySmall,tx,tx*r,tx*r,filling,T,U,None) for r in rs])
	
	p.terminate()
	import matplotlib
	# matplotlib.use('Agg')
	from matplotlib import rc
	rc('text', usetex=True)
	import pylab as pl
	for lamb in enumerate([lambdaSmall]):
		for S in [0,1]:
			for Tsym in [0,1]:
				label='$\\mathrm{Full\\ '+('E' if (S-.5)*(.5-Tsym)<0 else 'O')+('T' if S==1 else 'S')+('O' if Tsym==1 else 'E')+'}$'
				pl.plot(rs,[np.real(l[S][Tsym][0]) for l in lamb],marker='.',label=label)
	pl.legend()
	pl.xlim([rs[0],rs[-1]])
	pl.ylim([0,1.5])
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

if __name__ == "__main__":
	main()

