import numpy as np
from scipy.signal import convolve2d
from scipy.sparse.linalg import eigs, LinearOperator
from greensFunction import *

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import rc
rc('text', usetex=True)
import matplotlib.pyplot as pl
experimentName='1'

def savefig(n):
    pl.savefig('plots/'+experimentName + n +'.pdf')


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
	return getLambdas(Nc,Nx,Ny,tx,ty,t2,filling,T,linearized=False,U=U,St=St,returnU=not St==None)

def main():
	red=1
	nc=256//red
	nx=128//128//red
	ny=64//64//red
	nxSmall=64//red
	nySmall=4//red

	filling=.5
	tx=1
	T=.04
	U=4
	St=.97

	rs=np.linspace(0,1,40//red)

	from multiprocessing import Pool
	p=Pool(processes=min(4,len(rs)+1))
	# lambdas=p.map(gl,[(Nc,Nx,Ny,tx,x,filling,T,U,St) for x in xs])
	#lambdasOrig=p.map(gl,[(Nc,Nx,Ny,tx,0,filling,T,U,None,False) for T in Ts])
	#lambdasNodiag=p.map(gl,[(Nc,Nx,Ny,tx,0,filling,T,U,None,True) for T in Ts])
	
	import pickle
	cacheFile="cache/downsize3_red"+str(red)
	if False:
		allLambdas,allUs=pickle.load(open(cacheFile,'rb'))
	else:
		lambdas,Us=zip(*p.map(gl,[(nc,nx,ny,tx,tx*r,tx*r,filling,T,U,St) for r in rs]))
		lambdasSmall,UsSmall=zip(*p.map(gl,[(nc,nxSmall,nySmall,tx,tx*r,0,filling,T,U,St) for r in rs]))
		lambdasNoDiag,UsNoDiag=zip(*p.map(gl,[(nc,nx,ny,tx,tx*r,0,filling,T,U,St) for r in rs]))
		allLambdas=[lambdas,lambdasSmall,lambdasNoDiag]	
		allUs=[Us,UsSmall,UsNoDiag]	
		#pickle.dump((allLambdas,allUs),open(cacheFile,'wb'))

	print(allUs)

	p.terminate()
	for lambi,lamb in enumerate(allLambdas):
		pl.figure()
		for S in [0,1]:
			for Tsym in [0,1]:
				label='$\\mathrm{\\ '+('E' if (S-.5)*(.5-Tsym)<0 else 'O')+('T' if S==1 else 'S')+('O' if Tsym==1 else 'E')+'}$'
				pl.plot(rs,[np.real(l[S][Tsym][0]) for l in lamb],marker='.',label=label)
		pl.plot([],[],linestyle='--',color='black',label='$U$')
		pl.ylabel('$\lambda$')
		pl.xlabel('$t_y/t_x$')
		pl.grid()
		pl.legend()
		pl.gca().twinx()
		pl.plot(rs,allUs[lambi],linestyle='--',color='black',label='$U$')
		pl.ylim([0,max(allUs[lambi])])
		pl.xlim([rs[0],rs[-1]])
		#pl.ylim([0,1.5])
		# pl.xlabel('$t_y/t_x=t_2/t_x$')
		pl.ylabel('$U$')
		if lambi==0:	
			title='N_\\tau={0},\\ N_x={1},\\ N_y={2},\\ T={3}t_x,\\ S_t={4},\\ t_2=t_x'.format(nc,nx,ny,T,St)
		if lambi==1:	
			title='N_\\tau={0},\\ N_x={1},\\ N_y={2},\\ T={3}t_x,\\ S_t={4},\\ t_2=0'.format(nc,nxSmall,nySmall,T,St)
		if lambi==2:	
			title='N_\\tau={0},\\ N_x={1},\\ N_y={2},\\ T={3}t_x,\\ S_t={4},\\ t_2=0'.format(nc,nx,ny,T,St)
		pl.title('$'+title+'$')
		savefig('downsize_'+title.replace('\\',''))

if __name__ == "__main__":
	main()

