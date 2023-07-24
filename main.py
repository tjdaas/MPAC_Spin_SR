"""
Calculating the MP AC along the \lambda path for different spin systems (w) and the subleading order terms of the large \lambda limit.
"""

import numpy as np
import os.path
from sub_codes.run_HF import SCF_HF
from sub_codes.dmsuite import lagdif
from sub_codes.run_eps import epsilon_12,epsilon_14
from sub_codes.fin_spec_renorm import run_FSR
from sub_codes.export_files import output

##Input values
w = 1/2 #the amount of alpha-spin
s= 1 - 2*w + 2*w**2 #spin factor
spinflip=False # True or false to allow spinflip i.e. q=/=w or q=w
lambdastart=0 #the value of \lambda you want to start at (for negative \lambda, you need to change ci)
lambdamax=20 #the value of \lambda you want to end at (for negative \lambda, you need to change ci)
stepsize=0.05 #the steps between each lambda
nbasis = 10 #Number of basis states used in the STO ansatz
N=100 #number of grid points
b=40 #grid contraction, higher b's give more contracted grids
max_num_iterat=1000 #max number of iterations
ci=2 #add this when the denominator becomes negative, important for l>0 or \lambda<0
x,D = lagdif(N+1,2,b)  #to build the grid and define the laplacian 
D2 = D[1,1:N+1:,1:N+1:] #defining the laplacian               
x = np.delete(x,0)
I = np.eye(len(D2)) #identity matrix for the laplacian 
print(I.shape)
print(D2.shape)
print(x.shape)
stop
#HF code to find the HF orbital phi_zero that we need in our J and K
normvec,Cmat1=SCF_HF(s,nbasis) 

#Run loops over l=0,l=1 and l=2 using the spectral renormalization algorithm
FSR=run_FSR(N,x,D2,I,ci,nbasis,normvec,Cmat1)
epslistfull,data_array_l0,data_array_l1,data_array_l2,der_datal0,der_data,u0=FSR.run_loop(lambdastart,lambdamax,stepsize,spinflip,w,s,max_num_iterat)

#calculating the crossing of l=0 and l=1
try:
    cross_1,cross_2=FSR.cross()
except Exception:
    cross_1=cross_2=None

#calculating epsilon_{1/2}
epsilon12,u12=epsilon_12(x,s,D2,N,I)

#calculating epsilon_{1/4}
epsilon14=(epsilon_14(x,s,u12)) 

#making new directory to output data
old_pwd=os.getcwd()
datadir=old_pwd+"/data"
try:
    os.mkdir(datadir)
except Exception:
    pass
os.chdir(datadir)

#making figures
exp=output(epslistfull,data_array_l0,data_array_l1,data_array_l2,der_datal0,der_data,cross_1,cross_2,epsilon12,epsilon14)
exp.print_figs(x,u0,s,lambdamax)
#printimg output
exp.export_files(spinflip,w)
