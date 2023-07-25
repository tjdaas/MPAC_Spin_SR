"""
Calculating the MP AC along the \lambda path for different spin systems (w) and the subleading order terms of the large \lambda limit.
"""

import numpy as np
import os.path
import argparse
from sub_codes.run_HF import SCF_HF
from sub_codes.dmsuite import lagdif
from sub_codes.run_eps import epsilon_12,epsilon_14
from sub_codes.fin_spec_renorm import run_FSR
from sub_codes.export_files import output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--w", type=float, default=1., help="the amount of \beta-spin")
    parser.add_argument("--spinflip", type=bool, default=False, action=argparse.BooleanOptionalAction, help="allow for spinflip  i.e. q=/=w or q=w")
    parser.add_argument("--gridsize", type=float, default=100,help="the number of grid points")
    parser.add_argument("--gridfactor",type=int,default=30,help="grid contraction, higher values give more contracted grids")
    parser.add_argument("--lambdastart",type=float,default=0.,help="the starting value of \lambda")
    parser.add_argument("--lambdamax", type=float,default=20.,help="the maximum value of \lambda")
    parser.add_argument("--stepsize", type=float,default=0.05,help="the stepsize in \lambda")
    parser.add_argument("--nbasis", type=int, default=10,help="the number of basisfunctions used in the STO expansion of the HF orbital")
    parser.add_argument("--iter",type=int,default=1000, help="maximum number of iterations per \lambda for the FSR algorithm")
    parser.add_argument("--laplacianfactor",type=float,default=2.,help="constant added to avoid \eps+T to become negative, important for l>0 or \lambda<0")

##Input values
args = parser.parse_args()
w = args.w
s= 1 - 2*w + 2*w**2 #spin factor
spinflip = args.spinflip
lambdastart = args.lambdastart #for negative \lambda, you need to change ci
lambdamax = args.lambdamax #for negative \lambda, you need to change ci
stepsize = args.stepsize
nbasis = args.nbasis
N = args.gridsize
b = args.gridfactor
max_num_iterat = args.iter #max number of iterations
ci=args.laplacianfactor
x,D = lagdif(N+1,2,b)  #to build the grid and define the laplacian 
D2 = D[1,1:N+1:,1:N+1:] #defining the laplacian               
x = np.delete(x,0)
I = np.eye(len(D2)) #identity matrix for the laplacian 

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
