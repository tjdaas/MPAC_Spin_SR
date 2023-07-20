import numpy as np
from scipy.integrate import cumtrapz,trapz
import matplotlib.pyplot as plt
from sub_codes.run_HF import *
from sub_codes.dmsuite import *
import os.path

##Input values
w = 1 #the amount of alpha-spin
spinflip=False # True or false to allow spinflip i.e. q=/=w or q=w
lambdastart=0 #the value of \lambda you want to start at (for negative \lambda, you need to change ci)
lambdamax=20 #the value of \lambda you want to end at (for negative \lambda, you need to change ci)
stepsize=0.05 #the steps between each lambda
#HF code to find the HF orbital phi_zero that we need in our J and K
Nel = 1 #Number of electrons
s= 1 - 2*w + 2*w**2 #spin factor
Z = 1 #Nuclear charge
n = 0 #main quantum number
nbasis = 10 #Number of basis states
basisn = range(1,nbasis+1) #n for the nbasis states
basisl = [0]*nbasis #Angular momentum l for the nbasis states
basism = [0]*nbasis #Angular momentum m_l for the nbasis states
basisa = [1.]*nbasis #Exponents a for the nbasis states
normvec = np.zeros(nbasis) #Normalization vector
iters = 50  #Number of self-consistent iterations
for i in range(nbasis):
    normvec[i] = STOnorm(basisn[i],basisa[i]) #Fill normalization vector
F0mat,Smat = buildF0Smat(nbasis,basisn,basisl,basism,basisa,Z,normvec) #Build core fock matrix F_0 (kinetic + external potential) and overlap matrix
coulombmat = buildcoulombmat(nbasis,basisn,basisl,basism,basisa,s,normvec) #Build coulomb matrix
# Core Hamiltonian only starting values, eigh is the generalized eigensolver
eigenvals, Cmat = eigh(F0mat,Smat)
energies = np.zeros(iters) #Empty list for energies during self-consistent cycle
for iteration in range(iters):
    vhxmat = buildvhmat(nbasis,Cmat,coulombmat) #Build new hartree-exchange matrix
    Fmat = F0mat + 0.5*vhxmat #Build fock matrix from core fock matrix and hartree-exchange matrix
    eigenvals, Cmat = eigh(Fmat, Smat) #Solve generalized eigenvalue problem
    energies[iteration] = calculate_energy(eigenvals, Cmat, vhxmat,nbasis) #Calculate energy corresponding to the state
Cmat1=Cmat[:,0]
#defining the grid and other input data for the spectral renormalization/finite-difference renormalization code
N=100 #number of grid points
b=40; #grid contraction, higher b's give more contracted grids
max_num_iterat=1000 #max number of iterations
ci=2; #add this when the denominator becomes negative, important for small l>0 or \lambda<0 
x,D = lagdif(N+1,2,b)  # used to define the laplacian 
D2 = D[1,1:N+1:,1:N+1:] #defining the laplacian               
x = np.delete(x,0)
I = np.eye(len(D2)) #identity matrix for the laplacian 

#Using the HF orbital (phi_zero) from the premilinary HF calculation and calculating v_H
phi_zero=np.zeros(len(x))
for kj in range(nbasis):
    phi_zero+=normvec[kj]*Cmat1[kj]*x**(kj)*np.exp(-x)/np.sqrt(4*np.pi)
Ne=cumtrapz((x**2)*phi_zero**2,x,initial=0)
integrand_inf_0= x**(1)*phi_zero**2
I_int_num_inf_0=trapz(integrand_inf_0,x) - cumtrapz(integrand_inf_0,x,initial=0)
V_H=4*np.pi*((Ne/x)+I_int_num_inf_0)
Coulomb = -1/x
#The finite difference renormalization algorithm that allows us to calculate the psi_lambda, the E_lambda for all different spin systems 
data_array_l0=np.zeros((int(1+(lambdamax-lambdastart)/stepsize),2))  #array to store l=0 E_lambda results
data_array_l1=np.zeros((int(1+(lambdamax-lambdastart)/stepsize),2))  #array to store l=1 E_lambda results
data_array_l2=np.zeros((int(1+(lambdamax-lambdastart)/stepsize),2))  #array to store l=2 E_lambda results
array_list = [data_array_l0, data_array_l1, data_array_l2]
for li in range(3): #first loop over l=0, l=1, l=2
    l=li
    V_eff=l*(l+1)/(2*x**2);  #Angular kinetic energy for a certain l-channel
    linspace=np.linspace(lambdastart,lambdamax,int(1+(lambdamax-lambdastart)/stepsize)) #setting up which lambda will we calculate
    data_array=np.zeros((int(1+(lambdamax-lambdastart)/stepsize),2))  #E_lambda is stored here
    der_data=np.zeros((int(1+(lambdamax-lambdastart)/stepsize),2))    #W_lambda is stored here
    der_datal0=np.zeros((int(1+(lambdamax-lambdastart)/stepsize),2))  #W_lambda for l=0 is stored here
    #Loop over all \lambda's
    for ik in range(len(linspace)):
        lb=linspace[ik]
        u= x**(1)*np.exp(-x); #the initial guess, which can be random numbers
        coef_ini=cumtrapz(np.absolute(u)**2,x,initial=0); #normalizing our wavefunctions
        u=u/np.sqrt(coef_ini[N-1]); 
        if spinflip == False: #checking if spinflip is allowed and going through the conditions on q and \lambda
            q = w
        elif lb < 1 and w < 0.5:
            q = 0
        elif lb < 1 and w >= 0.5:
            q = 1
        elif lb >= 1 and w >= 0.5:
            q = 0
        else:
            q = 1
        #now we start our iterations 
        s_qw = (1-q)*(1-w)+q*w #the full spin factor
        for its in range(0,max_num_iterat):
            #K integrals worked out
            integrand_0   = x**(l+1)*phi_zero*u;
            integrand_inf = x**(-l)*phi_zero*u;
            I_int_num_0 = cumtrapz(integrand_0,x,initial=0);
            I_int_num_inf = trapz(integrand_inf,x) - cumtrapz(integrand_inf,x,initial=0);
            Total_pot = (4*np.pi*s_qw/(2*l+1))*(1-lb)*phi_zero*((x**(-l))*I_int_num_0 + (x**(l+1))*I_int_num_inf);

            #Compute T, J+V_ext+V_eff and K
            u_double_prime=np.matmul(D2,u);
            KE=0.5*cumtrapz(u*u_double_prime,x,initial=0);
            Pot_energy=cumtrapz(np.absolute(u)**2*(Coulomb+V_eff+(1-lb)*V_H),x,initial=0);
            Pot_energy_2=cumtrapz(u*Total_pot,x,initial=0);

            #calc normalization constant and epsilon_n
            denom=cumtrapz(np.absolute(u)**2,x,initial=0);
            epsilon=(-KE[N-1]+Pot_energy[N-1] - Pot_energy_2[N-1])/denom[N-1];

            # calc psi_(n+1) by solving the matrix equation
            RHS=u*(Coulomb+(1-lb)*V_H-ci) -  Total_pot ;
            M_mat=-ci*I+0.5*D2+epsilon*I-np.diag(V_eff);
            u_new=np.linalg.solve(M_mat,RHS);
            coef_new=cumtrapz(np.absolute(u_new)**2,x,initial=0); #renormalize wavefunction                                   
            u_new=u_new/np.sqrt(coef_new[N-1]); #update wavefunction
            if np.max(np.max(np.absolute(u_new-u)))<=10**(-9): #dont go too low especially for l=0 and s_wq=1
                break 
            u=u_new; #update wavefunctions
        data_array[ik,0]=lb; #first column will be lambda
        data_array[ik,1]=epsilon; #second column will be epsilon
        if l==0 and lb==0:
            u0_2=u #saving the wavefunction for l=0 at \lambda=0
    if l==0:
        u0=u #saving the wavefunction for l=0 at \lambda=lambdamax
    Uh=2*np.pi*trapz(x**2*phi_zero**2*V_H,x) #calculating the Hartree Energy (U_h)
    array_list[l]=data_array 
    data_array_l0=array_list[0] #Store the l=0 E_lambda data
    data_array_l1=array_list[1] #Store the l=1 E_lambda data
    data_array_l2=array_list[2] #Store the l=2 E_lambda data
gradl0=np.gradient(data_array_l0[:,1],data_array_l0[:,0]) #calc the derivative of E_\lambda for l=0
for ji in range(len(linspace)):
        der_datal0[ji,0]=data_array_l0[ji,0]; #first column will be lambda
        der_datal0[ji,1]=gradl0[ji]+Uh*(1-s); #second column will be W_\lambda for l=0

epslist=np.minimum(np.minimum(data_array_l0[:,1],data_array_l1[:,1]),data_array_l2[:,1]) #find the lowest E_\lambda of l=0, l=1 and l=2
epslistfull=np.zeros((int(1+(lambdamax-lambdastart)/stepsize),2))
for ils in range(len(epslist)):
    epslistfull[ils,0]=data_array_l0[ils,0]
    epslistfull[ils,1]=epslist[ils]

grad=np.gradient(epslistfull[:,1],epslistfull[:,0]) #calculating the derivative of E_\lambda
for ji in range(len(linspace)):
        der_data[ji,0]=epslistfull[ji,0]; #first column will be \lambda
        der_data[ji,1]=grad[ji]+Uh*(1-s); #second column will be W_\lambda for l=0

#find the point where the crossing happens between the two l-channels
res = []
test_list=data_array_l0[:,1]-data_array_l1[:,1]
for idx in range(0, len(data_array_l0) - 1):
    # checking for successive opposite index
    if test_list[idx] > 0 and test_list[idx + 1] < 0 or test_list[idx] < 0 and test_list[idx + 1] > 0:
        res.append(idx)
cross_1=data_array_l0[res[0],0]
cross_2=data_array_l0[res[1],0]

## Solving the l dependent eigenvalue of epsilon_12 for l=0.
c=30; #constant in case the laplacian becomes positive
l=0; # l=0 gives the lowest value in the strong coupling limit
V_eff=l*(l+1)/(2*x**2); #the radial kinetic energy
max_num_iterat=1000;  #number of iterations
u= x**(l+1)*np.exp(-x); #defining the initial guess
coef_ini=cumtrapz(np.absolute(u)**2,x,initial=0);                                 
u=u/np.sqrt(coef_ini[N-1]); #normalising the initial guess
#now we start our iterations 
for its in range(0,max_num_iterat):
    
    #K integrals worked out
    integrand_0   = x**(l+1)*u;
    integrand_inf = x**(-l)*u;
    I_int_num_0 = cumtrapz(integrand_0,x,initial=0);
    I_int_num_inf = trapz(integrand_inf,x) - cumtrapz(integrand_inf,x,initial=0);
    Total_pot = (s/(2*l+1))*((x**(-l))*I_int_num_0 + (x**(l+1))*I_int_num_inf);
    
    #Compute T, J+V_ext+V_eff and K
    u_double_prime=np.matmul(D2,u);
    KE=0.5*cumtrapz(u*u_double_prime,x,initial=0);
    Pot_energy=cumtrapz(np.absolute(u)**2*(V_eff+(x**(2))/6),x,initial=0);
    Pot_energy_2=cumtrapz(u*Total_pot,x,initial=0);
    
    #calculate normalization constant and epsilon_{1/2}
    denom=cumtrapz(np.absolute(u)**2,x,initial=0);
    epsilon12=(-KE[N-1]+Pot_energy[N-1] + Pot_energy_2[N-1])/denom[N-1];
    
    # calculate psi_(n+1) by solving the matrix equation
    RHS=u*(-c+(x**(2))/6) +  Total_pot ;
    M_mat=-c*I+0.5*D2+epsilon12*I-np.diag(V_eff);
    u_new=np.linalg.solve(M_mat,RHS);
    u=u_new; #update wavefunctions  
    coef_new=cumtrapz(np.absolute(u)**2,x,initial=0);                                   
    u=u/np.sqrt(coef_new[N-1]); #normalisation of the new wavefunction

## calculating the epsilon_14 using the 2n+1 trick.
Pot_energy=((1/x)+(x**3)/6)*np.absolute(u)**2 #Compute J
#K integrals worked out
integrand_0_1   = x**(1)*u; 
integrand_0_2   = x**(2)*u;
I_int_num_0_1 = cumtrapz(integrand_0_1,x,initial=0); 
I_int_num_0_2 = cumtrapz(integrand_0_2,x,initial=0);
Total_pot = ((x**(1))*I_int_num_0_1 + I_int_num_0_2); 
Pot_energy_2=u*Total_pot; #Compute K
epsilon14=-trapz(Pot_energy+(2*s)*Pot_energy_2 ,x) #compute \epsilon_{1/4}

# Plotting all the graphs
old_pwd=os.getcwd()
datadir=old_pwd+"/data"
try:
    os.mkdir(datadir)
except Exception:
    pass
os.chdir(datadir)
l0=plt.scatter(data_array_l0[:,0],data_array_l0[:,1]) #defining the E_\lambda curve for l=0
l1=plt.scatter(data_array_l1[:,0],data_array_l1[:,1]) #defining the E_\lambda curve for l=1
l2=plt.scatter(data_array_l2[:,0],data_array_l2[:,1]) #defining the E_\lambda curve for l=2
plt.legend((l0, l1, l2),('l=0', 'l=1', 'l=2'))
plt.title('s=%d' %s)
plt.ylabel(r'$E_\lambda$')
plt.xlabel(r'$\lambda$')
plt.savefig('Elambda.png')#plotting the E_\lambda curves for l=0, l=1, l=2
plt.scatter(x,u0/x)
plt.ylabel(r'$\psi_{\lambda=%d}$'%lambdamax)
plt.xlabel(r'$x$')
plt.title('l=0, s=%d' %s)
plt.savefig('psilambda.png') #plotting \psi_\lambda at \lambda=lambdamax for l=0
plt.scatter(x,u0)
plt.ylabel(r'$u_{\lambda=%d}$'%lambdamax)
plt.xlabel(r'$x$')
plt.title('l=0, s=%d' %s)
plt.savefig('ulambda.png')#plotting u_\lambda at \lambda=lambdamax for l=0
plt.scatter(der_datal0[:,0],der_datal0[:,1])
plt.ylabel(r'$W_\lambda$')
plt.xlabel(r'$\lambda$')
plt.title('l=0, s=%d' %s)
plt.savefig('Wlambda_l0.png') #plotting the W_\lambda curves for l=min (so channel that gives the lowest energy)
plt.scatter(der_data[:,0],der_data[:,1])
plt.ylabel(r'$W_\lambda$')
plt.xlabel(r'$\lambda$')
plt.title('s=%d' %s)
plt.savefig('Wlambda_lmin.png') #plotting the E_\lambda curves for l=0
#Export files and data to .csv files
if spinflip==False:
    Elamfile="E_lambda_s"+str(w)+".csv" #contains the full E_lambda curve
    Wlamfile="W_lambda_s"+str(w)+".csv" #contains the full W_lambda curve
    Elaml0file="E_lambda_l0_s"+str(w)+".csv" #contains only the E_lambda curve of l=0
    Wlaml0file="W_lambda_l0_s"+str(w)+".csv" #contains only the W_lambda curve of l=0
    Elaml1file="E_lambda_l1_s"+str(w)+".csv" #contains only the E_lambda curve of l=1
    Elaml2file="E_lambda_l2_s"+str(w)+".csv" #contains only the E_lambda curve of l=2
    Crossfile="Cross_s"+str(w)+".csv"  #contains the crossing positions
    Epsfile="Eps_s"+str(w)+".csv" #contains the values of epsilon_12 and epsilon_14
else:
    Elamfile="E_lambda_w"+str(w)+"sf"+".csv" #contains the full E_lambda curve
    Wlamfile="W_lambda_w"+str(w)+"sf"+".csv" #contains the full W_lambda curve
    Elaml0file="E_lambda_l0_w"+str(w)+"sf"+".csv" #contains only the E_lambda curve of l=0
    Wlaml0file="W_lambda_l0_w"+str(w)+"sf"+".csv" #contains only the W_lambda curve of l=0
    Elaml1file="E_lambda_l1_w"+str(w)+"sf"+".csv" #contains only the E_lambda curve of l=1
    Elaml2file="E_lambda_l2_w"+str(w)+"sf"+".csv" #contains only the E_lambda curve of l=2
    Crossfile="Cross_w"+str(w)+"sf"+".csv"  #contains the crossing positions
    Epsfile="Eps_w"+str(w)+"sf"+".csv" #contains the values of epsilon_12 and epsilon_14
np.savetxt(Elamfile,epslistfull, delimiter=",",fmt="%10.5f")
np.savetxt(Wlamfile,der_data, delimiter=",",fmt="%10.5f")
np.savetxt(Elaml0file,data_array_l0, delimiter=",",fmt="%10.5f")
np.savetxt(Wlaml0file,der_datal0, delimiter=",",fmt="%10.5f")
np.savetxt(Elaml1file,data_array_l1, delimiter=",",fmt="%10.5f")
np.savetxt(Elaml2file,data_array_l2, delimiter=",",fmt="%10.5f")
np.savetxt(Crossfile,[cross_1,cross_2], delimiter=",",fmt="%10.5f")
np.savetxt(Epsfile,[epsilon12,epsilon14], delimiter=",",fmt="%10.5f")