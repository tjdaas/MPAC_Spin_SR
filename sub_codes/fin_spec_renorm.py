from scipy.integrate import cumtrapz,trapz
from sub_codes.dmsuite import *
import numpy as np

#find the point where the crossing happens between the two l-channels
class run_FSR():
    def __init__(self,N,x,D2,I,ci,nbasis,normvec,Cmat1):
        """Class that runs the Finite difference Spectral Renormalisation Algorithm to solve the MP AC along the \lambda curve of l=0 to l=2.

        Args:
            N (integer): The number of grid points
            x (ndarray): 1d array containing the grid
            D2 (ndarray): 2d array containing the laplacian
            I (ndarray): 2d array containg the identity matrix
            ci (integer): constant that is added when \epsilon-T becomes positive
            nbasis (integer): the number of STO basis functions used
            normvec (ndarray): the vector containing the normalisation constant of the STO basisfunctions
            Cmat1 (ndarray): the vector containing the expansion coefficients of HF orbital
        """
        self.N=N
        self.x=x
        self.D2=D2
        self.I=I
        self.ci=ci
        self.nbasis=nbasis
        self.normvec=normvec
        self.Cmat1=Cmat1

    def build_HF(self):
        """Builds the HF orbital and the Hartree potential used in the loop.

        Returns:
            ndarray,ndarray: 1d arrays containing the HF orbital and the Hartree potential.
        """
        self.phi_zero=np.zeros(len(self.x))
        for kj in range(self.nbasis): #builds the HF orbital
            self.phi_zero+=self.normvec[kj]*self.Cmat1[kj]*self.x**(kj)*np.exp(-self.x)/np.sqrt(4*np.pi)
        Ne=cumtrapz((self.x**2)*self.phi_zero**2,self.x,initial=0) #cumulant
        integrand_inf_0= self.x**(1)*self.phi_zero**2
        I_int_num_inf_0=trapz(integrand_inf_0,self.x) - cumtrapz(integrand_inf_0,self.x,initial=0)
        self.V_H=4*np.pi*((Ne/self.x)+I_int_num_inf_0) #Hartree potential
        return self.phi_zero,self.V_H

    def run_loop(self,lambdastart,lambdamax,stepsize,spinflip,w,s,max_num_iterat):
        """calculates the E_\lambda curves for l=0, l=1 and l=2. 

        Args:
            lambdastart (float): the starting value of \lambda
            lambdamax (float): the maximum value of \lambda
            stepsize (float): the steps in \lambda
            spinflip (boolean): allow spinflip
            w (float): the amount of \beta spin
            s (float)): the spinfactor used in the HF orbital
            max_num_iterat (integer): the max amount of iterations

        Returns:
            ndarray,ndarray,ndarray,ndarray,ndarray,ndarray: 1d arrays containing the minimal E_\lambda for every channel, the E_\lambda for l=0, E_\lambda for l=1, E_\lambda l=2, the W_\lambda l=0, W_\lambda for for the lowest channel and the minimizing wavefunction at \lambdamax.
        """
        self.phi_zero,self.V_H=self.build_HF()
        self.data_array_l0=np.zeros((int(1+(lambdamax-lambdastart)/stepsize),2))  #array to store l=0 E_lambda results
        self.data_array_l1=np.zeros((int(1+(lambdamax-lambdastart)/stepsize),2))  #array to store l=1 E_lambda results
        data_array_l2=np.zeros((int(1+(lambdamax-lambdastart)/stepsize),2))  #array to store l=2 E_lambda results
        array_list = [self.data_array_l0, self.data_array_l1, data_array_l2]
        for li in range(3): #first loop over l=0, l=1, l=2
            l=li
            V_eff=l*(l+1)/(2*self.x**2)  #Angular kinetic energy for a certain l-channel
            Coulomb = -1/self.x #external potential
            linspace=np.linspace(lambdastart,lambdamax,int(1+(lambdamax-lambdastart)/stepsize)) #setting up which lambda will we calculate
            data_array=np.zeros((int(1+(lambdamax-lambdastart)/stepsize),2))  #E_lambda is stored here
            der_data=np.zeros((int(1+(lambdamax-lambdastart)/stepsize),2))    #W_lambda is stored here
            der_datal0=np.zeros((int(1+(lambdamax-lambdastart)/stepsize),2))  #W_lambda for l=0 is stored here
            #Loop over all \lambda's
            for ik in range(len(linspace)):
                lb=linspace[ik]
                u= self.x**(1)*np.exp(-self.x) #the initial guess, which can be random numbers
                coef_ini=cumtrapz(np.absolute(u)**2,self.x,initial=0) #normalizing our wavefunctions
                u=u/np.sqrt(coef_ini[self.N-1]) 
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
                    integrand_0   = self.x**(l+1)*self.phi_zero*u
                    integrand_inf = self.x**(-l)*self.phi_zero*u
                    I_int_num_0 = cumtrapz(integrand_0,self.x,initial=0)
                    I_int_num_inf = trapz(integrand_inf,self.x) - cumtrapz(integrand_inf,self.x,initial=0)
                    Total_pot = (4*np.pi*s_qw/(2*l+1))*(1-lb)*self.phi_zero*((self.x**(-l))*I_int_num_0 + (self.x**(l+1))*I_int_num_inf)

                    #Compute T, J+V_ext+V_eff and K
                    u_double_prime=np.matmul(self.D2,u)
                    KE=0.5*cumtrapz(u*u_double_prime,self.x,initial=0)
                    Pot_energy=cumtrapz(np.absolute(u)**2*(Coulomb+V_eff+(1-lb)*self.V_H),self.x,initial=0)
                    Pot_energy_2=cumtrapz(u*Total_pot,self.x,initial=0)

                    #calc normalization constant and epsilon_n
                    denom=cumtrapz(np.absolute(u)**2,self.x,initial=0)
                    epsilon=(-KE[self.N-1]+Pot_energy[self.N-1] - Pot_energy_2[self.N-1])/denom[self.N-1]

                    # calc psi_(n+1) by solving the matrix equation
                    RHS=u*(Coulomb+(1-lb)*self.V_H-self.ci) -  Total_pot 
                    M_mat=-self.ci*self.I+0.5*self.D2+epsilon*self.I-np.diag(V_eff)
                    u_new=np.linalg.solve(M_mat,RHS)
                    coef_new=cumtrapz(np.absolute(u_new)**2,self.x,initial=0) #renormalize wavefunction                                   
                    u_new=u_new/np.sqrt(coef_new[self.N-1]) #update wavefunction
                    if np.max(np.max(np.absolute(u_new-u)))<=10**(-9): #dont go too low especially for l=0 and s_wq=1
                        break 
                    u=u_new #update wavefunctions
                data_array[ik,0]=lb #first column will be lambda
                data_array[ik,1]=epsilon #second column will be epsilon
                if l==0 and lb==0:
                    u0_2=u #saving the wavefunction for l=0 at \lambda=0
            if l==0:
                u0=u #saving the wavefunction for l=0 at \lambda=lambdamax
            Uh=2*np.pi*trapz(self.x**2*self.phi_zero**2*self.V_H,self.x) #calculating the Hartree Energy (U_h)
            array_list[l]=data_array 
            self.data_array_l0=array_list[0] #Store the l=0 E_lambda data
            self.data_array_l1=array_list[1] #Store the l=1 E_lambda data
            data_array_l2=array_list[2] #Store the l=2 E_lambda data
        gradl0=np.gradient(self.data_array_l0[:,1],self.data_array_l0[:,0]) #calc the derivative of E_\lambda for l=0
        for ji in range(len(linspace)):
                der_datal0[ji,0]=self.data_array_l0[ji,0] #first column will be lambda
                der_datal0[ji,1]=gradl0[ji]+Uh*(1-s) #second column will be W_\lambda for l=0

        epslist=np.minimum(np.minimum(self.data_array_l0[:,1],self.data_array_l1[:,1]),data_array_l2[:,1]) #find the lowest E_\lambda of l=0, l=1 and l=2
        epslistfull=np.zeros((int(1+(lambdamax-lambdastart)/stepsize),2))
        for ils in range(len(epslist)):
            epslistfull[ils,0]=self.data_array_l0[ils,0]
            epslistfull[ils,1]=epslist[ils]

        grad=np.gradient(epslistfull[:,1],epslistfull[:,0]) #calculating the derivative of E_\lambda
        for ji in range(len(linspace)):
                der_data[ji,0]=epslistfull[ji,0] #first column will be \lambda
                der_data[ji,1]=grad[ji]+Uh*(1-s) #second column will be W_\lambda for l=0
        return epslistfull,self.data_array_l0,self.data_array_l1,data_array_l2,der_datal0,der_data,u0


    def cross(self):
        """calculates the crossings of states between the l=0 and l=1 channel

        Returns:
            float, float: the \lambda value of the first and second crossing
        """
        res = []
        test_list=self.data_array_l0[:,1]-self.data_array_l1[:,1]
        for idx in range(0, len(self.data_array_l0) - 1):
            # checking for successive opposite index
            if test_list[idx] > 0 and test_list[idx + 1] < 0 or test_list[idx] < 0 and test_list[idx + 1] > 0:
                res.append(idx)
        cross_1=self.data_array_l0[res[0],0]
        cross_2=self.data_array_l0[res[1],0]
        return cross_1, cross_2

