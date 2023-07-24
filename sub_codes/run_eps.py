""""
A file that calculates the \epsilon_{1/2} and \epsilon_{1/4} of the subleading order term (\lambda^{1/2}) and the term at \lambda^{1/4}.
"""

import numpy as np
from scipy.integrate import cumtrapz,trapz

def epsilon_12(x,s,D2,N,I):
    """solves the eigevalue equation of \epsilon_{1/2} of the subleading order term

    Args:
        x (ndarray): 1d array containing the grid
        s (float): the spinfactor used in calculating the HF orbital
        D2 (ndarray): 2d array containing the laplacian
        N (integer): the number of grid points
        I (ndarray): 2d array containing the identity matrix

    Returns:
        float,ndarray: the epsilon_{1/2} and the minimizing wavefunction u_{1/2}
    """
    ## Solving the l dependent eigenvalue of epsilon_12 for l=0.
    c=30 #constant in case the laplacian becomes positive
    l=0 # l=0 gives the lowest value in the strong coupling limit
    V_eff=l*(l+1)/(2*x**2) #the radial kinetic energy
    max_num_iterat=1000  #number of iterations
    u= x**(l+1)*np.exp(-x) #defining the initial guess
    coef_ini=cumtrapz(np.absolute(u)**2,x,initial=0)                                 
    u=u/np.sqrt(coef_ini[N-1]) #normalising the initial guess
    #now we start our iterations 
    for its in range(0,max_num_iterat):
        
        #K integrals worked out
        integrand_0   = x**(l+1)*u
        integrand_inf = x**(-l)*u
        I_int_num_0 = cumtrapz(integrand_0,x,initial=0)
        I_int_num_inf = trapz(integrand_inf,x) - cumtrapz(integrand_inf,x,initial=0)
        Total_pot = (s/(2*l+1))*((x**(-l))*I_int_num_0 + (x**(l+1))*I_int_num_inf)
        
        #Compute T, J+V_ext+V_eff and K
        u_double_prime=np.matmul(D2,u)
        KE=0.5*cumtrapz(u*u_double_prime,x,initial=0)
        Pot_energy=cumtrapz(np.absolute(u)**2*(V_eff+(x**(2))/6),x,initial=0)
        Pot_energy_2=cumtrapz(u*Total_pot,x,initial=0)
        
        #calculate normalization constant and epsilon_{1/2}
        denom=cumtrapz(np.absolute(u)**2,x,initial=0)
        epsilon12=(-KE[N-1]+Pot_energy[N-1] + Pot_energy_2[N-1])/denom[N-1]
        
        # calculate psi_(n+1) by solving the matrix equation
        RHS=u*(-c+(x**(2))/6) +  Total_pot 
        M_mat=-c*I+0.5*D2+epsilon12*I-np.diag(V_eff)
        u_new=np.linalg.solve(M_mat,RHS)
        u=u_new #update wavefunctions  
        coef_new=cumtrapz(np.absolute(u)**2,x,initial=0)                                   
        u=u/np.sqrt(coef_new[N-1]) #normalisation of the new wavefunction
    return epsilon12, u

def epsilon_14(x,s,u):
    """finds the eigevalue \epsilon_{1/4} using the wigner 2n+1 rule.

    Args:
        x (ndarray): 1d array containing the grid
        s (float): the spinfactor used in calculating the HF orbital
        u (ndarray): 1d array containing the minimizing wavefunction of the subleading order (u_{1/2})

    Returns:
        float: the eigenvalue (\epsilon_{1/4}) at order \lambda^{1/4}
    """
    ## calculating the epsilon_14 using the 2n+1 trick.
    Pot_energy=((1/x)+(x**3)/6)*np.absolute(u)**2 #Compute J
    #K integrals worked out
    integrand_0_1   = x**(1)*u 
    integrand_0_2   = x**(2)*u
    I_int_num_0_1 = cumtrapz(integrand_0_1,x,initial=0) 
    I_int_num_0_2 = cumtrapz(integrand_0_2,x,initial=0)
    Total_pot = ((x**(1))*I_int_num_0_1 + I_int_num_0_2) 
    Pot_energy_2=u*Total_pot #Compute K
    epsilon14=-trapz(Pot_energy+(2*s)*Pot_energy_2 ,x) #compute \epsilon_{1/4}
    return epsilon14
