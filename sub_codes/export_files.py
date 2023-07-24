"""
The class of functions used to make the interesting figures and output the data into .csv files
"""
import numpy as np
import matplotlib.pyplot as plt

class output():
    def __init__(self,epslistfull,data_array_l0,data_array_l1,data_array_l2,der_datal0,der_data,cross_1,cross_2,epsilon12,epsilon14):
        """A class that makes the figures and outputs the data in .csv files.

        Args:
            epslistfull (ndarray): 1d array containing the E_\lambda curves for the lowest channel
            data_array_l0 (ndarray): 1d array containing the E_\lambda curves for the l=0 channel
            data_array_l1 (ndarray): 1d array containing the E_\lambda curves for the l=1 channel
            data_array_l2 (ndarray): 1d array containing the E_\lambda curves for the l=2 channel
            der_datal0 (ndarray): 1d array containing the W_\lambda curves for the l=0 channel
            der_data (ndarray): 1d array containing the W_\lambda curves for the lowest channel
            cross_1 (float): the first crossing between the l=0 and l=1 channel
            cross_2 (float): the second crossing between the l=0 and l=1 channel
            epsilon12 (float): the eigenvalue of the subleading order term (\lambda^{1/2})
            epsilon14 (float): the eigenvalue of the next order term (\lambda^{1/4})
        """
        self.epslistfull=epslistfull
        self.data_array_l0=data_array_l0
        self.data_array_l1=data_array_l1
        self.data_array_l2=data_array_l2
        self.der_datal0=der_datal0
        self.der_data=der_data
        self.cross_1=cross_1
        self.cross_2=cross_2
        self.epsilon12=epsilon12
        self.epsilon14=epsilon14

    def print_figs(self,x,u0,s,lambdamax):
        """prints the figures that are present in the paper or are interesting.

        Args:
            x (ndarray): 1d array containing the grid
            u0 (ndarray): 1d array containing the minimizing wavefunction at \lambda=lambdamax
            s (float): the spin factor used in the calculation of the HF orbital
            lambdamax (float): the maximum value of \lambda
        """
        l0=plt.scatter(self.data_array_l0[:,0],self.data_array_l0[:,1]) #defining the E_\lambda curve for l=0
        l1=plt.scatter(self.data_array_l1[:,0],self.data_array_l1[:,1]) #defining the E_\lambda curve for l=1
        l2=plt.scatter(self.data_array_l2[:,0],self.data_array_l2[:,1]) #defining the E_\lambda curve for l=2
        plt.legend((l0, l1, l2),('l=0', 'l=1', 'l=2'))
        plt.title('s=%d' %s)
        plt.ylabel(r'$E_\lambda$')
        plt.xlabel(r'$\lambda$')
        plt.savefig('Elambda.png')#plotting the E_\lambda curves for l=0, l=1, l=2
        plt.close()
        plt.scatter(x,u0/x)
        plt.ylabel(r'$\psi_{\lambda=%d}$'%lambdamax)
        plt.xlabel(r'$x$')
        plt.title('l=0, s=%d' %s)
        plt.savefig('psilambda.png') #plotting \psi_\lambda at \lambda=lambdamax for l=0
        plt.close()
        plt.scatter(x,u0)
        plt.ylabel(r'$u_{\lambda=%d}$'%lambdamax)
        plt.xlabel(r'$x$')
        plt.title('l=0, s=%d' %s)
        plt.savefig('ulambda.png')#plotting u_\lambda at \lambda=lambdamax for l=0
        plt.close()
        plt.scatter(self.der_datal0[:,0],self.der_datal0[:,1])
        plt.ylabel(r'$W_\lambda$')
        plt.xlabel(r'$\lambda$')
        plt.title('l=0, s=%d' %s)
        plt.savefig('Wlambda_l0.png') #plotting the W_\lambda curves for l=min (so channel that gives the lowest energy)
        plt.close()
        plt.scatter(self.der_data[:,0],self.der_data[:,1])
        plt.ylabel(r'$W_\lambda$')
        plt.xlabel(r'$\lambda$')
        plt.title('s=%d' %s)
        plt.savefig('Wlambda_lmin.png') #plotting the E_\lambda curves for l=0
        plt.close()

    #Export files and data to .csv files
    def export_files(self,spinflip,w):
        """Exports all the data into .csv files.

        Args:
            spinflip (boolean): spinflip is allowed
            w (float): the \beta spin of the HF orbital
        """
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
        np.savetxt(Elamfile,self.epslistfull, delimiter=",",fmt="%10.5f")
        np.savetxt(Wlamfile,self.der_data, delimiter=",",fmt="%10.5f")
        np.savetxt(Elaml0file,self.data_array_l0, delimiter=",",fmt="%10.5f")
        np.savetxt(Wlaml0file,self.der_datal0, delimiter=",",fmt="%10.5f")
        np.savetxt(Elaml1file,self.data_array_l1, delimiter=",",fmt="%10.5f")
        np.savetxt(Elaml2file,self.data_array_l2, delimiter=",",fmt="%10.5f")
        if self.cross_1==None:
            pass
        else:
            np.savetxt(Crossfile,[self.cross_1,self.cross_2], delimiter=",",fmt="%10.5f")
        np.savetxt(Epsfile,[self.epsilon12,self.epsilon14], delimiter=",",fmt="%10.5f")