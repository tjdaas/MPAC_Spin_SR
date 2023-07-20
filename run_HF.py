#Code written by Derk P. Kooi
#These are all numerical integrals where the general integrals were worked out in mathematica and the solution is put here
import numpy as np
from scipy.special import factorial,hyp2f1
from sympy.physics.wigner import wigner_3j
from scipy.linalg import eigh

def STOnorm(n=1, a=1.):
    ### STO normalization ###
    return (2**n) *a**(n+1) / np.sqrt(a*n*factorial(2*n-1))

def kin(n1=1, n2=1, a1=1., a2=1.):
    ### Radial kinetic energy elements ###
    return -1/2 * np.power(a1+a2,-1-n1-n2)*(a2*a2*(n1-1)*n1-2*a1*a2*n1*n2+a1*a1*(n2-1)*n2)*factorial(n1+n2-2)

def kinnorm(n1=1, n2=1, a1=1., a2=1.):
    ### Radial kinetic energy elements normalized ###
    return kin(n1, n2, a1,a2)*(2**n1) *a1**(n1+1) / np.sqrt(a1*n1*factorial(2*n1-1))*(2**n2) *a2**(n2+1) / np.sqrt(a2*n2*factorial(2*n2-1))

def S(n1=1, n2=1, a1=1., a2=1.):
    ### Overlap elements (radial) ###
    return np.power(a1+a2,-1-n1-n2) * factorial(n1+n2)

def Snorm(n1=1, n2=1, a1=1., a2=1.):
    ### Overlap elements normalized ###  
    return np.power(a1+a2, -1-n1-n2)*2**(n1+n2)*a1**(n1+1)*a2**(n2+1)*factorial(n1+n2) / (np.sqrt(a1*n1*factorial(2*n1-1)*a2*n2*factorial(2*n2-1)))

def Vext(n1=1, n2=1, a1=1., a2=1.):
    ### External potential -1/r, not multiplied with Z ###
    return - np.power(a1+a2, -n1-n2)*factorial(n1+n2-1)

def Vextnorm(n1=1, n2=1, a1=1., a2=1.):
    ### External potential normalized
    return - 2**(n1+n2)*a1**(n1+1)*a2**(n2+1)*np.power(a1+a2, -n1-n2)*factorial(n1+n2-1) / (np.sqrt(a1*n1*factorial(2*n1-1)*a2*n2*factorial(2*n2-1)))

def Veff(n1=1, n2=1, a1=1., a2=1.,l=1):
    ### Expectation value of effective potential l(l+1)/(2r^2) ###
    return 0.5 * l*(l+1)*np.power(a1+a2, 1-n1-n2)*factorial(n1+n2-2)

def Veffnorm(n1=1,n2=1,a1=1.,a2=1.,l=1):
    ### Effective potential normalized ###
    return 2**(n1+n2-1)*l*(l+1)*np.power(a1+a2, 1-n1-n2)*factorial(n1+n2-2)*a1**(n1+1)*a2**(n2+1)/(np.sqrt(a1*n1*factorial(2*n1-1)*a2*n2*factorial(2*n2-1)))

def hyp2f1reg(a, b, c, z):
    return hyp2f1(a,b,c,z) / factorial(c-1)

def radint(ni=1, nj=1, nk=1, nl=1, ai=1., aj=1., ak=1., al=1., l=0):
    ### Radial part of the coulomb integral for given l. ###
        return (aj+al)**(-1-nj-nl)*((ai+ak)**(l-ni-nk)*(aj+al)**(-l)*factorial(ni+nk-l-1)*factorial(l+nj+nl)+(aj+al)**(-ni-nk)*factorial(ni+nj+nk+nl)*(factorial(l+ni+nk)*hyp2f1(1+l+ni+nk, 1+ni+nj+nk+nl,2+l+ni+nk, - (ai+ak)/(aj+al))/factorial(1+l+ni+nk)-factorial(ni+nk-l-1)*hyp2f1(ni+nk-l, 1+ni+nj+nk+nl, 1-l+ni+nk, -(ai+ak)/(aj+al))/factorial(-l+ni+nk)))

def coulombint(ni=1, nj=1, nk=1, nl=1, ai=1., aj=1., ak=1., al=1.,li=0,lj=0,lk=0,ll=0,mi=0,mj=0,mk=0,ml=0):
    result = 0.
    if mk-mi == mj-ml:
        for l in range(max([abs(li-lk),abs(lj-ll),abs(mk-mi)]), min([li+lk, lj+ll])+1):
            if mk-mi == 0:
                result+= radint(ni,nj,nk,nl,ai,aj,ak,al,l)*np.sqrt((2*li+1)*(2*lk+1)*(2*lj+1)*(2*ll+1))*wigner_3j(li, lk, l, 0, 0, 0)*wigner_3j(lj,ll,l,0,0,0)*wigner_3j(li, lk, l, -mi, mk, 0)*wigner_3j(lj,ll,l,-mj, ml, 0)*(-1.)**(mi+mj)
            else:
                result+= radint(ni,nj,nk,nl,ai,aj,ak,al,l)*np.sqrt((2*li+1)*(2*lk+1)*(2*lj+1)*(2*ll+1))*wigner_3j(li, lk, l, 0, 0, 0)*wigner_3j(lj,ll,l,0,0,0)*(wigner_3j(li, lk, l, -mi, mk, mk-mi)*wigner_3j(lj,ll,l,-mj, ml, mi-mk)*(-1.)**(mk-mi)+wigner_3j(li, lk, l, -mi, mk, mi-mk)*wigner_3j(lj,ll,l,-mj, ml, mk-mi)*(-1.)**(mi-mk))*(-1.)**mi*(-1.)**mj
    return result  

def buildF0Smat(nbasis,basisn,basisl,basism,basisa,Z,normvec):
    ### Build unchanging part of Fock matrix and overlap Matrix ###
    F0mat = np.zeros((nbasis,nbasis))
    Smat = np.zeros((nbasis,nbasis))
    for i in range(nbasis):
        for j in range(nbasis):
            if basisl[i] == basisl[j]:
                F0mat[i,j] = (kin(basisn[i],basisn[j],basisa[i],basisa[j])+Z*Vext(basisn[i],basisn[j],basisa[i],basisa[j])+Veff(basisn[i],basisn[j],basisa[i],basisa[j],basisl[i]))*normvec[i]*normvec[j]
                Smat[i,j] = S(basisn[i],basisn[j],basisa[i],basisa[j])*normvec[i]*normvec[j]
            else:
                F0mat[i, j] = 0
                Smat[i, j] = 0  
    return F0mat,Smat

def buildcoulombmat(nbasis,basisn,basisl,basism,basisa,s,normvec):
    ### Build coulomb integrals symmetrized and normalized ###
    coulombmat = np.zeros((nbasis,nbasis,nbasis,nbasis))
    for i in range(nbasis):
        for j in range(nbasis):
            for k in range(nbasis):
                for l in range(nbasis):
                    coulombmat[i, j, k, l] = normvec[i]*normvec[j]*normvec[k]*normvec[l]*(2*coulombint(basisn[i],basisn[j],basisn[k],basisn[l],basisa[i],basisa[j],basisa[k],basisa[l],basisl[i],basisl[j],basisl[k],basisl[l],basism[i],basism[j],basism[k],basism[l])-2*s*coulombint(basisn[i],basisn[j],basisn[l],basisn[k],basisa[i],basisa[j],basisa[l],basisa[k],basisl[i],basisl[j],basisl[l],basisl[k],basism[i],basism[j],basism[l],basism[k]))
    return coulombmat

def buildvhmat(nbasis,Cmat,coulombmat):
    ### Build Hartree-Exchange matrix with einsum and normalized orbitals ###
     return np.einsum('x,w,iwjx->ij',Cmat[::,0],Cmat[::,0],coulombmat)

def calculate_energy(eigenvals,Cmat,vhxmat,nbasis):
    ### Calculate energy corresponding to the restricted Hartree-Fock state ###
    energy = 0.
    energy = eigenvals[0]
    for i in range(nbasis):
        for j in range(nbasis):
            energy -= 1/4*Cmat[i,0]*Cmat[j,0]*vhxmat[i,j]
    return energy