# Investigation of the spin dependence of the Møller-Plesset adiabatic connection for one electron systems.
A code used to calculate the hydrogen atom with different fractional spins along the Møller Plesset Adiabatic Connection using the finite difference spectral renormalisation algorithm.

Depends on the following packages:
- [numpy]
- [scipy]
- [matplotlib]
- [sympy]
- [argparse]

## Input parameters to give on the command line:
1. --w: the amount of \beta-spin (float) (default=1.)
2. --spinflip: allows or disallows spinflip  i.e. q=/=w or q=w (bool) (default=False)
3. --gridsize: determines the size of the grid (int) (default=100)
4. --gridfactor: contracts or expands the grid, higher values give more contracted grids (int) (default=40)
5. --lambdastart: the starting value of \lambda (float) (default=0.)
6. --lambdamax: the maximum value of \lambda (float) (default=20.)
7. --stepsize: the size of the steps in \lambda (float) (default=0.05)
8. --nbasis: the number of basisfunctions used in the STO expansion of the HF orbital (int) (default=10)
9. --iter: the maximum number of iterations of the FSR algorithm for every \lambda (int) (default=1000)
10. --laplacianfactor: a constant added to avoid \eps+T to become negative, important for l>0 or \lambda<0" (float) (default=2.)

## Known Issues:
None

## Future implementations:
1. Allow for complex \lambda's, energies and wavefunctions.

## License
Møller-Plesset adiabatic connection Spectral Renormalisation Spin (MPAC_SR_Spin), a code used to calculate the hydrogen atom with different fractional spins along the Møller Plesset Adiabatic Connection using the finite difference spectral renormalisation algorithm.

Copyright (C) 2023 Kimberly J. Daas

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see https://www.gnu.org/licenses/.
