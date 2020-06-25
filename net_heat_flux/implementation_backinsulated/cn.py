"""
Crank-Nicholson implementation of 1D heat transfer
"""

# import libraries
import numpy as np

def general_temperatures(T_initial, T_air, time_total, k, alphas, dx, x_grid, space_divisions,
                    upsilon, bc_surface, q, h, hc, emissivity, sigma, time_step, T):
    
    """
    General function to implement CN.
    Creates different functions of time for the incident heat flux.
    Returns dictionary that includes all calculated temperature for given boundary conditions evaluated at different
    properties and parameters
    
    CN Scheme:
    ---------
        [A]{T^n+1} = [B]{T^n} = [b]
    
    
    Parameters:
    ----------    
    T_initial: initial temperature in K
        int
        
    T_air: air(infinity) temperature in K for convective losses. Usually equals T_initial but not necessary
        int        

    time_total: total time for calculations
        int
        
    k: thermal conductivity in W/mK.
        float
        
    alpha: thermal diffusivity in m2/s.
        float
        
    dx: size of cell in space domain in m
        float 
    
    x_grid: one dimensional spatial domain
        np.array

    space_divisions: number of nodes in the spatial domain
        int

    t_grid: list where each entry is the temporal domain for a given alpha and dt
        list
        
    upsilon: Fourier number divided by 2
        float
        
    bc_surface: surface boundary condition (linear or non-linear)
        str
        
    q: array of values for heat flux calculations to determine heat flux as a function of time
        np.array
    
    h: total heat transfer coefficient for the linearised surface boundary condition
        int
        
    hc: convective heat transfer coefficient
        int

    emissivity: surface emmisivity, assumed constant
        float
        
    sigma: Stefan Boltzman constant
        float
        
    time_step: current time step
        int
        
    T: array of present temperatures
        np.array

    Returns:
    -------
    Temperature: dictionary with temperature arrays for different alpha and k
        dict
    
    """
        
    # create tridiagonal matrix A
    A = tridiag_matrix(bc_surface, upsilon, space_divisions, dx, k, T, h, hc, emissivity, sigma)

    
    # create vector b
    b = vector_b(bc_surface, upsilon, space_divisions, dx, k, T, T_initial, T_air, q, h, hc, 
                 emissivity, sigma, time_step)
    
    # calculate value of future temperature
    Tn = np.linalg.solve(A,b)
    
    nhf = -k *(Tn[0]-Tn[1])/(x_grid[0]-x_grid[1])
            
    return Tn, Tn[0], nhf   
    
    

# function to create tri-diagonal matrix
def tridiag_matrix(bc_surface, upsilon, space_divisions, dx, k, T, h, hc, emissivity, sigma):
    """
    Creates tridiagonal matrix A
    Linear system to be solved is Ax = b, and x represents temperature values at time n+1

    Parameters:
    ----------
    bc_surface: boundary condition at the surface
        str
    
    upsilon: Fourier number divided by 2. Upsilon = alpha*dt/2*dx2
        float
        
    space_divisions: number of nodes in the spatial domain
        int
        
    dx: size of cell in space domain in m
        float
        
    k: thermal conductivity in W/mK
        float
        
    T: array of present temperatures
        np.array
        
    h: total heat transfer coefficient for the linearised surface boundary condition
        int
        
    hc: convective heat transfer coefficient
        int
        
    emissivity: surface emmisivity, assumed constant
        float
        
    sigma: Stefan Boltzman constant
        float
    
    Return:
    ------
    
    A: matrix to be inverted
        np.array
    
    """
    # create tri-diagonal matrix
    A = np.diagflat([-upsilon for i in range(space_divisions - 1)], -1) +\
        np.diagflat([1 + 2 * upsilon for i in range(space_divisions)]) +\
        np.diagflat([-upsilon for i in range(space_divisions - 1)], 1)

    # adjust matrix depending on the boundary condition at the exposed surface
    if bc_surface == "Linear":
        A[0,0] = 1 + 2*upsilon + 2*upsilon*dx*h/k
        A[0,1] = -2*upsilon
    
    elif bc_surface == "Non-linear":
#        A[0,0] = 1 + 2*upsilon + 2*upsilon*hc*dx/k + 2*upsilon*emissivity*sigma*dx*T[0]**3
        A[0,0] = 1 + 2*upsilon + 2*dx*hc*upsilon/k+ 8*emissivity*sigma*dx*upsilon*T[0]**3/k
        A[0,1] = -2*upsilon
    
    # adjust matrix for the back boundary conditions
    A[-1, -2] = - 2 * upsilon
    A[-1, -1] = 1 + 2 * upsilon

    return A




def vector_b(bc_surface, upsilon, space_divisions, dx, k, T, T_initial, T_air, q, h, hc, emmissivity, sigma, time_step):
    """
    Calculates vector b. Right hand side of linear system of equations

    Parameters:
    ----------
    bc_surface: boundary condition at the surface
        str
    
    upsilon: Fourier number divided by 2. Upsilon = alpha*dt/2*dx2
        float
        
    space_divisions: number of nodes in the spatial domain
        int
        
    dx: size of cell in space domain in m
        float
        
    k: thermal conductivity in W/mK
        float
        
    T: array of present temperatures
        np.array

    T_initial: initial temperature in K
        int
        
    T_air: air(infinity) temperature in K for convective losses. Usually equals T_initial but not necessary
        int   
        
    q: array of size t_grid that contains the incident heat flux at each time step
        np.array
        
    h: total heat transfer coefficient for the linearised surface boundary condition
        int
        
    hc: convective heat transfer coefficient
        int

    emissivity: surface emmisivity, assumed constant
        float
        
    sigma: Stefan Boltzman constant
        float
        
    time_step: present iteration number
        int
    
    Returns:
    -------
    b: vector to solve linear system of equations
        np.array
    """
    
    # matrix B, similar to matrix A but multiplies T at present
    B = np.diagflat([upsilon for i in range(space_divisions - 1)], -1) +\
        np.diagflat([1 - 2 * upsilon for i in range(space_divisions)]) +\
        np.diagflat([upsilon for i in range(space_divisions - 1)], 1)

    # Calculate vector b
    b = np.zeros(space_divisions)
    b[1:-1] = B[1:-1, :].dot(T)
    
    # adjust vector for the front boundary condition
    if bc_surface == "Linear":
        b[0] = 2*upsilon*T[1] + (1 - 2*upsilon - upsilon*2*dx*h/k)*T[0] + 4*upsilon*dx*h*T_air/k + \
            2*dx*upsilon/k * (q[time_step]+q[time_step - 1])
    
    elif bc_surface == "Non-linear":
        b[0] = 2*upsilon*T[1] + (1- 2*upsilon - 2*dx*hc*upsilon/k)*T[0] + 4*dx*hc*upsilon*T_air/k + \
            4*emmissivity*sigma*dx*upsilon*T[0]**4/k + 2*dx*upsilon/k * (q[time_step]+q[time_step - 1])
    
    # adjust vector for the back boundary condition
    b[-1] = (1 - 2*upsilon)*T[-1] + 2*upsilon*T[-2]

    return b    
    



