"""
This script uses a PID controller to maintain the NHF going into a sample at a constant value.
It is applied to numerical experiments which use the Crank-Nicolson scheme to calculate temperature evolution.

Boundary conditions
------------------
Surface: 
    a) q_inc - q _conv - q_rad (where q_inc is a function of time)
    b) q_inc - q_losses (using a total heat transfer coefficient)
Unexposed: 
    a) insulated boundary

1-D. Inet solid. Homogoneous. Constant properties.
"""

# import python libraries
import numpy as np
import pickle
import time

# import my own function
from cn import general_temperatures


# parameters for the numerical experiment
nhf_setpoint = 5000
h = 45

# parameters for the Crank-Nicolson solution to heat diffusion
T_initial = 288                      # K
T_air = T_initial                    # K
time_total = 601                      # s
sample_length = 0.025                # m
space_divisions = 100                # -
h = 45                               # W/m2K for linearised surface bc with constant heat transfer coefficient
hc = 10                              # W/m2K convective heat transfer coefficient
emissivity = 1                       # -
sigma = 5.67e-8                      # W/m2K4

# define a range of properties to be used to evaluate the response of the material (these are syntethic properties)
ks = np.linspace(0.2, 0.5, 4)           # W/mK
alphas = np.linspace(1e-7, 1.0e-6, 4)   # J/kgK

# create spatial mesh
dx = sample_length/(space_divisions - 1)
x_grid = np.array([i * dx for i in range(space_divisions)])

# define time step based on the spatial mesh for each alpha and create mesh
dts = (1 / 3) * (dx**2 / alphas)
time_divisions = time_total / dts
t_grids = []
q_arrays = []
for i,dt in enumerate(dts):
    t_grids.append(np.array([n * dt for n in range(int(time_divisions[i]))]))
    # define the heat flux as an array which starts at the nhf_setpoint but will change.
    q_arrays.append(np.zeros_like(t_grids[i])+ 20000) 
    
upsilons = (alphas*dts)/(2*dx**2)

# create list with surface boundary conditions to be validated
boundary_conditions_surface = ["Linear", "Non-linear"]
all_data = {}

########
# -- First set of experiments can immediatly correct the Incident Heat Flux
########


# model both the linear and non-linear surface boundary conditions
for bc_surface in boundary_conditions_surface:
    start = time.time()
    
    print("---------")
    print(f" Calculting for {bc_surface} boundary condition. Immediate IHF correction.")
    print("---------")
    all_data[bc_surface] = {}

    # store the different data with different keys in the general dictionary
    all_data[bc_surface]["temperature_profile"] = {}
    all_data[bc_surface]["surface_temperature"] = {}
    all_data[bc_surface]["incident_heatflux"] = {}
    all_data[bc_surface]["net_heatflux"] = {}
    
    
    # iterate over 4 different values of alpha and k
    for i in range(len(alphas)):
        k_this = ks[i]
        upsilon_this = upsilons[i]
        q_this = q_arrays[i]
        t_grid_this = t_grids[i]
        alpha_this = alphas[i]
        
        # define initial temperatures array
        T = np.zeros_like(x_grid) + T_initial
    
        # store the different data with different keys in the general dictionary
        all_data[bc_surface]["temperature_profile"][alphas[i]] = {}
        all_data[bc_surface]["surface_temperature"][alphas[i]] = {}
        all_data[bc_surface]["incident_heatflux"][alphas[i]]= {}
        all_data[bc_surface]["net_heatflux"][alphas[i]] = {}
        
        # iterate over every single time step
        for time_step,t in enumerate(t_grid_this[1:]):
            time_step=time_step+1
    
            temperature_profile, surface_temperature, nhf = general_temperatures(
                                                            T_initial, T_air, time_total, k_this, alpha_this, dx, x_grid,
                                                            space_divisions, upsilon_this, bc_surface,
                                                            q_this, h, hc, emissivity, sigma, time_step, T)
    
              
            all_data[bc_surface]["temperature_profile"][alphas[i]][t] = temperature_profile
            all_data[bc_surface]["surface_temperature"][alphas[i]][t] = surface_temperature
            all_data[bc_surface]["net_heatflux"][alphas[i]][t] = nhf
            
            # update temperature array
            T = temperature_profile.copy()
            
            
            # update the new value of the incident heat flux
            all_data[bc_surface]["incident_heatflux"][alphas[i]][t] = q_this[time_step]
        
    print(f" time taken for {bc_surface} boundary condition: {np.round(time.time() - start,2)} seconds")      
    

# condense all data to be saved including important plotting parameters
total_data = {"all_data": all_data, #  includes temperature profile, surface temperature, incident HF and net HF. 
              "extra_data": { 
                      "x_grid":x_grid, 
                      "time_total": time_total, 
                      "alpha": alphas, 
                      "k": ks,},
    }

# save in a pickle to retrieve and plot later
with open('total_data_backinsulated_immediatecorrection.pickle', 'wb') as handle:
    pickle.dump(total_data, handle)
    print("\n")
    print("All data saved into total_data_backinsulated_immediatecorrection.pickle")

#def PID_IHF(Input, Setpoint, previous_time, lastErr, lastInput, errSum):
#    """
#    Uses a PID algorithm to calculate the IHF based on target and current nhf
#    Output: IHF. Input: NHF: Setpoint: Target NHF.
#    See: http://brettbeauregard.com/blog/2011/04/improving-the-beginners-pid-introduction/
#    """
#
#    # Define the maximum and minimum voltages
#    Output_max = 4
#    Output_min = 0
#
#    # Define the k constants (determined experimentally)
#    kp = 0.01
#    ki = 0.005
#    kd = 0.0025
#
#    # How long since we last calculated
#    now = datetime.now().time()
#    # Calculates time difference between last scan and this one
#    timeChange = (datetime.combine(date.min, now) - datetime.combine(date.min, previous_time)).total_seconds()
#
#    # Compute all the working error variables
#    error = Setpoint - Input
#    errSum += error * timeChange
#    # dERr ia according to the initial design, but later dInput is used
#    dErr = (error - lastErr) / timeChange
#    dInput = (Input - lastInput) / timeChange
#
#    # Compute the Output
#    Output = kp * error + ki * errSum - kd * dInput
#
#    if Output > Output_max:
#        Output = Output_max
#    elif Output < Output_min:
#        Output = Output_min
#
#    return Output, now, error, Input, errSum

########
# -- Second set of expeimerents can only correct the incident heat flux once per every second
########