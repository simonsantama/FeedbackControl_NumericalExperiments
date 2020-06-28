"""
This script uses a PID controller to maintain the NHF going into a sample at a constant value.
It is applied to numerical experiments which use the Crank-Nicolson scheme to calculate temperature evolution.
The algorithm corrects the IHF at the next dt, which increases convergence but wouldn't be real in a physical experimental
since there would be a delay of seconds.

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
from pid_controller import PID


# parameters for the numerical experiment
nhf_setpoint = 10000                 # W/m2K
kp = [2,2,2,1.25]
ki = [2,2,2,1.25]
kd = [0.05,0.05,0.05,0.01]
    
# parameters for the Crank-Nicolson solution to heat diffusion
T_initial = 288                      # K
T_air = T_initial                    # K
time_total = 601                     # s
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
upsilons = (alphas*dts)/(2*dx**2)

# define the heat flux as an array which starts at the nhf_setpoint but will change.
for i in range(len(t_grids)):
    q = np.zeros_like(t_grids[i])
    q[0] = nhf_setpoint
    q[1] = nhf_setpoint
    q[2] = nhf_setpoint
    q_arrays.append(q) 
    
# create list with surface boundary conditions to be validated
boundary_conditions_surface = ["Linear", "Non-linear"]
all_data = {}

# model both the linear and non-linear surface boundary conditions
for bc_surface in boundary_conditions_surface:
    start = time.time()
    
    print("---------")
    print(f" Calculting for {bc_surface} boundary condition. Immediate IHF correction.")
    all_data[bc_surface] = {}

    # store the different data with different keys in the general dictionary
    all_data[bc_surface]["temperature_profile"] = {}
    all_data[bc_surface]["surface_temperature"] = {}
    all_data[bc_surface]["incident_heatflux"] = {}
    all_data[bc_surface]["net_heatflux"] = {}
    all_data[bc_surface]["error"] = {}
    
    # iterate over 4 different values of alpha and k
    for i in range(len(alphas)):
        
        # debugging
#        if i in [0,1,2]:
#            continue
        
        k_this = ks[i]
        upsilon_this = upsilons[i]
        q_this = q_arrays[i]
        t_grid_this = t_grids[i]
        alpha_this = alphas[i]
        kp_this = kp[i]
        ki_this = ki[i]
        kd_this = kd[i]
        
        # define initial temperatures array
        T = np.zeros_like(x_grid) + T_initial
    
        # store the different data with different keys in the general dictionary
        all_data[bc_surface]["temperature_profile"][alphas[i]] = {}
        all_data[bc_surface]["surface_temperature"][alphas[i]] = {}
        all_data[bc_surface]["incident_heatflux"][alphas[i]]= {}
        all_data[bc_surface]["net_heatflux"][alphas[i]] = {}
        all_data[bc_surface]["error"][alphas[i]] = {}
        
        # define variables that are used by the PID controller
        error_sum = 0
        error_array = np.zeros_like(t_grid_this)
        nhf_array = np.zeros_like(t_grid_this)
        
        # iterate over every single time step
        for time_step,t in enumerate(t_grid_this[:-2]):
            if time_step == 0:
                temperature_profile = T
                surface_temperature = T[0]
                nhf_array[time_step] = q_this[0]
                error_array[time_step] = 0
            else:
                temperature_profile, surface_temperature, nhf = general_temperatures(
                                                                T_initial, T_air, time_total, k_this, alpha_this, dx, x_grid,
                                                                space_divisions, upsilon_this, bc_surface,
                                                                q_this, h, hc, emissivity, sigma, time_step, T)
                
                # call PID controller to change the incident heat flux as a function of the net heat flux
                last_error = error_array[time_step-1]
                last_nhf = nhf_array[time_step-1]
                nhf_array[time_step] = nhf
                new_q, error_array[time_step], error_sum = PID(nhf, nhf_setpoint, last_error, last_nhf, error_sum, dts[i],
                                  kp_this, ki_this, kd_this)
                
                # update the incident heat flux with the output value from the PID algorithm
                q_this[time_step+2] = new_q
                
#                # PID debugging
#                print(time_step)
#                print(f"Set NHF: {nhf_setpoint}")
#                print(f"Current ihf: {q_this[time_step]}")
#                print(f"Next ihf: {q_this[time_step + 1]}")            
#                print(f"Previous nhf:{last_nhf}")
#                print(f"Current nhf: {nhf}")
#                print(f"Error sum:{error_sum}")
#                print(f"Current error: {error_array[time_step]}")
#                time.sleep(0.5)
            
            # update data in the all_data dictionary that is saved to use later
            all_data[bc_surface]["temperature_profile"][alphas[i]][t] = temperature_profile
            all_data[bc_surface]["surface_temperature"][alphas[i]][t] = surface_temperature
            all_data[bc_surface]["net_heatflux"][alphas[i]][t] = nhf_array[time_step]
            all_data[bc_surface]["incident_heatflux"][alphas[i]][t] = q_this[time_step]
            all_data[bc_surface]["error"][alphas[i]][t] = error_array[time_step]
            
            # update temperature array
            T = temperature_profile.copy()
        
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