"""
This script uses a PID controller to maintain the NHF absorned by the sample at a constant value.
It is applied to numerical experiments which use the Crank-Nicolson scheme to calculate temperature evolution.
The PID controller is called:
    1. At every single time step in the temporal grid (unrealistic for physical experiments)
    2. After every second (logging time for temperatures but doesn't account for FPA delay)
    3. After every 5 seconds (comparison with previous two)

Boundary conditions
------------------
Surface: 
    a) q_inc - q _conv - q_rad (where q_inc is a function of time)
    b) q_inc - q_losses (where q_inc is a function of time and using a total heat transfer coefficient hT)
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
from alldata_to_1Hz import convert_1Hz

# parameters for the numerical experiment
nhf_setpoint = 10000                 # W/m2K

# PID parameters were tuned manually by looking at the evolution of the nhf
kps = {0: [1.75,1.75,2,2],
       1: [1.05,1.1,1.1,1.1],
       2: [1.15,1.15,1.2,1.2]}
kis = {0: [0.75,1,1,1],
       1: [0.95,0.95,0.95,0.95],
       2: [0.85,0.85,0.75,0.65]}
kds = {0: [0,0,0,0],
       1: [0.025,0,0,0],
       2: [0.0025,0,0,0]}
# LISTS USED FOR TUNNING THE PID PARAMETERS
alpha_not_considered = []
scenario_not_considered = []


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

# material properties
ks = np.linspace(0.2, 0.5, 4)           # W/mK
alphas = np.linspace(1e-7, 1.0e-6, 4)   # J/kgK

# create spatial mesh
dx = sample_length/(space_divisions - 1)
x_grid = np.array([i * dx for i in range(space_divisions)])

# define time step based on the spatial mesh for each alpha and create mesh
dts = (1 / 3) * (dx**2 / alphas)
time_divisions = time_total / dts
t_grids = []
for i,dt in enumerate(dts):
    t_grids.append(np.array([n * dt for n in range(int(time_divisions[i]))]))
upsilons = (alphas*dts)/(2*dx**2)

# define the incident heat flux as an array the size of each temporal grid
q_arrays = []
for t_grid in t_grids:
    q = np.zeros_like(t_grid)
    q_arrays.append(q)

# create list with type of surface boundary conditions to be analysed
boundary_conditions_surface = ["Linear", "Non-linear"]

# create dictionary to store all the data
all_data = {}

time_lag_scenarios = ["immediate", "1second", "5seconds"]

for scenario_number, scenario in enumerate(time_lag_scenarios):
    
    # DEBUGGING
    if scenario_number in scenario_not_considered:
        continue
    
    
    print("\n---------")
    kp = kps[scenario_number]
    kd = kds[scenario_number]
    ki = kis[scenario_number]
    
    # start timer to know how long it takes per scenario
    start = time.time()

    # define the time lag (time it takes for the PID to be called)    
    if scenario == "immediate":
        time_lag = 0
    else:
        time_lag = int(scenario.split("second")[0])
    # number of readings which approximately cover the time lag desired
    number_timereadings = (time_lag/dts).astype(int)

    # model both the linear and non-linear surface boundary conditions
    for bc_surface in boundary_conditions_surface:
        start = time.time()
    
        print(f" calculating for {bc_surface} boundary condition. {time_lag} second correction.")
        all_data[bc_surface] = {}

        # store the different data with different keys in the general dictionary
        all_data[bc_surface]["temperature_profile"] = {}
        all_data[bc_surface]["surface_temperature"] = {}
        all_data[bc_surface]["incident_heatflux"] = {}
        all_data[bc_surface]["net_heatflux"] = {}
        all_data[bc_surface]["error"] = {}
    
        # iterate over 4 different values of alpha and k
        for i in range(len(alphas)):
            
            # DEBUGGING
            if i in alpha_not_considered:
                continue
            
            # define initial temperatures array
            T = np.zeros_like(x_grid) + T_initial
        
            # extend dictionaries for each alpha value
            all_data[bc_surface]["temperature_profile"][alphas[i]] = {}
            all_data[bc_surface]["surface_temperature"][alphas[i]] = {}
            all_data[bc_surface]["incident_heatflux"][alphas[i]]= {}
            all_data[bc_surface]["net_heatflux"][alphas[i]] = {}
            all_data[bc_surface]["error"][alphas[i]] = {}    
    
            # define variables that are used by the PID controller
            error_sum = 0
            error_array = np.zeros_like(t_grids[i])
            nhf_array = np.zeros_like(t_grids[i])    
    
            # create a counter to determine when the PID controller needs to be called
            time_lag_counter = 0
            
            # since operations are defined 2*time lags in advace, only iterate over t_grid from 0 till end - 2*time_lag
            if scenario_number == 0:
                maximum_limit_tgrid = 2
            else:
                maximum_limit_tgrid = 2*number_timereadings[i]
            for time_step,t in enumerate(t_grids[i][:-maximum_limit_tgrid]):
                q_arrays[i][0:maximum_limit_tgrid] = nhf_setpoint
    
                # first time step
                if time_step == 0:
                    temperature_profile = T
                    surface_temperature = T[0]
                    # last_nhf saves the nhf at a frequency that corresponds to the time lag
                    last_nhf = q_arrays[i][time_step]
                    # nhf array saves the actual nhf at every time step
                    nhf_array[time_step] = q_arrays[i][time_step]
                    # last_error saves the error at a frequency that corresponds to the time lag
                    last_error = 0
                    # error array saves the actual array at every time step
                    error_array[time_step] = 0
                
                # all other time steps
                else:
                    # first, calculate the new temperature for the next time step (Crank-Nicolson)
                    temperature_profile, surface_temperature, nhf = general_temperatures(
                                                                    T_initial, T_air, time_total, ks[i], alphas[i], 
                                                                    dx, x_grid,space_divisions, upsilons[i], bc_surface,
                                                                    q_arrays[i], h, hc, emissivity, sigma, time_step, T)
                    
                    # update temperature array, net heat flux and surface temperature
                    T = temperature_profile.copy()
                    surface_temperature = T[0]
                    nhf_array[time_step] = nhf
                    error_array[time_step] = nhf_setpoint - nhf
                    
                    # call the PID only once every time lag
                    if time_lag_counter < number_timereadings[i]:
                        time_lag_counter += 1
                    else:
                        # call PID
                        new_q, new_error, error_sum = PID(nhf, nhf_setpoint, last_error, last_nhf, 
                                                              error_sum, dts[i],kp[i], ki[i], kd[i])
    
#                        # ---- PID debugging
#                        print("\n")
#                        print("PID called")
#                        print(f"Scenario: {scenario}")
#                        print(f"alpha: {alphas[i]}. {i+1}/{len(alphas)}")      
#                        print(f"Time step:  {time_step}/{len(t_grids[i])}")
#                        print(f"Time:  {t}")
#                        print(f"Set NHF: {nhf_setpoint}")
#                        print("-----")
#                        print(f"Current IHF: {q_arrays[i][time_step]}")
#                        print(f"Next IHF:    {new_q}")            
#                        print(f"Previous nhf:{last_nhf}")
#                        print(f"Current nhf: {nhf}")
#                        print(f"Error sum: {error_sum}")
#                        print(f"Current error: {nhf_setpoint - nhf}")
#                        time.sleep(0.5)
#                        # ----
    
                        # update parameters
                        q_arrays[i][time_step+1:time_step+maximum_limit_tgrid] = new_q
                        last_nhf = nhf
                        last_error = new_error
    
                        # update the counter
                        time_lag_counter = 0
                        
                # update data in the all_data dictionary that is saved to use later
                all_data[bc_surface]["temperature_profile"][alphas[i]][t] = temperature_profile
                all_data[bc_surface]["surface_temperature"][alphas[i]][t] = surface_temperature
                all_data[bc_surface]["net_heatflux"][alphas[i]][t] = nhf_array[time_step]
                all_data[bc_surface]["incident_heatflux"][alphas[i]][t] = q_arrays[i][time_step]
                all_data[bc_surface]["error"][alphas[i]][t] = error_array[time_step]
                    
    
    # condense all data to be saved
    total_data = {"all_data": all_data,  
                  "extra_data": { 
                          "x_grid":x_grid, 
                          "time_total": time_total, 
                          "alpha": alphas, 
                          "k": ks,},
        }
    
    # save in a pickle
    with open(f"total_data_backinsulated_{scenario}_correction.pickle", 'wb') as handle:
        pickle.dump(total_data, handle)
        print(f"  total_data_backinsulated_{scenario}_correction.pickle")
        
    # convert data to 1 Hz
    total_data_1Hz = convert_1Hz(total_data, scenario)

    # save 1Hz data in a pickle
    with open(f"total_data_backinsulated_{scenario}_correction_1Hz.pickle", 'wb') as handle:
        pickle.dump(total_data_1Hz, handle)
        print(f"  total_data_backinsulated_{scenario}_correction_1Hz.pickle")
        
    print(f"Time taken for {scenario} correction: {np.round(time.time() - start,2)}")






