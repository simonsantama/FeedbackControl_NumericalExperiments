"""
This script uses a PID controller to maintain the surface temperature of the sample at a constant value.
It is applied to numerical experiments which use the Crank-Nicolson scheme to calculate temperature evolution.
The PID controller is called:
    1. At every single time step in the temporal grid (unrealistic for physical experiments)
    2. After every second (logging time for temperatures but doesn't account for FPA delay)
    3. After every 5 seconds (comparison with previous two)

Boundary conditions
------------------
Surface: 
    a) q_inc - q _conv - q_rad (where q_inc is a function of time) -> Non-linear
    b) q_inc - q_losses (where q_inc is a function of time and using a total heat transfer coefficient hT) -> Linear
Unexposed: 
    a) insulated boundary

1-D. Inet solid. Homogoneous. Constant properties.
"""

# import python libraries
import numpy as np
import pickle
import time
#
# import my own function
from cn import general_temperatures
from pid_controller import PID
from alldata_to_1Hz import convert_1Hz

# PID parameters were tuned manually
kps = {"immediate": [100,100],
       "1second": [100,0],
       "5seconds": [100,0]}

kis = {"immediate": [20,20],
       "1second": [15,0],
       "5seconds": [10,10]}

kds = {"immediate": [5,5],
       "1second": [5,5],
       "5seconds": [5,5]}
# LISTS USED FOR TUNNING THE PID PARAMETERS
scenario_not_considered = [0,1]
alpha_not_considered = []


# parameters for the Crank-Nicolson solution to heat diffusion
T_initial = 288                      # K
T_air = T_initial                    # K
time_total = 911                     # s
sample_length = 0.025                # m
space_divisions = 100                # -
h = 45                               # W/m2K for linearised surface bc with constant heat transfer coefficient
hc = 10                              # W/m2K convective heat transfer coefficient
emissivity = 1                       # -
sigma = 5.67e-8                      # W/m2K4

# parameters for the numerical experiment
surface_temperature_setpoint = 300 + T_initial   # K

# material properties
ks = np.array([0.2,0.5])            # W/mK
alphas = np.array([1e-7, 1.0e-6])   # J/kgK

# create spatial mesh
dx = sample_length/(space_divisions - 1)
x_grid = np.array([i * dx for i in range(space_divisions)])

# define time step based on the spatial mesh for each alpha and create mesh
dts = (1 / 3) * (dx**2 / alphas)
time_divisions = time_total / dts
t_grids = []
for i,dt in enumerate(dts):
    t_grids.append(np.array([n * dt for n in range(int(time_divisions[i]))]))
    del i,dt
upsilons = (alphas*dts)/(2*dx**2)

# define the incident heat flux as an array the size of each temporal grid
q_arrays = []
for t_grid in t_grids:
    q = np.zeros_like(t_grid)
    q_arrays.append(q)
    del q, t_grid

# create list with type of surface boundary conditions to be analysed
boundary_conditions_surface = ["Linear", "Non-linear"]

# create dictionary to store all the data
all_data = {}

time_lag_scenarios = ["immediate", "1second", "5seconds"]

for scenario_number, scenario in enumerate(time_lag_scenarios):
    
    # DEBUGGING
    if scenario_number in scenario_not_considered:
        continue
    
    print(f"Calculating for {scenario} correction")
    kp = kps[scenario]
    kd = kds[scenario]
    ki = kis[scenario]
    
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
    
        print(f" calculating for {bc_surface} boundary condition")
        all_data[bc_surface] = {}

        # store the different data with different keys in the general dictionary
        all_data[bc_surface]["temperature_profile"] = {}
        all_data[bc_surface]["surface_temperature"] = {}
        all_data[bc_surface]["incident_heatflux"] = {}
        all_data[bc_surface]["net_heatflux"] = {}
        all_data[bc_surface]["error"] = {}
    
        # iterate over 4 different values of alpha and k
        for alpha_number in range(len(alphas)):
            
            # DEBUGGING
            if alpha_number in alpha_not_considered:
                continue
            
            # define initial temperatures array
            T = np.zeros_like(x_grid) + T_initial
        
            # extend dictionary for storing the temperature profile at every time step
            all_data[bc_surface]["temperature_profile"][alphas[alpha_number]] = {}
            all_data[bc_surface]["surface_temperature"][alphas[alpha_number]] = {}
            all_data[bc_surface]["net_heatflux"][alphas[alpha_number]] = {}
            all_data[bc_surface]["incident_heatflux"][alphas[alpha_number]] = {}
            all_data[bc_surface]["error"][alphas[alpha_number]] = {}
    
            # -- Define variables that are used by the PID controller
            error_sum = 0
            # nhf array and surface_temperature_array save the actual nhf at every time step
            nhf_array = np.zeros_like(t_grids[alpha_number])
            surface_temperature_array = np.zeros_like(t_grids[alpha_number])
            # last_surface_temperature saves the nhf at a frequency that corresponds to the time lag
            last_surface_temperature = T_initial
            # error array saves the actual array at every time step
            error_array = np.zeros_like(t_grids[alpha_number])
            # last_error saves the error at a frequency that corresponds to the time lag
            last_error = 0
            # last_error saves the error at a frequency that corresponds to the time log
            last_time = 0
            # create a counter to determine when the PID controller needs to be called
            time_lag_counter = 0
            
            
            # since operations are defined 2*time lags in advace, only iterate over t_grid from 0 till end - 2*time_lag
            if scenario_number == 0:
                maximum_limit_tgrid = 2
            else:
                maximum_limit_tgrid = 2*number_timereadings[alpha_number]
                
            # step in time
            for time_step,t in enumerate(t_grids[alpha_number][:-maximum_limit_tgrid]):
                # q arrays starts at a value of 10 kW/m2 for the first time lag readings
                q_arrays[alpha_number][0:maximum_limit_tgrid] = 10000
                
                # first, calculate the new temperature for the next time step (Crank-Nicolson)
                temperature_profile, surface_temperature, nhf = general_temperatures(
                                                                T_initial, T_air, time_total, ks[alpha_number], alphas[alpha_number], 
                                                                dx, x_grid,space_divisions, upsilons[alpha_number], bc_surface,
                                                                q_arrays[alpha_number], h, hc, emissivity, sigma, time_step, T)
                
                # update temperature array, net heat flux and surface temperature
                T = temperature_profile.copy()
                surface_temperature_array[time_step] = surface_temperature
                nhf_array[time_step] = nhf
                error_array[time_step] = surface_temperature_setpoint - surface_temperature
                timeChange = t - last_time
                if timeChange == 0:
                    timeChange = 1e-6
                
                # call the PID only once every time lag
                if time_lag_counter < number_timereadings[alpha_number]:
                    time_lag_counter += 1
                else:
                    # call PID
                    new_q, new_error, error_sum = PID(surface_temperature, surface_temperature_setpoint, 
                                                      last_error, last_surface_temperature, 
                                                          error_sum, timeChange,kp[alpha_number], 
                                                          ki[alpha_number], kd[alpha_number])
                    
                    # update parameters
                    q_arrays[alpha_number][time_step+1:time_step+maximum_limit_tgrid] = new_q
                    
                    # ---- PID debugging
#                    print("\n")
#                    print("PID called")
#                    print(f"Scenario: {scenario}")
#                    print(f"alpha: {alphas[alpha_number]}")      
#                    print(f"Time step:  {time_step+1}/{len(t_grids[alpha_number])}")
#                    print(f"Time:  {t}")
#                    print(f"Set Surface Temperature: {surface_temperature_setpoint}")
#                    print("-----")
#                    print(f"Current IHF: {q_arrays[alpha_number][time_step]}")
#                    print(f"Next IHF:    {q_arrays[alpha_number][time_step + 1]}")            
#                    print(f"Previous surface temperature:{last_surface_temperature}")
#                    print(f"Current surface temperature: {surface_temperature}")
#                    print(f"Error sum: {error_sum}")
#                    print(f"Current error: {error_array[time_step]}")
#                    time.sleep(0.5)
                    # ----
                    
                    last_surface_temperature = surface_temperature
                    last_error = new_error
                    last_time = t
                    
                    # update the counter
                    time_lag_counter = 0
                        
                # update at every time step the temperature profile
                all_data[bc_surface]["temperature_profile"][alphas[alpha_number]][t] = temperature_profile
            
                # update 
                all_data[bc_surface]["surface_temperature"][alphas[alpha_number]][t] = surface_temperature
                all_data[bc_surface]["net_heatflux"][alphas[alpha_number]][t] = nhf
                all_data[bc_surface]["incident_heatflux"][alphas[alpha_number]][t] = q_arrays[alpha_number][time_step]
                all_data[bc_surface]["error"][alphas[alpha_number]][t] = error_array[time_step]
                    
    
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
    with open(f"animations/total_data_backinsulated_{scenario}_correction_1Hz.pickle", 'wb') as handle:
        pickle.dump(total_data_1Hz, handle)
        print(f"  total_data_backinsulated_{scenario}_correction_1Hz.pickle")
        
    print(f"Time taken for {scenario} correction: {np.round(time.time() - start,2)}")






