"""
This script takes the temperature data calculated on different time grids (as delta t depends on the thermal diffusivity)
and it outputs a similar dictionary but with temperature data at 1 Hz for all conditions.
"""

# import libraries
import time
import pickle
import numpy as np

########
# -- First set of experiments can immediatly correct the Incident Heat Flux
########

# import data created in main_validation.py
with open('total_data_backinsulated_immediatecorrection.pickle', 'rb') as handle:
    total_data = pickle.load(handle)

# extract all the data from the pickle file
all_data = total_data["all_data"]


x_grid = total_data["extra_data"]["x_grid"]
time_total = total_data["extra_data"]["time_total"]
alphas = total_data["extra_data"]["alpha"]
ks = total_data["extra_data"]["k"]

# create dictionary to store 1 Hz data for plots and animations
all_data_1Hz = {}

start = time.time()
for level1_bcsurface in all_data:
    
    all_data_1Hz[level1_bcsurface] = {}
    for level2_datatype in all_data[level1_bcsurface]:
        all_data_1Hz[level1_bcsurface][level2_datatype] = {}
        
        for level3_alphavalue in all_data[level1_bcsurface][level2_datatype]:
            all_data_1Hz[level1_bcsurface][level2_datatype][level3_alphavalue] = {}

            # order time stamps since dictionary keys are not ordered
            ordered_timestamps = [x for x in all_data[level1_bcsurface][level2_datatype][level3_alphavalue].keys()]
            ordered_timestamps.sort()
            
            # takes value of time stamp that is closest (but larger) than the integer time
            time_int = 0
            for t_number, time_stamp in enumerate(ordered_timestamps):
                
                # time 0
                if t_number == 0:
                    pass
                    all_data_1Hz[level1_bcsurface][level2_datatype][level3_alphavalue][time_int] = \
                    all_data[level1_bcsurface][level2_datatype][level3_alphavalue][time_stamp]
                  
                # other times
                else:
                    pass
                    time_int_new = int(time_stamp)
                    if time_int_new > time_int:
                        all_data_1Hz[level1_bcsurface][level2_datatype][level3_alphavalue][time_int] = \
                        all_data[level1_bcsurface][level2_datatype][level3_alphavalue][time_stamp]

                        time_int = time_int_new
                        
                    else:
                        pass
                    
print(f"Time taken for reducing the temperature data to 1 Hz: {np.round(time.time() - start,2)} seconds")
  
# condense all data to be saved including important plotting parameters
total_data_1Hz = {"all_data": all_data_1Hz, #  includes temperature profile, surface temperature, incident HF and net HF. 
                  "extra_data": { 
                          "x_grid":x_grid, 
                          "time_total": time_total, 
                          "alpha": alphas, 
                          "k": ks,},
    }

# save in a pickle to retrieve and plot later
with open('total_data_backinsulated_immediatecorrection_1Hz.pickle', 'wb') as handle:
    pickle.dump(total_data_1Hz, handle)
    print("\n")
    print("All data saved into total_data_backinsulated_immediatecorrection_1Hz.pickle")

 