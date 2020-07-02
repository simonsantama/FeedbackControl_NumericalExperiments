"""
This function takes the temperature data calculated on different time grids (as delta t depends on the thermal diffusivity)
and it outputs a similar dictionary but with temperature data at 1 Hz for all conditions.

This is necessary to be able to animate the evolution of the desired quantities for all conditions at the same time

"""

def convert_1Hz(total_data, time_lag_scenario):
    """
    Parameters:
    ----------
    total_data: dictionary that includes all the calculated as well as temporal and spatial grids. 
        dict
        
    time_lag_scenario: current time lag scenario
        list
    
    Returns:
    -------
    total_data_1Hz: similar to total data but the data is now presented at a logging frequency of 1Hz
        dict
    """

    # extract the data from the all data dictionary
    all_data = total_data["all_data"]
    x_grid = total_data["extra_data"]["x_grid"]
    time_total = total_data["extra_data"]["time_total"]
    alphas = total_data["extra_data"]["alpha"]
    ks = total_data["extra_data"]["k"]
    
    # create dictionary to store 1 Hz data for plots and animations
    all_data_1Hz = {}
    
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
                        
      
    # condense all data to be saved including important plotting parameters
    total_data_1Hz = {"all_data": all_data_1Hz, #  includes temperature profile, surface temperature, incident HF and net HF. 
                      "extra_data": { 
                              "x_grid":x_grid, 
                              "time_total": time_total, 
                              "alpha": alphas, 
                              "k": ks,},
        }

    return total_data_1Hz

 