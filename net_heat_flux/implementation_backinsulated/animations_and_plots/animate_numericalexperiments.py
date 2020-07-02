"""
This script creates animations to show the evolution of the numerical experiments.
Feedback control of the NHF using a Crank-Nicolson solver to determine the temperature profile.

Must run main.py before running this script
"""

# import libraries
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D
import pickle
import numpy as np
import time

# plotting parameters
figure_size = (12,6)
fs_figure_title = 15
fs_subplot_title = 13
fs_axis_labels = 12
text_fs = 11

line_styles = ["-", "--", ":", "-."]
line_color = ["black", "royalblue"] #"royalblue", "seagreen", "black","blueviolet", "firebrick"
line_width = 1.5
line_width_setpoint = 1
line_color_q = "dimgrey"
x_limits = [-45,900]
t_grid_plotting = np.linspace(0,900,901)

time_lag_scenarios = ["immediate", "1second", "5seconds"]
for scenario_number, scenario in enumerate(time_lag_scenarios):
    
    # DEBUGGING
    if scenario_number in []:
        continue
    
    start = time.time()
    print(f"{scenario_number}.Creating animation for {scenario} correction")
    
    with open(f"total_data_backinsulated_{scenario}_correction_1Hz.pickle", "rb") as handle:
        total_data_1Hz = pickle.load(handle)

    
    # extract data from the pickle
    all_data = total_data_1Hz["all_data"]
    alphas = total_data_1Hz["extra_data"]["alpha"]
    ks = total_data_1Hz["extra_data"]["k"]
    time_total = total_data_1Hz["extra_data"]["time_total"]
    x_grid = total_data_1Hz["extra_data"]["x_grid"]
    
    # iterate over both surface boundary conditions
    for bc_surface in all_data:
        start2 = time.time()
        print(f"  annimation for {bc_surface} boundary condition")

        # define the title for the figure depending on the surface boundary condition
        if bc_surface == "Linear":
            figure_title = "Linear BC: $\dot{q}''_{net} = \dot{q}''_{inc} - h_T(T_{surf} - T_{\infty})$\n" + \
                            "Insulated back surface"
        elif bc_surface == "Non-linear":
            figure_title = "Non-Linear BC: $\dot{q}''_{net} = \dot{q}''_{inc} - h_c(T_{surf} - T_{\infty}) - \epsilon \sigma T_{surf}^4$\n" + \
                            "Insulated back surface"
        figure_title = figure_title + f"\n {scenario} response"
        
        # formatting the plot
        fig, axis = plt.subplots(2,2, constrained_layout = True, figsize = figure_size)
        fig.suptitle(figure_title, fontsize = fs_figure_title)
        for i, ax in enumerate(axis.flatten()):
            ax.grid(color = "gainsboro", linestyle = "--", linewidth = 0.75)
            ax.set_xlabel("Time [$s$]", fontsize = fs_axis_labels)
            ax.set_xlim(x_limits)
            ax.set_xticks(np.linspace(0,x_limits[1],10))
        axis[0,0].set_title("Incident Heat flux", fontsize = fs_subplot_title)
        axis[0,1].set_title("Net Heat Flux", fontsize = fs_subplot_title)
        axis[1,0].set_title("Surface temperature", fontsize = fs_subplot_title)
        axis[1,1].set_title("Error", fontsize = fs_subplot_title)
        for ax in axis.flatten()[:2]:
            ax.set_ylabel("Heat Flux [$kW/m^2$]", fontsize = fs_axis_labels)
        axis[0,0].set_ylim([-6, 40])
        axis[0,0].set_yticks(np.linspace(0,60,5))
        axis[0,1].set_ylim([-1, 10])
        axis[0,1].set_yticks(np.linspace(0,10,5))
        axis[1,0].set_ylabel("Temperature [$^\circ$C]", fontsize = fs_axis_labels)
        axis[1,0].set_ylim([-80,800])
        axis[1,0].set_yticks(np.linspace(0,800,5))
        axis[1,1].set_ylabel("Error [$\%$]")
        axis[1,1].set_ylim([-40,120])
        axis[1,1].set_yticks(np.linspace(-40,120,5))
        
        # plot a line showing the desired NHF value
        axis[0,1].plot([-45,900], [5.0,5.0], linewidth = line_width_setpoint, color = "firebrick", linestyle = "-.")
        
        # add legend to the error plot to show the value of alpha and k
        custom_lines = []
        for alpha_number, alpha in enumerate(alphas):
            generic_line = Line2D([0],[0], color = line_color[alpha_number], lw = line_width, 
                                  linestyle = line_styles[alpha_number])
            custom_lines.append(generic_line)
        axis[1,1].legend(custom_lines, list(zip(alphas,ks)), fancybox = True, loc = "upper right",
            title = r"$\alpha$ [$m^2/s$], $k[W/mK]$")

        # create list to store every plotting line
        all_lines = []
        for ax in axis.flatten():
            for alpha_number, alpha in enumerate(alphas):
                line, = ax.plot([],[],linewidth = line_width, color = line_color[alpha_number], 
                                linestyle = line_styles[alpha_number])
                all_lines.append(line)
        
        # define init function for animation
        def init():
            for line in all_lines:
                line.set_data([],[])
            return all_lines
        
        # define arrays for ease of plotting (data is in dictionary form)
        incident_heatflux = {}
        net_heatflux = {}
        surface_temperature = {}
        error = {}
        
        for alpha_number in alphas:
            # initialize for each alpha value
            incident_heatflux[alpha_number] = np.zeros_like(t_grid_plotting)
            net_heatflux[alpha_number] = np.zeros_like(t_grid_plotting)
            surface_temperature[alpha_number] = np.zeros_like(t_grid_plotting)
            error[alpha_number] = np.zeros_like(t_grid_plotting)
            
            # extract data from dict and store in arrays
            for key in all_data[bc_surface]["incident_heatflux"][alpha_number]:
                if key > 900:
                    break
                incident_heatflux[alpha_number][key] = all_data[bc_surface]["incident_heatflux"][alpha_number][key]
                net_heatflux[alpha_number][key] = all_data[bc_surface]["net_heatflux"][alpha_number][key]
                surface_temperature[alpha_number][key] = all_data[bc_surface]["surface_temperature"][alpha_number][key]
                error[alpha_number][key] = all_data[bc_surface]["error"][alpha_number][key]
        
        # define animate function
        def animate(k):

            for l,line in enumerate(all_lines):
                # first two lines plot the incident heat flux
                if l<2:
                    x = t_grid_plotting[:k]
                    y = incident_heatflux[alphas[l]][:k]/1000
                    line.set_data(x,y)
                    
                # second set of two lines plot the net heat flux
                elif 2 <= l < 4:
                    x = t_grid_plotting[:k]
                    y = net_heatflux[alphas[l-2]][:k]/1000
                    line.set_data(x,y)
                
                # third set of two lines plot the surface temperature
                elif 4 <= l < 6:
                    x = t_grid_plotting[:k]
                    y = surface_temperature[alphas[l-4]][:k] - 288
                    line.set_data(x,y)
                
                # fourth set of two lines plot the temperature profile
                elif 6 <= l < 8:
                    x = t_grid_plotting[:k]
                    # extract temperature profile for this alpha and this time step
                    y = error[alphas[l-6]][:k]*100/5000
                    line.set_data(x,y)

            return all_lines
            
        # create animations
        anim = FuncAnimation(fig, animate, init_func=init,
                                       frames=900, 
                                       interval=20, blit=True)
        
        # save animation in the corresponding folder
        file_name_animation = f"{bc_surface}surface_backinsulated_{scenario}correction"
        anim.save(f'{file_name_animation}.mp4', dpi = 300, fps = 30)
        plt.close()
            
    print(f"  total time for {scenario} correction: {np.round(time.time() - start,2)} seconds")
            
