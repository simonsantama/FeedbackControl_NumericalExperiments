"""
This script creates animations to show the evolution of the numerical experiments.
Feedback control of the NHF using a Crank-Nicolson solver to determine the temperature profile.

Must run main.py before running this script

"""

# import libraries
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import pickle
import numpy as np
import time

# plotting parameters
figure_size = (8,6)
fs_figure_title = 15
fs_subplot_title = 13
fs_axis_labels = 12
text_fs = 11

line_styles = ["-", "--", ":", "-."]
line_color = ["firebrick", "royalblue", "seagreen", "black"] # "blueviolet"]
line_width = 1.75
line_width_inset = 1.25
line_color_q = "dimgrey"


time_lag_scenarios = ["immediate", "1second", "5seconds"]
for scenario_number, scenario in enumerate(time_lag_scenarios):
    start = time.time()
    
    # DEBUGGING
    if scenario in []:
        continue
    
    with open(f"total_data_backinsulated_{scenario}_correction_1Hz.pickle", "rb") as handle:
        total_data_1Hz = pickle.load(handle)
    
    print("------")
    print(f"Creating animation for {scenario} correction")
    
    # extract data from the pickle
    all_data = total_data_1Hz["all_data"]
    alphas = total_data_1Hz["extra_data"]["alpha"]
    ks = total_data_1Hz["extra_data"]["k"]
    time_total = total_data_1Hz["extra_data"]["time_total"]
    x_grid = total_data_1Hz["extra_data"]["x_grid"]
    
    # additional plotting parameters which depend on the actual experiments
    x_limits = [-30,time_total-1]
    t_grid_plotting = np.linspace(0,time_total-1, time_total)

    # iterate over both surface boundary conditions
    for bc_surface in all_data:
        print(f"  creating animations for {bc_surface} boundary conditions")

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
        axis[0,0].set_title("Heat flux", fontsize = fs_subplot_title)
        axis[0,1].set_title("Error:  NHF - NHFdesired", fontsize = fs_subplot_title)
        axis[1,0].set_title("Surface temperature", fontsize = fs_subplot_title)
        axis[1,1].set_title("Temperature profile", fontsize = fs_subplot_title)
        
        for ax in [axis[0,0], axis[0,1], axis[1,0]]:
            ax.set_xlabel("Time [$s$]", fontsize = fs_axis_labels)
            ax.set_xlim(-(time_total - 1)*0.1, time_total-1)
            ax.set_xticks(np.linspace(0,time_total-1,7))
        axis[1,1].set_xlim(-x_grid[-1]*100, x_grid[-1]*1000)
        axis[1,1].set_xlabel("Sample depth [$mm$]", fontsize = fs_axis_labels)
        
        for ax in axis.flatten()[:2]:
            ax.set_ylabel("Heat Flux [$kW/m^2$]", fontsize = fs_axis_labels)
        axis[0,0].set_ylim([-6, 60])
        axis[0,0].set_yticks(np.linspace(0,60,7))
        axis[0,1].set_ylim([-1.2, 12])
        axis[0,1].set_yticks(np.linspace(0,12,7))
        for ax in axis.flatten()[-2:]:
            ax.set_ylabel("Temperature [$^\circ$C]", fontsize = fs_axis_labels)
            ax.set_ylim([-80,800])
            ax.set_yticks(np.linspace(0,800,5))
    
        # add text with counter to the last subplot
        counter = axis[0,1].text(300,10, "Time: 0 seconds", fontsize = text_fs,
                      bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5'))
        
        
        # add custom legends and additional information to the plots
        custom_lines = []
        for z, alpha_value in enumerate(alphas[0::3]):
            generic_line = Line2D([0],[0], color = line_color[z], lw = line_width, linestyle = line_styles[0])
            custom_lines.append(generic_line)
        axis[0,0].legend(custom_lines, alphas[0::3], fancybox = True, 
                            title = r"$\alpha$ [$m^2/s$]", 
                            loc = "upper left", ncol = 1) 
     
        # add legend to thrid plot showing the different values of k
        custom_lines = []
        for z, alpha_value in enumerate(ks[0::3]):
            generic_line = Line2D([0],[0], color = line_color[z], lw = line_width, linestyle = line_styles[0])
            custom_lines.append(generic_line)
        axis[1,0].legend(custom_lines, ks[0::3], fancybox = True, 
                            title = r"$k$ [$W/mK$]", 
                            loc = "upper left", ncol = 1) 
#            
#    
#        # create list to store every plotting line
#        all_lines = []
#        for subplot_number in range(4):
#            for alpha_number, alpha in enumerate(alphas):
#                line, = axis.flatten()[subplot_number].plot([],[], linewidth = line_width, color = line_color[alpha_number],
#                                    linestyle = line_styles[alpha_number])
#                all_lines.append(line)
#                
#        # define init function for animation
#        def init():
#            for line in all_lines:
#                line.set_data([],[])
#            counter.set_text("Time: 0 seconds")
#            return all_lines
#        
#        # define arrays for ease of plotting incident heat flux, net heat flux and surface temperature
#        incident_heatflux = {}
#        net_heatflux = {}
#        surface_temperature = {}
#        
#        for alpha_number in alphas:
#            incident_heatflux[alpha_number] = np.zeros_like(t_grid_plotting)
#            net_heatflux[alpha_number] = np.zeros_like(t_grid_plotting)
#            surface_temperature[alpha_number] = np.zeros_like(t_grid_plotting)
#            for key in all_data[bc_surface]["incident_heatflux"][alpha_number]:
#                incident_heatflux[alpha_number][key] = all_data[bc_surface]["incident_heatflux"][alpha_number][key]
#                net_heatflux[alpha_number][key] = all_data[bc_surface]["net_heatflux"][alpha_number][key]
#                surface_temperature[alpha_number][key] = all_data[bc_surface]["surface_temperature"][alpha_number][key]
#        
#        # define animate function
#        def animate(k):
#            
#            # update counter
#            counter.set_text(f"Time: {k} seconds")
#            
#            # update lines
#            for l,line in enumerate(all_lines):
#                # first four lines plot the incident heat flux
#                if l<4:
#                    x = t_grid_plotting[:k]
#                    y = incident_heatflux[alphas[l]][:k]/1000
#                    line.set_data(x,y)
#                    
#                # second set of four plot the net heat flux
#                elif 4 <= l < 8:
#                    x = t_grid_plotting[:k]
#                    y = net_heatflux[alphas[l-4]][:k]/1000
#                    line.set_data(x,y)
#                
#                # third set plots the surface temperature evolution
#                elif 8 <= l < 12:
#                    x = t_grid_plotting[:k]
#                    y = surface_temperature[alphas[l-8]][:k] - 288
#                    line.set_data(x,y)
#                
#                # fourth set plots the surface temperature evolution
#                elif 12 <= l < 16:
#                    x = x_grid*1000
#                    # extract temperature profile for this alpha and this time step
#                    y = all_data[bc_surface]["temperature_profile"][alphas[l-12]][k] - 288
#                    line.set_data(x,y)
#            
#            return all_lines
#            
#        # create animations
#        anim = FuncAnimation(fig, animate, init_func=init,
#                                       frames=time_total-1, 
#                                       interval=20, blit=True)
#        
#        # save animation in the corresponding folder
#        file_name_animation = f"./animations/{bc_surface}_surface-{correction_type.split('_')[3]}"
#        anim.save(f'{file_name_animation}.mp4', dpi = 300, fps = 30)
#        plt.close()
#            
#    
#        print(f" time taken for {bc_surface} surface -- {correction_type.split('_')[3]}\n:" + \
#              f"{np.round(time.time() - start,2)} seconds")
#            
