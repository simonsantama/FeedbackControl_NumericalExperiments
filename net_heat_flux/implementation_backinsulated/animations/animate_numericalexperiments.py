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
line_color = ["firebrick", "royalblue", "seagreen", "black"] # "blueviolet"]
line_width = 1.75
line_width_inset = 1.25
line_color_q = "dimgrey"
x_limits = [-90,900]
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
#        for i, ax in enumerate(axis.flatten()):
#            ax.grid(color = "gainsboro", linestyle = "--", linewidth = 0.75)
#        axis[0,0].set_title("Incident Heat flux", fontsize = fs_subplot_title)
#        axis[0,1].set_title("Net Heat Flux", fontsize = fs_subplot_title)
#        axis[1,0].set_title("Surface temperature", fontsize = fs_subplot_title)
#        axis[1,1].set_title("Temperature profile", fontsize = fs_subplot_title)
#        
#        for ax in [axis[0,0], axis[0,1], axis[1,0]]:
#            ax.set_xlabel("Time [$s$]", fontsize = fs_axis_labels)
#            ax.set_xlim(-(time_total - 1)*0.1, time_total-1)
#            ax.set_xticks(np.linspace(0,time_total-1,7))
#        axis[1,1].set_xlim(-x_grid[-1]*100, x_grid[-1]*1000)
#        axis[1,1].set_xlabel("Sample depth [$mm$]", fontsize = fs_axis_labels)
#        
#        for ax in axis.flatten()[:2]:
#            ax.set_ylabel("Heat Flux [$kW/m^2$]", fontsize = fs_axis_labels)
#        axis[0,0].set_ylim([-8, 80])
#        axis[0,0].set_yticks(np.linspace(0,80,5))
#        axis[0,1].set_ylim([-2, 20])
#        axis[0,1].set_yticks(np.linspace(0,20,5))
#        for ax in axis.flatten()[-2:]:
#            ax.set_ylabel("Temperature [$^\circ$C]", fontsize = fs_axis_labels)
#            ax.set_ylim([-80,800])
#            ax.set_yticks(np.linspace(0,800,5))
#
#        # add inset to first subplot to show evolution of the heat flux
#        ax_inset = inset_axes(axis[0,0], width = "35%", height = "45%", borderpad = 0.75, loc = "upper left")
#        ax_inset.set_xlim([0,600])
#        ax_inset.set_xticks(np.linspace(0,600,7))
#        ax_inset.set_xticklabels([""]*7)
#        ax_inset.yaxis.set_label_position("right")
#        ax_inset.yaxis.tick_right()
#        ax_inset.set_ylabel("error [kW/m$^2$]", fontsize = 10)
#        ax_inset.set_ylim([0,10])
#        ax_inset.set_yticks(np.linspace(0,10,3))
#        ax_inset.grid(color = "gainsboro", linestyle = "--", linewidth = 0.5)  
#
#
#
#        # add text with counter to the last subplot
#        counter = axis[0,1].text(300,17, "Time: 0 seconds", fontsize = text_fs,
#                      bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5'))
#        
#        # add custom legends and additional information to the plots
#        custom_lines = []
#        for z, alpha_value in enumerate(alphas[0::3]):
#            generic_line = Line2D([0],[0], color = line_color[z], lw = line_width, linestyle = line_styles[z])
#            custom_lines.append(generic_line)
#        axis[1,1].legend(custom_lines, alphas[0::3], fancybox = True, 
#                            title = r"$\alpha$ [$m^2/s$]", 
#                            loc = "upper right", ncol = 1) 
#     
#        # add legend to thrid plot showing the different values of k
#        custom_lines = []
#        for z, alpha_value in enumerate(ks[0::3]):
#            generic_line = Line2D([0],[0], color = line_color[z], lw = line_width, linestyle = line_styles[z])
#            custom_lines.append(generic_line)
#        axis[1,0].legend(custom_lines, ks[0::3], fancybox = True, 
#                            title = r"$k$ [$W/mK$]", 
#                            loc = "upper left", ncol = 1) 
#            
#    
#        # create list to store every plotting line
#        all_lines = []
#        for subplot_number in range(4):
#            for alpha_number, alpha in enumerate(alphas[0::3]):
#                line, = axis.flatten()[subplot_number].plot([],[], linewidth = line_width, color = line_color[alpha_number],
#                                    linestyle = line_styles[alpha_number])
#                all_lines.append(line)
#                
#        # add lines in the inset that correspond to the error (nhf-set - nhf-actual)
#        for alpha_number, alpha in enumerate(alphas[0::3]):
#            line, = ax_inset.plot([],[], linewidth = line_width_inset, color = line_color[alpha_number], linestyle = line_styles[z])
#            all_lines.append(line)
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
#        error = {}
#        
#        # this is for easier plotting (better to have an array of data in time than a dictionary with time steps being each key)
#        for alpha_number in alphas[0::3]:
#            # initialize for each alpha value
#            incident_heatflux[alpha_number] = np.zeros_like(t_grid_plotting)
#            net_heatflux[alpha_number] = np.zeros_like(t_grid_plotting)
#            surface_temperature[alpha_number] = np.zeros_like(t_grid_plotting)
#            error[alpha_number] = np.zeros_like(t_grid_plotting)
#            
#            # extract data from dict and store in arrays
#            for key in all_data[bc_surface]["incident_heatflux"][alpha_number]:
#                incident_heatflux[alpha_number][key] = all_data[bc_surface]["incident_heatflux"][alpha_number][key]
#                net_heatflux[alpha_number][key] = all_data[bc_surface]["net_heatflux"][alpha_number][key]
#                surface_temperature[alpha_number][key] = all_data[bc_surface]["surface_temperature"][alpha_number][key]
#                error[alpha_number][key] = all_data[bc_surface]["error"][alpha_number][key]
#        
#        # define animate function
#        def animate(k):
#            
#            # update counter
#            counter.set_text(f"Time: {k} seconds")
#            
#            # update lines
#            for l,line in enumerate(all_lines):
#                # first two lines plot the incident heat flux
#                if l<2:
#                    x = t_grid_plotting[:k]
#                    y = incident_heatflux[alphas[0::3][l]][:k]/1000
#                    line.set_data(x,y)
#                    
#                # second set of two lines plot the net heat flux
#                elif 2 <= l < 4:
#                    x = t_grid_plotting[:k]
#                    y = net_heatflux[alphas[0::3][l-2]][:k]/1000
#                    line.set_data(x,y)
#                
#                # third set of two lines plot the surface temperature
#                elif 4 <= l < 6:
#                    x = t_grid_plotting[:k]
#                    y = surface_temperature[alphas[0::3][l-4]][:k] - 288
#                    line.set_data(x,y)
#                
#                # fourth set of two lines plot the temperature profile
#                elif 6 <= l < 8:
#                    x = x_grid*1000
#                    # extract temperature profile for this alpha and this time step
#                    y = all_data[bc_surface]["temperature_profile"][alphas[0::3][l-6]][k] - 288
#                    line.set_data(x,y)
#                    
#                # last set of two lines plot the error in the inset
#                if 8 <= l < 10:
#                    x = t_grid_plotting[:k]
#                    y = error[alphas[0::3][l-8]][:k]/1000
#                    line.set_data(x,y)
#            
#            return all_lines
#            
#        # create animations
#        anim = FuncAnimation(fig, animate, init_func=init,
#                                       frames=time_total-11, 
#                                       interval=20, blit=True)
#        
#        # save animation in the corresponding folder
#        file_name_animation = f"./animations/{bc_surface}surface_backinsulated_{scenario}correction"
#        anim.save(f'{file_name_animation}.mp4', dpi = 300, fps = 30)
#        plt.close()
#            
#    
#        print(f"  - time taken: {np.round(time.time() - start2,2)} seconds")
#    print(f"  Total time taken for {scenario} correction: {np.round(time.time() - start,2)} seconds")
#            