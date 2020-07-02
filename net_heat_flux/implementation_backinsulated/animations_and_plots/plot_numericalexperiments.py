"""
This script creates a function to plot the evolution of the numerical experiments.
Feedback control of the NHF using a Crank-Nicolson solver to determine the temperature profile.

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

def plot_numericalexperiments(total_data_1Hz, scenario, scenario_number):
    """
    Creates plot for both boundary conditions for a given correction scenario.
    
    Parameters:
    ----------
    total_data_1Hz: dictionary of data with all the calculated values from main.py
        dict
        
    scenario: scenario that reflects the correction delay time in the feedback control loop
        string
        
    scenario_number: number of the iteration over the different scenarios
        int
        
    Returns:
    -------
    none
    
    """

    
    # extract data from the dictionary
    all_data = total_data_1Hz["all_data"]
    alphas = total_data_1Hz["extra_data"]["alpha"]
    ks = total_data_1Hz["extra_data"]["k"]
    
    # iterate over both surface boundary conditions
    for bc_surface in all_data:

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
        
        # define arrays for ease of plotting (data is in dictionary form)
        incident_heatflux = {}
        net_heatflux = {}
        surface_temperature = {}
        error = {}
        
        for alpha in alphas:
            # initialize for each alpha value
            incident_heatflux[alpha] = np.zeros_like(t_grid_plotting)
            net_heatflux[alpha] = np.zeros_like(t_grid_plotting)
            surface_temperature[alpha] = np.zeros_like(t_grid_plotting)
            error[alpha] = np.zeros_like(t_grid_plotting)
            
            # extract data from dict and store in arrays
            for key in all_data[bc_surface]["incident_heatflux"][alpha]:
                if key > 900:
                    break
                incident_heatflux[alpha][key] = all_data[bc_surface]["incident_heatflux"][alpha][key]
                net_heatflux[alpha][key] = all_data[bc_surface]["net_heatflux"][alpha][key]
                surface_temperature[alpha][key] = all_data[bc_surface]["surface_temperature"][alpha][key]
                error[alpha][key] = all_data[bc_surface]["error"][alpha][key]
        
        data_plotting = [incident_heatflux, net_heatflux, surface_temperature, error]
        
        # plot lines
        for ax_number, ax in enumerate(axis.flatten()):
            for alpha_number, alpha in enumerate(alphas):
                ax.plot(t_grid_plotting,data_plotting[alpha], linewidth = line_width, color = line_color[alpha_number],
                        linestyle = line_styles[alpha_number])
      
        # save plot
        file_name_plot = f"animations_and_plots/{scenario}correction_{bc_surface}surface_backinsulated.png"
        plt.savefig(file_name_plot, dpi = 300)
        plt.close()      
