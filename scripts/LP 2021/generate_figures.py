import matplotlib.pyplot as plt
import covbayesvar.large_bvar as bvar
from matplotlib.lines import Line2D
import pickle
import numpy as np

# Define the results folder
results_folder = 'results'
plot_impulse_responses = True
plot_conditional_forecasts_fig3 = True
plot_conditional_forecasts_fig4 = True
plt.ion()  # Turn on interactive mode to view the matplotlib figures while debugging

# Function to load pickle files
def load_results(filename, results_folder='results'):
    with open(f"{results_folder}/{filename}", "rb") as f:
        return pickle.load(f)

# Load datasets
may2021_vintage_date = 'May2021'
june2020_vintage_date = 'Jun2020'
cv_feb2020_may2021 = load_results(f"CVFeb2020_{may2021_vintage_date}.pkl")
cv_may2021 = load_results(f"CV_{may2021_vintage_date}.pkl")
baseline_may2021 = load_results(f"Baseline_{may2021_vintage_date}.pkl")
baseline_june2020 = load_results(f"Baseline_{june2020_vintage_date}.pkl")
cv_feb2020_june2020 = load_results(f"CV_Feb2020_{june2020_vintage_date}.pkl")

# Define color schemes
ColorBase = (44 / 255, 127 / 255, 184 / 255)
ColorGrey = (0.5, 0.5, 0.5)
ColorCovid = (0.8941, 0.1020, 0.1098)
red = np.array([200, 30, 30]) / 255  # Red color for conditional
blue = (55 / 255, 126 / 255, 184 / 255)

qqq = [0.025 ,0.16, 0.5, 0.84,0.975]
series = ['Unemployment Rate', 'Employment', 'PCE', 'PCE: Services', 'PCE (Price)', 'PCE: Services (Price)',
          'Core PCE (Price)']
YLABELirf = ["percentage points", "100 x log points", "100 x log points", "100 x log points", "100 x log points",
             "100 x log points", "100 x log points"]
YLABELfcst = ["percentage points", "index", "index", "index", "index", "index", "index"]

# Define time range for x-axis
x_dates = np.arange(2019 + 0.5 / 12, 2022 + 5.5 / 12 + 1 / 12, 1 / 12)
x_ticks = [2019 + 0.5 / 12, 2020 + 0.5 / 12, 2021 + 0.5 / 12, 2022 + 0.5 / 12]
x_tick_labels = ['Jan 2019', 'Jan 2020', 'Jan 2021', 'Jan 2022']
start_end_dates_xlim = [2019 + 0.5 / 12, 2022 + 5.5 / 12]
fig3_plot_range = np.arange(2020 + 6.5/12, 2021 + 4.5/12 + 1/12, 1/12)


if plot_impulse_responses:
    # Extract common parameters
    n = baseline_may2021['n']
    H = baseline_may2021['H']
    M = baseline_may2021['M']
    # Create figure
    fig = plt.figure(2, figsize=(15, 12))

    # === Plot for Baseline_May2021 === #
    ColorPlot = ColorCovid
    sIRF1 = baseline_may2021['sIRF1']

    for jn in range(n):
        ax = plt.subplot(np.ceil(n / 2).astype(int), 2, jn + 1)
        quantiles = np.squeeze(sIRF1[:, jn, (np.array(qqq) * M).astype(int)])
        bvar.quantile_plot(np.arange(0, H + 1), quantiles / sIRF1[0, 0, int(0.5 * M)], base_color=red,
                           show_plot=False, run_scenario_analysis=True)
        # Formatting
        ax.grid(True, color='lightgrey', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Horizon', fontsize=10, weight='bold', color='black')
        ax.set_ylabel(YLABELirf[jn], fontsize=10, weight='bold', color='black')
        ax.set_title(series[jn], fontsize=12, weight='bold', color='black')
        ax.tick_params(axis='both', colors='black')
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(left=0)  # Align Y-axis exactly at x = 0

    # === Plot for CVFeb2020_May2021 === #
    ColorPlot = ColorBase
    sIRF1 = cv_feb2020_may2021['sIRF1']

    for jn in range(n):
        plt.subplot(np.ceil(n / 2).astype(int), 2, jn + 1)
        plt.plot(np.arange(0, H + 1), sIRF1[:, jn, int(0.5 * M)] / sIRF1[0, 0, int(0.5 * M)],
                 color=ColorPlot, linewidth=2, linestyle='--')
        plt.grid(True, color='lightgrey', linestyle='--', linewidth=0.5)

    # === Plot for CV_May2021 === #
    ColorPlot = ColorGrey
    sIRF1 = cv_may2021['sIRF1']

    for jn in range(n):
        plt.subplot(np.ceil(n / 2).astype(int), 2, jn + 1)
        plt.plot(np.arange(0, H + 1), sIRF1[:, jn, int(0.5 * M)] / sIRF1[0, 0, int(0.5 * M)],
                 color=ColorPlot, linewidth=2, linestyle='-.')
        plt.axhline(0, color='k')  # Adding zero line
        plt.grid(True, color='lightgrey', linestyle='--', linewidth=0.5)

    # Add legend in the empty subplot (4th row, 2nd column)
    legend_ax = plt.subplot(np.ceil(n / 2).astype(int), 2, 8)
    legend_ax.axis('off')  # Hide the subplot frame
    handles = [
        Line2D([0], [0], color=ColorCovid, linewidth=2),  # Posterior medians (Covid volatility)
        Line2D([0], [0], color=ColorCovid, linewidth=10, alpha=0.5),  # 68-percent credible region
        Line2D([0], [0], color=ColorCovid, linewidth=10, alpha=0.15),  # 95-percent credible region
        Line2D([0], [0], color=ColorBase, linewidth=2, linestyle='--'),  # Constant volatility (2020:2)
        Line2D([0], [0], color=ColorGrey, linewidth=2, linestyle='-.'),  # Constant volatility (2021:5)
    ]
    legend_ax.legend(handles, [
        'Covid volatility: posterior medians',
        'Covid volatility: 68-percent credible regions',
        'Covid volatility: 95-percent credible regions',
        'constant volatility - sample ends in 2020:2: posterior medians',
        'constant volatility - sample ends in 2021:5: posterior medians'
    ], loc='center',
     fontsize=14,  # Larger font size
     frameon=True,  # Add a box around the legend
     borderpad=2,  # Padding inside the legend box
     handlelength=3,  # Length of the legend lines
     bbox_to_anchor=(0.5, 0.5),  # Center the legend
     ncol=1  # Keep items in a single column
                     )
    # Adjust layout and display
    plt.tight_layout()
    plt.savefig(f"{results_folder}/fig2_IRFs_{may2021_vintage_date}.png", dpi=600)


if plot_conditional_forecasts_fig3:
    # Extract common variables
    n = baseline_june2020['n']
    H = baseline_june2020['H']
    M = baseline_june2020['M']

    DataMACRO = baseline_june2020['DataMACRO']
    T1av = baseline_june2020['T1av']
    Tend = baseline_june2020['Tend']
    indmacro = baseline_june2020['indmacro']

    #### === Figure 31: Baseline_June2020 === ####
    fig31, axs31 = plt.subplots(7, 1, figsize=(5, 12))  # Create figure
    fig31.suptitle('Covid volatility - est. sample ends in 2020:6', fontsize=12, fontweight='bold', color=ColorCovid)

    for ii in range(n):
        ax = axs31[ii]  # Assign correct subplot

        if ii != 0:
            aux = np.exp(np.squeeze(baseline_june2020['IRFAsorted'][ii, :, (np.array(qqq) * M).astype(int)]) / 100).T
            normalization = aux[12, 2]
            aux = 100 * aux / normalization
            realization = 100 * np.exp(np.log(DataMACRO.loc[T1av + 1:Tend, indmacro[ii]])) / normalization
        else:
            aux = np.squeeze(baseline_june2020['IRFAsorted'][ii, :, (np.array(qqq) * M).astype(int)]).T
            realization = 100 * np.log(DataMACRO.loc[T1av + 1:Tend, indmacro[ii]])

        # Plot quantiles
        plt.sca(ax)
        bvar.quantile_plot(x_dates, aux, base_color=ColorCovid, show_plot=False, run_scenario_analysis=True)

        if ii != 0:
            ax.plot(fig3_plot_range, realization, 'k+', linewidth=1.5)

        ax.grid(True, color='lightgrey', linestyle='--', linewidth=0.5)
        ax.set_xlim(start_end_dates_xlim)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels)
        ax.set_ylabel(YLABELfcst[ii], fontsize=10, weight='bold', color='black')
        ax.set_title(series[ii], fontsize=12, weight='bold', color='black')

    plt.tight_layout()
    plt.savefig(f'{results_folder}/fig31_conditional_forecasts.png', dpi=600)
    plt.close(fig31)  # Close figure to free memory

    #### === Figure 32: CVFeb2020_June2020 === ####
    fig32, axs32 = plt.subplots(7, 1, figsize=(5, 12))
    fig32.suptitle('Constant volatility - est. sample ends in 2020:2', fontsize=12, fontweight='bold', color=ColorBase)

    for ii in range(n):
        ax = axs32[ii]

        if ii != 0:
            aux = np.exp(np.squeeze(cv_feb2020_june2020['IRFAsorted'][ii, :, (np.array(qqq) * M).astype(int)]) / 100).T
            normalization = aux[12, 2]
            aux = 100 * aux / normalization
            realization = 100 * np.exp(np.log(DataMACRO.loc[T1av + 1:Tend, indmacro[ii]])) / normalization
        else:
            aux = np.squeeze(cv_feb2020_june2020['IRFAsorted'][ii, :, (np.array(qqq) * M).astype(int)]).T
            realization = 100 * np.log(DataMACRO.loc[T1av + 1:Tend, indmacro[ii]])

        # Plot quantiles
        plt.sca(ax)
        bvar.quantile_plot(x_dates, aux, base_color=ColorBase, show_plot=False, run_scenario_analysis=False)

        if ii != 0:
            ax.plot(fig3_plot_range, realization, 'k+', linewidth=1.5)

        ax.grid(True, color='lightgrey', linestyle='--', linewidth=0.5)
        ax.set_xlim(start_end_dates_xlim)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels)
        ax.set_ylabel(YLABELfcst[ii], fontsize=10, weight='bold', color='black')
        ax.set_title(series[ii], fontsize=12, weight='bold', color='black')

    plt.tight_layout()
    plt.savefig(f'{results_folder}/fig32_conditional_forecasts.png', dpi=600)
    plt.close(fig32)  # Close figure to free memory

    #### === Adjust Y-Limits Across Subplots === ####
    for jj in range(n):
        ylim1 = axs31[jj].get_ylim()
        ylim2 = axs32[jj].get_ylim()
        YLIM = [min(ylim1[0], ylim2[0]), max(ylim1[1], ylim2[1])]
        axs31[jj].set_ylim(YLIM)
        axs32[jj].set_ylim(YLIM)



if plot_conditional_forecasts_fig4:
    # Extract common variables
    n = baseline_may2021['n']
    H = baseline_may2021['H']
    M = baseline_may2021['M']

    DataMACRO = baseline_may2021['DataMACRO']
    T1av = baseline_may2021['T1av']
    Tend = baseline_may2021['Tend']
    indmacro = baseline_may2021['indmacro']

    #### === Figure 41: Baseline_May2021 === ####
    fig1, axs1 = plt.subplots(7, 1, figsize=(5, 12))
    fig1.suptitle('Covid volatility - est. sample ends in 2021:5', fontsize=12, fontweight='bold', color=ColorCovid)

    for ii in range(n):
        ax = axs1[ii]  # Assign correct subplot

        if ii != 0:
            aux = np.exp(np.squeeze(baseline_may2021['IRFAsorted'][ii, :, (np.array(qqq) * M).astype(int)]) / 100).T
            normalization = aux[12, 2]  # 13th index in MATLAB is 12 in Python
            aux = 100 * aux / normalization
            realization = 100 * np.exp(np.log(DataMACRO.loc[T1av + 1:Tend, indmacro[ii]])) / normalization
        else:
            aux = np.squeeze(baseline_may2021['IRFAsorted'][ii, :, (np.array(qqq) * M).astype(int)]).T
            realization = 100 * np.log(DataMACRO.loc[T1av + 1:Tend, indmacro[ii]])

        # Force bvar.quantile_plot to plot on ax
        plt.sca(ax)  # Set active axis
        bvar.quantile_plot(x_dates, aux, base_color=ColorCovid, show_plot=False, run_scenario_analysis=True)

        ax.grid(True, color='lightgrey', linestyle='--', linewidth=0.5)
        ax.set_xlim(start_end_dates_xlim)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels)
        ax.set_ylabel(YLABELfcst[ii], fontsize=10, weight='bold', color='black')
        ax.set_title(series[ii], fontsize=12, weight='bold', color='black')

    plt.tight_layout()
    plt.savefig(f'{results_folder}/fig41_conditional_forecasts_{may2021_vintage_date}.png', dpi=600)  # Save Covid Volatility Figure
    plt.close(fig1)  # Close the figure to free memory

    #### === Figure 42: CVFeb2020_May2021 === ####
    fig2, axs2 = plt.subplots(7, 1, figsize=(5, 12))
    fig2.suptitle('Constant volatility - est. sample ends in 2020:2', fontsize=12, fontweight='bold', color=ColorBase)


    for ii in range(n):
        ax = axs2[ii]

        if ii != 0:
            aux = np.exp(np.squeeze(cv_feb2020_may2021['IRFAsorted'][ii, :, (np.array(qqq) * M).astype(int)]) / 100).T
            normalization = aux[12, 2]
            aux = 100 * aux / normalization
            realization = 100 * np.exp(np.log(DataMACRO.loc[T1av + 1:Tend, indmacro[ii]])) / normalization
        else:
            aux = np.squeeze(cv_feb2020_may2021['IRFAsorted'][ii, :, (np.array(qqq) * M).astype(int)]).T
            realization = 100 * np.log(DataMACRO.loc[T1av + 1:Tend, indmacro[ii]])

        # Plot quantiles
        plt.sca(ax)  # Set active axis
        bvar.quantile_plot(x_dates, aux, base_color=ColorBase, show_plot=False, run_scenario_analysis=False)

        ax.grid(True, color='lightgrey', linestyle='--', linewidth=0.5)
        ax.set_xlim(start_end_dates_xlim)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels)
        ax.set_ylabel(YLABELfcst[ii], fontsize=10, weight='bold', color='black')
        ax.set_title(series[ii], fontsize=12, weight='bold', color='black')

    plt.tight_layout()
    #plt.show(block=True)  # Stops execution until you close the window

    #### === Adjust Y-Limits Across Subplots === ####
    for jj in range(n):
        ylim1 = axs1[jj].get_ylim()
        ylim2 = axs2[jj].get_ylim()
        YLIM = [min(ylim1[0], ylim2[0]), max(ylim1[1], ylim2[1])]
        axs1[jj].set_ylim(YLIM)
        axs2[jj].set_ylim(YLIM)

    # Save and show figures
    plt.tight_layout()
    plt.savefig(f'{results_folder}/fig42_conditional_forecasts.png', dpi=600)
    plt.show()