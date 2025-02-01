# Preliminaries

import pandas as pd
import numpy as np
import os
import pickle
import sys
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import minimize
from matplotlib.ticker import MaxNLocator
import covbayesvar.large_bvar as bvar
# import large_bvar as bvar

#import covbayesvar.large_bvar as bvar

# Configuration settings
vis = True  # Set to False to hide figures
estimateBVAR = False
runUNC = False
plot_uncondi_forecasts_tilted = True
plot_uncondi_forecasts_tilted_median = True
plot_joint_distribution = True

lags = 4  # Number of lags in VAR, if monthly data
Ndraws = 40000  # Number of draws in MCMC simulation
discard = 20000  # Number of initial draws to discard (burn-in period)
dots_per_inch = 600 # DPI to save high-resolution figures

# Create plot directory based on the COVID variable
scenario_name = 'entropicTilting'
plotDir = f'{scenario_name}/'
os.makedirs(plotDir, exist_ok=True)
# Define the start date for forecasting
vint = datetime(2024, 12, 1)

# Load data from the specified path

path = '/Users/sjoshis/Desktop/covbayesvar/scripts/Data/Replication_Quarterly_Data.xlsx'
data = pd.read_excel(path, sheet_name="Medium Data")
Spec = pd.read_excel(path, sheet_name="Descriptive")

# Extracting and transforming dates
dates = pd.to_datetime(data['Date'])
data_array = data.drop(columns=['Date']).to_numpy()

# Transform data according to specification file
data_transformed = bvar.transform_data(Spec, data_array)
T, n = data_transformed.shape

# Extract position indices for specific data transformation
pos = [str(i) for i, val in enumerate(Spec['Prior']) if val == 'WN']

# Conditioning assumptions: SEP released on Dec 2024 SEP for 2027
# Center of the central tendency: midpoint of the range
# find the index of the PCE inflation rate, Federal Funds Rate, and Unemployment Rate in the Spec DataFrame
idxCVPCE = Spec[Spec['SeriesName'] == 'PCE: Chain-Type Price Index'].index[0]
valCVPCE = (2 + 2) / 2

# Core PCE
idxCVPCEcore = Spec[Spec['SeriesName'] == 'PCE Excluding Food and Energy'].index[0]
valCVPCEcore = (2 + 2) / 2

# Unemployment Rate
idxCVLR = Spec[Spec['SeriesName'] == 'Civilian Unemployment Rate'].index[0]
valCVLR = (4 + 4.4) / 2

# Real GDP
idxCVGDP = Spec[Spec['SeriesName'] == 'Real Gross Domestic Product'].index[0]
valCVGDP = (1.8 + 2) / 2

# COVID specific settings
T0 = 0 # Index of start date of estimation
# Index of end date of estimation (February 2020)
Tfeb2020 = np.where((dates.dt.year == 2020) & (dates.dt.month == 3))[0][0]
# First time period of COVID (March 2020)
Tcovid = Tfeb2020 - T0 + 1

############################ Estimate BVAR Model ############################

priors_params = {
    'lambda_mode': 0.2, #  "tightness" of the Minnesota prior: controls the scale of variances and covariances
    'miu_mode': 1, # mean reversion hyperparameter: for long to shocks persist
    'theta_mode': 1, # mode of  cross-variable shrinkage

    'lambda_sd': 0.4, # standard deviation of the Minnesota tightness prior
    'miu_sd': 1, # standard deviation of the persistence prior
    'theta_sd': 1,

    'eta_mode': [0, 0, 0, 0.8], # mode of COVID-19 scaling factor, applied to first 3 months of COVID-19 period
    'eta_sd': [0, 0, 0, 0.2], # standard deviation of the covid-19 scaling factor


    'lambda_min': 0.0001,
    'alpha_min': 0.1,
    'theta_min': 0.0001,
     'miu_min': 0.0001,
    'eta_min': [1, 1, 1, 0.005],

   'lambda_max': 5,
    'alpha_max': 5,
    'theta_max': 50,
    'miu_max': 50,
    'eta_max': [500, 500, 500, 0.995]

}

# Convert the date to a string in the format 'yyyy_mmdd'
date_str = vint.strftime('%Y_%m%d')
# Define the relative file path
bvar_res_file_path = f'./Results/BVAR_Results_{date_str}.pkl'

if estimateBVAR:
    # Estimate on complete panel
    # Find the last time period that does not have NaN values
    Testim = np.nanmax(np.where(~np.isnan(data_transformed.sum(axis=1)))[0]) + 1
    bvar_results = bvar.bvarGLP_covid(data_transformed[:Testim, :], lags=lags, priors_params=priors_params, mcmc=1,
                                      MCMCconst=1, MNpsi=1, sur=0, noc=0, Ndraws=Ndraws, Ndrawsdiscard=discard,
                                      hyperpriors=1, Tcovid=Tcovid)

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(bvar_res_file_path), exist_ok=True)
    # Save the results
    with open(bvar_res_file_path, 'wb') as f:
        pickle.dump(bvar_results, f)

else:
    # Load the results
    with open(bvar_res_file_path, 'rb') as f:
        bvar_results = pickle.load(f)

# Determine the number of draws in the MCMC simulation
ndraws = bvar_results['mcmc']['beta'].shape[2]

############################ Get unconditional forecasts ############################

h = 12  # Forecasting horizon
# Initialize YFore with NaNs
YFore = np.nan * np.ones((len(dates) + h, data_transformed.shape[1]))
YFore[:len(dates), :data_transformed.shape[1]] = data_transformed
# Get the last year and month from the dates
yEnd, mEnd = dates.iloc[-1].year, dates.iloc[-1].month
next_quarter_start = pd.Timestamp(f'{yEnd}-{mEnd + 3}-01')
fore_dates = pd.date_range(start=next_quarter_start, periods=h, freq='QS')
fore = fore_dates.strftime('%d-%b-%Y')
dates = dates.dt.date
fore_series = pd.Series(pd.to_datetime(fore).date)
DateAll = pd.concat([dates, fore_series]).reset_index(drop=True)

# Create a boolean array to indicate forecasts: 0: original data, 1: forecasts
h_fore = (np.array([0] * len(dates) + [1] * h)).reshape(-1, 1)
# Initialize PredY_unc for unconditional density forecasts
PredY_unc = np.nan * np.ones((len(DateAll), data_transformed.shape[1], ndraws))
uncondi_forecasts_file_path = f'./Results/Uncondi_Forecasts_{date_str}.pkl'

if runUNC:
    for j in range(ndraws):  # Loop through the number of draws

        if (j % 1000) == 0 or j == ndraws - 1:
            print(f"Generating unconditional forecasts: {j} of {ndraws} draws...")
            sys.stdout.flush()

        # Extract the j-th draw for beta and sigma
        beta_j = bvar_results['mcmc']['beta'][:, :, j]
        Gamma_j = np.vstack((beta_j[1:, :], beta_j[0, :]))
        Su_j = bvar_results['mcmc']['sigma'][:, :, j]

        PredY_unc[:, :, j] = bvar.VARcf_DKcks(YFore, bvar_results['lags']['lags'], Gamma_j, Su_j, 1)

    print("\n Unconditional Forecast generation complete.")

    # Prepare the data to be saved
    save_forecast_data = {
        'PredY_unc': PredY_unc,
        'DateAll': DateAll,  # assuming DateAll and other variables are defined
        'h_fore': h_fore,
        'ndraws': ndraws,
        'Spec': Spec,
        'dataTransformed': data_transformed
    }

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(uncondi_forecasts_file_path), exist_ok=True)
    # Save the results
    with open(uncondi_forecasts_file_path, 'wb') as f:
        pickle.dump(save_forecast_data, f)
    print(f"Results saved to {uncondi_forecasts_file_path}")

else:
    # Load the results
    try:
        with open(uncondi_forecasts_file_path, 'rb') as f:
            # saves Pred_unc, DateAll, h_fore, ndraws, Spec, dataTransformed
            loaded_forecast_data = pickle.load(f)

        # Extract variables
        PredY_unc = loaded_forecast_data['PredY_unc']
        print("Data loaded successfully.")

    except FileNotFoundError:
        print(f"File not found: {uncondi_forecasts_file_path}")
    except KeyError as e:
        print(f"Missing key in loaded data: {e}")


############################ Entropic Tilting: Shifting the mean ############################

n = PredY_unc.shape[1]  # Number of series (second dimension)
# 12-month change
dPredY_unc = np.vstack((
    np.full((4, n, ndraws), np.nan),  # Add NaNs at the beginning
    PredY_unc[4:, :, :] - PredY_unc[:-4, :, :]
))
# MCMC Draws for tilting variables (PCE inflation, unemployment rate, and federal funds rate) at forecast horizon
part1 = dPredY_unc[-1, [idxCVPCE, idxCVPCEcore, idxCVGDP], :].T
part2 = PredY_unc[-1, [idxCVLR], :].T
YYh = np.hstack((part1, part2))
# Target values: FOMC's central tendency projections for the targeted variables
target = np.array([valCVPCE, idxCVPCEcore, idxCVGDP, valCVLR])

# Objective function to find the optimal lambda (Lagrange multiplier) values
def fun(gamma, YY, g0):
    return np.sum(np.exp((YY - g0) @ gamma))

# Optimization setup
opts = {'tol': 1e-20, 'options': {'maxiter': 1000}}
objective = lambda x: fun(x, YYh, target) # Objective function for optimization
gamma_init = np.ones(len(target)) # Initial guess for gamma
# Uses a quasi-Newton method for constrained optimization:
# minimizes the divergence between the original and tilted distribution while ensuring the mean of the tilted
# distribution satisfies the target constraints
res = minimize(objective, gamma_init, method='L-BFGS-B', **opts)
# optimal gamma values that adjust the weights of the forecast draws to align the distribution with the targets
gammaStarMean = res.x

# minimized objective function value (Kullback-Leibler divergence between the original and tilted distributions)
fStar = res.fun
# re-calculate weights for the forecast draws using the optimal gamma values
# normalize the weights to sum to 1
wStarMean = np.exp(YYh @ gammaStarMean) / np.sum(np.exp(YYh @ gammaStarMean))

# Verify moment condition
# check the mean of the re-weighted draws to ensure it aligns with the target values
print("Moment:")
# weighted average of the forecasted draws that must equalize the target values
print(np.mean(wStarMean[:, None] * YYh * 10000, axis=0))  # Scale by 10000
print("Target:")
print(target)

############################ Entropic Tilting: Shifting the median ############################
# Create the indicator matrix to check if every value in the forecast matrix is less than or equal to the target value
YYhTemp = YYh <= target
YYhTemp = YYhTemp.astype(int)  # Convert the binary (True/False) to integer (1 and 0)

# Define the optimization objective
# find the optimal gamma that minimizes the Kullback-Leibler divergence between the original and tilted distributions
objective = lambda gamma: np.sum(np.exp((YYhTemp - 0.5) @ gamma))

# Perform optimization
gamma_init =  np.ones(len(target))  # Initial guess
res = minimize(objective, gamma_init, method='L-BFGS-B', options=opts)
gammaStarMedian = res.x
fStar = res.fun

# Calculate weights for each forecast draw using the optimal gamma values
# Reweight the forecast draws (rows of YYh) using the optimized gamma values.
# Normalize the weights so they sum to 1.
wStarMedian = np.exp(YYhTemp @ gammaStarMedian) / np.sum(np.exp(YYhTemp @ gammaStarMedian))

# Check that conditioning assumptions are satisfied
# Sort the forecast values (YYh) for each variable to compute the weighted median.
temp_s = np.sort(YYh, axis=0)  # Sorted values
idx = np.argsort(YYh, axis=0)  # Indices for sorting
cumsum_w = np.cumsum(wStarMedian[idx[:, 0]])  # Cumulative weights for the first column
j = np.argmin(np.abs(cumsum_w - 0.5))  # Find the index closest to 0.5
print("Value at median:", temp_s[j])  # Value corresponding to median
print("Target:")
print(target)
print("Median:")
median_values = np.array([
    bvar.wquantile(YYh[:, 0].reshape(1,-1), 0.5, wStarMedian),  # Weighted median for column 1
    bvar.wquantile(YYh[:, 1].reshape(1,-1), 0.5, wStarMedian),
    bvar.wquantile(YYh[:, 2].reshape(1,-1), 0.5, wStarMedian)
])
print(median_values)

############################ Plot unconditional forecasts against tilted ############################

if plot_uncondi_forecasts_tilted:
    # Quantiles of interest
    QQ = [0.2, 0.3, 0.4, 0.5, 0.6,  0.7, 0.8]  # Quantiles of interest
    blue_color = (55 / 255, 126 / 255, 184 / 255)
    red_color = (228 / 255, 26 / 255, 28 / 255)

    # Define start and end dates for plotting
    plot_start = datetime(2019, 12, 1)
    plot_end = DateAll.max() + timedelta(days=50)

    # Loop through each variable
    for iVar in range(n):  # Looping over variables in Spec
        plt.close('all')  # Close all previous figures

        # Create a new figure
        fig, ax = plt.subplots(figsize=(9, 5))  # Size in inches

        # Check the transformation type for the current variable
        if Spec.loc[iVar, 'Transformation'] == 'log':
            # Calculate quantiles for the 12-month log differences if transformation is 'log'
            yQQNew_transposed = np.squeeze(np.quantile(dPredY_unc[:, iVar, :], QQ, axis=1))
            yQQNew = yQQNew_transposed.T
            yQQTiltMean = bvar.wquantile(dPredY_unc[:, iVar, :], QQ, wStarMean)
            yQQTiltMedian = bvar.wquantile(dPredY_unc[:, iVar, :], QQ, wStarMedian)

            ax.set_title(f"{Spec.loc[iVar, 'SeriesName']} (4-quarter percent change)")
        else:
            # Calculate quantiles for the actual predictions if no log transformation
            yQQNew_transposed = np.squeeze(np.quantile(PredY_unc[:, iVar, :], QQ, axis=1))
            yQQNew = yQQNew_transposed.T
            yQQTiltMean = bvar.wquantile(PredY_unc[:, iVar, :], QQ, wStarMean)
            yQQTiltMedian = bvar.wquantile(PredY_unc[:, iVar, :], QQ, wStarMedian)

            ax.set_title(f"{Spec.loc[iVar, 'SeriesName']}")

        # Plot unconditional quantiles
        bvar.quantile_plot(DateAll, yQQNew, show_plot=False)
        # Plot tilted quantiles
        pUNC, = ax.plot([], [], color=blue_color)  # Unconditional (legend placeholder)
        for i in range(yQQTiltMean.shape[1]):  # Iterate over quantiles if 2D
            ax.plot(DateAll, yQQTiltMean[:, i], color='red', linestyle=':',
                    label=r"$\mathrm{mean} \ \pi_{\mathrm{LR}} = 2$" if i == 0 else None)
        for i in range(yQQTiltMedian.shape[1]):  # Iterate over quantiles if 2D
            ax.plot(DateAll, yQQTiltMedian[:, i], color='green', linestyle='-.',
                    label=r"$\mathrm{med} \ \pi_{\mathrm{LR}} = 2$" if i == 0 else None)
        # Plot the actual data
        ax.plot(dates, yQQNew[:len(dates), 2], 'k-', label='Actual')

        # Set x-ticks and x-limits
        ax.set_xticks(pd.date_range(start=plot_start, end=plot_end, periods=5))
        ax.set_xlim(plot_start, plot_end)
        # Format x-axis with date formatter
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.tick_params(axis='x', labelsize=12, rotation=25)  # Rotate x-axis labels

        # Add grid, labels, and legend
        ax.set_xlabel('Year', fontsize=12)
        ax.grid(True, color='lightgrey', linestyle='--', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=8.5)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=3, frameon=False)
        #plt.show()
        # Save the plot
        fig.savefig(f'{plotDir}/fore_unc_ET_{Spec.loc[iVar, "SeriesID"]}.png', bbox_inches='tight',
                    dpi=dots_per_inch)
        plt.close(fig) # Close the figure to free memory

############################ Plot unconditional forecasts against tilted median ############################

if plot_uncondi_forecasts_tilted_median:
    # Quantiles of interest
    QQ = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # Quantiles of interest
    blue_color = (55 / 255, 126 / 255, 184 / 255)
    red_color = np.array([228, 26, 28]) / 255
    # Define start and end dates for plotting
    plot_start = datetime(2019, 12, 1)
    plot_end = DateAll.max() + timedelta(days=50)

    # Loop through each variable
    for iVar in range(n):  # Looping over variables in Spec
        plt.close('all')  # Close all previous figures

        # Create a new figure
        fig, ax = plt.subplots(figsize=(9, 5))  # Size in inches

        # Check the transformation type for the current variable
        if Spec.loc[iVar, 'Transformation'] == 'log':
            # Calculate quantiles for the 12-month log differences if transformation is 'log'
            yQQNew_transposed = np.squeeze(np.quantile(dPredY_unc[:, iVar, :], QQ, axis=1))
            yQQNew = yQQNew_transposed.T
            yQQTiltMean = bvar.wquantile(dPredY_unc[:, iVar, :], QQ, wStarMean)
            yQQTiltMedian = bvar.wquantile(dPredY_unc[:, iVar, :], QQ, wStarMedian)

            ax.set_title(f"{Spec.loc[iVar, 'SeriesName']} (4-quarter percent change)")
        else:
            # Calculate quantiles for the actual predictions if no log transformation
            yQQNew_transposed = np.squeeze(np.quantile(PredY_unc[:, iVar, :], QQ, axis=1))
            yQQNew = yQQNew_transposed.T
            yQQTiltMean = bvar.wquantile(PredY_unc[:, iVar, :], QQ, wStarMean)
            yQQTiltMedian = bvar.wquantile(PredY_unc[:, iVar, :], QQ, wStarMedian)
            ax.set_title(f"{Spec.loc[iVar, 'SeriesName']}")

        # Plot unconditional quantiles
        bvar.quantile_plot(DateAll, yQQNew, show_plot=False)
        # Plot tilted quantiles
        pUNC, = ax.plot([], [], color=blue_color, label='Unconditional')
        bvar.quantile_plot(DateAll, yQQTiltMedian, show_plot=False, base_color=red_color, run_scenario_analysis=True)
        pETMed, = ax.plot([], [], color='red', label='Conditional')

        # Plot the actual data
        ax.plot(dates, yQQNew[:len(dates), 2], 'k-', label='Actual')
        # Set x-ticks and x-limits
        ax.set_xticks(pd.date_range(start=plot_start, end=plot_end, periods=5))
        ax.set_xlim(plot_start, plot_end)
        # Format x-axis with date formatter
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.tick_params(axis='x', labelsize=14, rotation=25)  # Increase x-axis label font size
        ax.tick_params(axis='y', labelsize=14)  # Increase y-axis label font size
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))  # Adjust `nbins` for fewer ticks

        # Add grid, labels, and legend
        ax.set_xlabel('Year', fontsize=14)
        ax.grid(True, color='lightgrey', linestyle='--', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=8.5)
        ax.legend(loc='best', fontsize=12,  frameon=False)
        # ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=3, frameon=False)
        #plt.show()
        # Save the plot
        fig.savefig(f'{plotDir}/fore_unc_ETMedian_{Spec.loc[iVar, "SeriesID"]}.png', bbox_inches='tight' ,
                    dpi=dots_per_inch)
        plt.close(fig)  # Close the figure to free memory

########################### Joint distribution of PCE inflation rate and unemployment rate ###########################

if plot_joint_distribution:
    # SEP Forecast values
    GDP2026 = 2  # central tendencies of SEP Forecast of GDP for 2026: 1.9-2.1 percent
    GDP2027 = 1.9  # central tendencies of SEP forecast for 2027: 1.8-2 percent
    CorePCE2026 = 2.15  # central tendencies of SEP Forecast of Core PCE inflation rate for 2026: 2-2.3
    CorePCE2027 = 2.0  # central tendencies SEP forecast for 2027: 2-2 percent

    # Illustrative plots: Joint distribution of PCE inflation rate and unemployment rate

    # Plot joint/marginal for Nov 2027 in Dec 2024
    YY = np.array([dPredY_unc[-1, idxCVPCEcore, :], dPredY_unc[-1, idxCVGDP, :]]).T
    fig, ax = bvar.plot_weighted_joint_and_marginals(YY, wStarMedian, 'Core PCE', 'Real GDP', vis=True, LW=1.5, Y0=[CorePCE2027, GDP2027])
    ax.set_title('Forecast for Q4-2027 in Q4-2024')
    plt.savefig(f'{plotDir}/jointMarginal_corePCE_GDP_ET_2027Q4.png', dpi=dots_per_inch)
    plt.close(fig)

    # Plot joint/marginal for Nov 2026 in Dec 2024
    YY = np.array([dPredY_unc[-5, idxCVPCEcore, :], dPredY_unc[-5, idxCVGDP, :]]).T
    fig, ax = bvar.plot_weighted_joint_and_marginals(YY, wStarMedian, 'Core PCE', 'Real GDP', vis=True, LW=1.5, Y0=[CorePCE2026, GDP2026])
    ax.set_title('Forecast for Q4-2026 in Q4-2024')
    plt.savefig(f'{plotDir}/jointMarginal_corePCE_GDP_ET_2026Q4.png', dpi=dots_per_inch)
    plt.close(fig)

    # At the tilted horizon
    YY = np.array([dPredY_unc[-1, idxCVPCEcore, :], dPredY_unc[-1, idxCVGDP, :]]).T
    color_map = 'inferno_r'
    # Visualization of weights: Match the means
    plt.figure()
    plt.scatter(
        YY[:, 0],
        YY[:, 1],
        s=wStarMean * 10000,  # Use transformed weights for size
        c=wStarMean,  # Color based on transformed weights
        cmap=color_map,  # Use shades of blue
        alpha=1,  # Increase alpha for visibility
    )
    plt.xlabel('Core PCE inflation Rate', labelpad=15)  # Add padding to ensure the label is visible
    plt.ylabel('Real GDP')
    plt.gcf().set_size_inches(8, 4)  # Increase figure size for better visibility
    # Customize gridlines for lighter appearance
    plt.grid(color='lightgrey', linestyle='--', linewidth=0.5)
    plt.savefig(f'{plotDir}/mean_matching_corePCE_GDP_ET_2027Q4.png',bbox_inches='tight', dpi=dots_per_inch)
    plt.close()

    # Visualization of weights: Match the medians
    plt.figure()
    scatter = plt.scatter(
        YY[:, 0],
        YY[:, 1],
        s=wStarMedian * 10000,  # Use transformed weights for size
        c=wStarMedian,  # Color based on transformed weights
        cmap=color_map,  # Use shades of blue
        alpha=1,
    )
    plt.title('Draws for Q4-2027 in Q4-2024 (weighted)')
    plt.grid(True)
    plt.box(True)
    plt.xlabel('Core PCE inflation Rate', labelpad=15)  # Add padding to ensure the label is visible
    plt.ylabel('Real GDP')
    plt.colorbar(scatter, label='Weight Intensity')  # Add colorbar for reference
    plt.gcf().set_size_inches(8, 4)
    plt.grid(color='lightgrey', linestyle='--', linewidth=0.5)
    plt.savefig(f'{plotDir}/draws_corePCE_GDP_ET_202711.png', bbox_inches='tight', dpi=dots_per_inch)
    plt.close()