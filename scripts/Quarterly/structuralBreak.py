# Preliminaries

import pandas as pd
import numpy as np
import os
import pickle
import sys
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import textwrap
import covbayesvar.large_bvar as bvar
#import large_bvar as bvar

# Configuration settings
vis = True  # Set to False to hide figures
estimateBVAR = False
lags = 4  # Number of lags in VAR, if monthly data
Ndraws = 40000  # Number of draws in MCMC simulation
discard = 20000  # Number of initial draws to discard (burn-in period)
dots_per_inch = 600 # DPI to save high-resolution figures

# Create plot directory based on the COVID variable
scenario_name = 'structuralBreak'
plotDir = f'{scenario_name}/'
os.makedirs(plotDir, exist_ok=True)
# Define the start date for forecasting
vint = datetime(2024, 12, 1)

# Load data from the specified path

path = ('/Users/sjoshis/Desktop/Fall 2022/ECON527 Macroeconometrics/BVAR of US Economy/'
        'Python Codes/Data/Replication_Quarterly_Data.xlsx')
data = pd.read_excel(path, sheet_name="Medium Data")
Spec = pd.read_excel(path, sheet_name="Descriptive")

# Extracting and transforming dates
dates = pd.to_datetime(data['Date'])
data_array = data.drop(columns=['Date']).to_numpy()
data_transformed = bvar.transform_data(Spec, data_array)
T, n = data_transformed.shape
# Find the date of the break
jBreak = np.where((dates.dt.year == 2020) & (dates.dt.month == 9))[0][0]  # Date of break
heff = T - jBreak  # Number of periods being forecasted
pos = [str(i) for i, val in enumerate(Spec['Prior']) if val == 'WN']

# List of real activity variables
realActivityVars = [
    'GDPC1',  # Real GDP
    'PCECC96', # Personal Consumption Expenditures
    'PRFIx',  # Real Private Residential Fixed Investment
    'PNFIx', # Real Private Non-Residential Fixed Investment
    'EXPGSC1',  # Real Exports of Goods and Services,
    'IMPGSC1',  # Real Imports of Goods and Services,
    'GCEC1',  # Real Government Consumption Expenditures and Gross Investment
    'INDPRO',  # Industrial Production Index
    'CUMFNS',  # Capacity Utilization: Manufacturing
    'UNRATE',  # Civilian Unemployment Rate
    'RCPHBS',  # Business Sector: Real Compensation Per Hour
    'HOUST',  # Housing Starts
    'DPIC96'  # Real Disposable Personal Income

]

# Settings for two conditioning schemes
Settings = [
    {
        'idxCond': Spec.index[Spec['SeriesID'].isin(realActivityVars)].tolist(),
        'labCond': 'real activity'  # Condition on real activity
    },
    {
        'idxCond': Spec.index[Spec['SeriesID'].isin(['GDPC1', 'UNRATE'])].tolist(),
        'labCond': 'GDP and unemployment rate'  # Condition on GDP and unemployment rate
    }
]

# COVID specific settings
T0 = 0 # Index of start date of estimation
Tfeb2020 = np.where((dates.dt.year == 2019) & (dates.dt.month == 12))[0][0]
Tcovid = Tfeb2020 - T0 + 1 # First time period of COVID (March 2020)

############################ Estimate BVAR Model ############################

priors_params = {
    'lambda_mode': 0.6, #  "tightness" of the Minnesota prior: controls the scale of variances and covariances
    'miu_mode': 1, # mean reversion hyperparameter
    'theta_mode': 1, # mode of  cross-variable shrinkage

    'lambda_sd': 0.3, # standard deviation of the Minnesota tightness prior
    'miu_sd': 0.5, # standard deviation of the persistence prior
    'theta_sd': 0.5,

    'eta_mode': [0, 0.8, 0.7, 0.6], # mode of COVID-19 scaling factor, applied to first 3 quarters of COVID-19 period
    'eta_sd': [0, 0.2, 0.15, 0.1], # standard deviation of the covid-19 scaling factor


    'lambda_min': 0.0001,
    'alpha_min': 0.1,
    'theta_min': 0.0001,
     'miu_min': 0.0001,
    'eta_min': [0.0, 0.5, 0.5, 0.3],

   'lambda_max': 5,
    'alpha_max': 5,
    'theta_max': 50,
    'miu_max': 50,
    'eta_max': [500, 500, 500, 0.995]

}

# Convert the date to a string in the format 'yyyy_mmdd'
date_str = vint.strftime('%Y_%m%d')
bvar_res_IS_file_path = f'./Results/BVAR_Results_IS_{date_str}.pkl'
bvar_res_OOS_file_path = f'./Results/BVAR_Results_OOS_{date_str}.pkl'
condi_forecasts_file_path = f'./Results/CF_{scenario_name}_{date_str}.pkl'

if estimateBVAR:
    # Out of sample: model estimated through the break
    # These results are out-of-sample in the sense that forecasts are generated for dates beyond the break
    bvar_results_OOS = bvar.bvarGLP_covid(data_transformed[:jBreak, :], lags=lags, priors_params=priors_params,
                                         mcmc=1, MCMCconst=1, MNpsi=1, sur=0, noc=0, Ndraws=Ndraws,
                                         Ndrawsdiscard=discard, hyperpriors=1, Tcovid=Tcovid)

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(bvar_res_OOS_file_path), exist_ok=True)
    # Save the results
    with open(bvar_res_OOS_file_path, 'wb') as f:
        pickle.dump(bvar_results_OOS, f)

    # In-sample: model estimated on full date
    # These results are in-sample in the sense that forecasts are generated for the dates contained in this sample
    Testim = np.nanmax(np.where(~np.isnan(data_transformed.sum(axis=1)))[0]) + 1
    bvar_results_IS = bvar.bvarGLP_covid(data_transformed[:Testim, :], lags=lags, priors_params=priors_params,
                                         mcmc=1, MCMCconst=1, MNpsi=1, sur=0, noc=0, Ndraws=Ndraws,
                                         Ndrawsdiscard=discard, hyperpriors=1, Tcovid=Tcovid)

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(bvar_res_IS_file_path), exist_ok=True)
    # Save the results
    with open(bvar_res_IS_file_path, 'wb') as f:
        pickle.dump(bvar_results_IS, f)


else:
    # Load the results
    with open(bvar_res_IS_file_path, 'rb') as f:
        bvar_results_IS = pickle.load(f)
    with open(bvar_res_OOS_file_path, 'rb') as f:
        bvar_results_OOS = pickle.load(f)

# Determine the number of draws in the MCMC simulation
ndraws = bvar_results_IS['mcmc']['beta'].shape[2]

############################ Get unconditional forecasts ############################

for setting in Settings:
    idxCond = setting['idxCond']
    labCond = setting['labCond']

    # Generate conditional forecasts
    YCond = np.nan * np.ones((T, n))
    YCond[:jBreak, :] = data_transformed[:jBreak, :] # fill historical data up to the break for non-conditioning variables
    YCond[:, idxCond] = data_transformed[:, idxCond] # fill paths of conditioning variables for all time period

    # Initialize storage for conditional forecasts
    PredYIS = np.nan * np.ones((T, n, Ndraws - discard))
    PredYOOS = np.nan * np.ones((T, n, Ndraws - discard))

    for j in range(Ndraws - discard):  # Loop through the number of draws

        if (j % 1000) == 0 or j == Ndraws - 1:
            print(f"Processing draw {j} of {Ndraws - discard}...")
            sys.stdout.flush()

        # In-Sample (IS) forecasts
        beta_j = bvar_results_IS['mcmc']['beta'][:, :, j]
        Gamma_j = np.vstack((beta_j[1:, :], beta_j[0, :]))
        Su_j = bvar_results_IS['mcmc']['sigma'][:, :, j]
        PredYIS[:, :, j] = bvar.VARcf_DKcks(YCond, bvar_results_IS['lags']['lags'], Gamma_j, Su_j, 1)

        # In-Sample (IS) forecasts
        beta_j = bvar_results_OOS['mcmc']['beta'][:, :, j]
        Gamma_j = np.vstack((beta_j[1:, :], beta_j[0, :]))
        Su_j = bvar_results_OOS['mcmc']['sigma'][:, :, j]
        PredYOOS[:, :, j] = bvar.VARcf_DKcks(YCond, bvar_results_OOS['lags']['lags'], Gamma_j, Su_j, 1)

    # Growth rates (quarterly growth, annualized)
    dPredYIS = np.concatenate((np.nan * np.ones((1, n, Ndraws - discard)),
        (PredYIS[1:, :, :] - PredYIS[:-1, :, :]) * 4), axis=0)

    dPredYOOS = np.concatenate((np.nan * np.ones((1, n, Ndraws - discard)),
        (PredYOOS[1:, :, :] - PredYOOS[:-1, :, :]) * 4), axis=0)

    # Compute growth rates for the historical data
    dY = np.concatenate((np.nan * np.ones((1, n)), 4 * (data_transformed[1:, :] - data_transformed[:-1, :])
    ), axis=0)

    ######################### Create figures. Use in-sample bands. OOS median ########################################

    # Identify variables not included in the conditioning set
    idxNotCond = list(set(range(n)) - set(idxCond))
    # Define plot range
    xl_start = datetime(2019, 12, 1).date()  # The plots start from Jan 1, 2005
    xl_end = dates.iloc[-1].date()  # End date
    QQ = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # Quantiles of interest
    blue_color = (55 / 255, 126 / 255, 184 / 255)
    red_color = np.array([228, 26, 28]) / 255
    # Define a maximum title length before wrapping
    MAX_TITLE_LENGTH = 50

    # Loop over all variables not in the conditioning set
    for jVar in idxNotCond:
        plt.close('all')  # Close previous figures

        if Spec.loc[jVar, 'Transformation'] == 'log':
            YYIS = np.squeeze(dPredYIS[:, jVar, :])  # In-sample growth rates
            YYOOS = np.squeeze(dPredYOOS[:, jVar, :])  # Out-of-sample growth rates
            Yobs = dY[:, jVar]
            title_suffix = "(log)"# Observed growth rates
        elif Spec.loc[jVar, 'Transformation'] == 'lin':
            YYIS = np.squeeze(PredYIS[:, jVar, :])  # In-sample levels
            YYOOS = np.squeeze(PredYOOS[:, jVar, :])  # Out-of-sample levels
            Yobs = data_transformed[:, jVar]  # Observed levels
            title_suffix = "(level)"

        # Create the figure
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.grid(True, linestyle='--', linewidth=0.5, color='lightgrey')

        # Plot in-sample forecast quantiles
        bvar.quantile_plot(dates, np.quantile(YYIS, QQ, axis=1).T, show_plot=False)
        # Plot tilted quantiles
        pUNC, = ax.plot([], [], color=blue_color, label='In-Sample')
        # Plot out-of-sample median forecasts
        ax.plot(dates, np.median(YYOOS, axis=1), color=red_color, linewidth=2, linestyle=':', label='Out-of-Sample')
        # Plot observed data
        ax.plot(dates, Yobs, color='black', linewidth=1.5, label='Actual')
        # Wrap title if it's too long
        title_text = f"{Spec.loc[jVar, 'SeriesName']}, conditional on {labCond}"
        if len(title_text) > MAX_TITLE_LENGTH:
            title_text = "\n".join(textwrap.wrap(title_text, width=MAX_TITLE_LENGTH))

        ax.set_title(title_text, fontsize=12, fontweight='bold')  # Reduce fontsize for very long titles
        # Set plot limits
        ax.set_xlim([xl_start, xl_end])
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.tick_params(axis='x', labelsize=12, rotation=25)  # Increase font size and rotate labels
        ax.tick_params(axis='y', labelsize=12)
        # Set y-axis locator for fewer ticks
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.set_xlabel('Year', fontsize=14)
        ax.legend(loc='best', fontsize=12, frameon=False)  # Adjust legend formatting
        # plt.show()
        fig.savefig(f"{plotDir}/condFore_IS_OOS_conditioning_{labCond}_{Spec.loc[jVar, 'SeriesID']}.png",
                    dpi=dots_per_inch, bbox_inches='tight')
        plt.close(fig)


