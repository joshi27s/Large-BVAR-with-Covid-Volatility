# Preliminaries

import pandas as pd
import numpy as np
import os
import pickle
import sys
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import covbayesvar.large_bvar as bvar
import plot_bvar
#import large_bvar as bvar

# Configuration settings
vis = True  # Set to False to hide figures
estimateBVAR = False
runUNC = False
runCF = True
plot_scenario_analyses = True
plot_subplot_scenarios = True
plot_conditional_forecasts = True
lags = 4  # Number of lags in VAR, if monthly data
Ndraws = 40000  # Number of draws in MCMC simulation
discard = 20000  # Number of initial draws to discard (burn-in period)
dots_per_inch = 600 # DPI to save high-resolution figures

# Create plot directory based on the COVID variable
scenario_name = 'financialTurmoil'
plotDir = f'{scenario_name}/plots/'
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
data_transformed = bvar.transform_data(Spec, data_array)
T, n = data_transformed.shape
pos = [str(i) for i, val in enumerate(Spec['Prior']) if val == 'WN']

# COVID specific settings
T0 = 0 # Index of start date of estimation
# Index of end date of estimation (February 2020)
Tfeb2020 = np.where((dates.dt.year == 2019) & (dates.dt.month == 12))[0][0]
# First time period of COVID (March 2020)
Tcovid = Tfeb2020 - T0 + 1

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
bvar_res_file_path = f'./Results/BVAR_Results_{date_str}.pkl'
condi_forecasts_file_path = f'./Results/CF_{scenario_name}_{date_str}.pkl'

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

# Scenario starts from 2015, assume historical correlations don't vary over time
# Baseline model without financial market turbulance
# take unconditional forecast based on economic and financial data through 2015:Q1
# first, gather all the data till March 2015
t2015Q2 = np.where((dates.dt.year == 2015) & (dates.dt.month == 9))[0][0]

# Subset the dates and data up to 2015 Q2 + 1
Date = dates.iloc[:t2015Q2 + 1].reset_index(drop=True)
DataRaw = data.iloc[:t2015Q2 + 1, :].reset_index(drop=True)
dataTransformed = data_transformed[:t2015Q2 + 1, :]
h = 12  # Forecasting horizon: 12 quarters in the future

# Initialize YFore with NaNs
YFore = np.nan * np.ones((len(Date) + h, data_transformed.shape[1]))
YFore[:t2015Q2, :n] = dataTransformed[:t2015Q2, :n]

# Get the last year and month from the dates
yEnd, mEnd = Date.iloc[-1].year, Date.iloc[-1].month
next_quarter_start = Date.iloc[-1] + relativedelta(months=3)
fore_dates = pd.date_range(start=next_quarter_start, periods=h, freq='QS')
fore = fore_dates.strftime('%d-%b-%Y')
dates = Date.dt.date
fore_series = pd.Series(pd.to_datetime(fore).date)
DateAll = pd.concat([dates, fore_series]).reset_index(drop=True)

# Create a boolean array to indicate forecasts: 0: original data, 1: forecasts
h_fore = (np.array([0] * len(Date) + [1] * h)).reshape(-1, 1)
# Initialize PredY_unc for unconditional density forecasts
PredY_unc = np.nan * np.ones((len(YFore), YFore.shape[1], ndraws))
uncondi_forecasts_file_path = f'./Results/Uncondi_Forecasts_{scenario_name}_{date_str}.pkl'

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

############################ Calculate conditional forecasts ############################

# n is the number of variables and T is the length of the initial data
Ycond = np.nan * np.ones((len(Date)+h, n))
# Fill Ycond with dataTransformed for rows up to t2015Q2+1 and all columns
Ycond[:t2015Q2 + 1, :n] = dataTransformed[:t2015Q2 + 1, :n]
# Set row t2015Q2+1 conditional only on financial information
Ycond[t2015Q2, ~Spec['isFinancial']] = np.nan  # Set non-financial variables to NaN
Shock = np.nan * np.ones((h_fore.sum(), n))  # h_fore is a boolean array indicating forecasts
# Initialize PredY_con for conditional density forecasts
PredY_con = np.nan * np.ones((len(YFore), YFore.shape[1], ndraws))
if runCF:
    for j in range(ndraws):  # Loop through the number of draws

        if (j % 1000) == 0:
            print(f"Generating conditional forecasts: {j} of {ndraws} draws...")
            sys.stdout.flush()

        # Extract the j-th draw for beta and sigma
        beta_j = bvar_results['mcmc']['beta'][:, :, j]
        Gamma = np.vstack((beta_j[1:, :], beta_j[0, :]))
        Su = bvar_results['mcmc']['sigma'][:, :, j]
        PredY_con[:, :, j] = bvar.VARcf_DKcks(Ycond, bvar_results['lags']['lags'], Gamma, Su, 0)

    print("\n Conditional Forecast generation complete.")

    # Prepare the data to be saved
    save_condi_forecast_data = {
        'PredY_con': PredY_con,
        'Ycond': Ycond
    }

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(condi_forecasts_file_path), exist_ok=True)
    # Save the results
    with open(condi_forecasts_file_path, 'wb') as f:
        pickle.dump(save_condi_forecast_data, f)
    print(f"Results saved to {condi_forecasts_file_path}")

else:
    # Load the results
    try:
        with open(condi_forecasts_file_path, 'rb') as f:
            # saves Pred_unc, DateAll, h_fore, ndraws, Spec, dataTransformed
            loaded_condi_forecast_data = pickle.load(f)

        # Extract variables
        PredY_con = loaded_condi_forecast_data['PredY_con']
        print("Data loaded successfully.")

    except FileNotFoundError:
        print(f"File not found: {condi_forecasts_file_path}")
    except KeyError as e:
        print(f"Missing key in loaded data: {e}")

# Compute the difference between conditional and unconditional forecasts
dY = PredY_con - PredY_unc

# Calculate quantiles of the difference
quantiles = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
dYQQ = np.quantile(dY, quantiles, axis=2)
# Transpose the result to match MATLAB's output format
dYQQ = np.transpose(dYQQ, (1, 2, 0))

############################ Plot scenario analyses ############################

if plot_scenario_analyses:
    plot_bvar.plot_scenario_analyses(DateAll, Spec, dYQQ, Shock, h_fore, plotDir, n, h, dots_per_inch)
    plot_bvar.create_tiled_plots(plotDir)

########### Get conditional forecasts given financial variables except oil prices ############################

plotDir = f'{scenario_name}/plots_noOilCommodity/'
os.makedirs(plotDir, exist_ok=True)
condi_forecasts_file_path = f'./Results/CF_{scenario_name}_noOil_{date_str}.pkl'

Ycond = np.nan * np.ones((len(Date)+h, n))
Ycond[:t2015Q2 + 1, :n] = dataTransformed[:t2015Q2 + 1, :n]
Ycond[t2015Q2, ~Spec['isFinancial']] = np.nan  # Set non-financial variables to NaN

idx_PZTEXP = Spec.index[Spec['SeriesID'] == 'OILPRICEx'][0]  # WTI Spot oil prices (variable # 29)
#idx_GSNE = Spec.index[Spec['SeriesID'] == 'VIXCLSx'][0]  # Non-energy commodities nearby index (variable # 30)
Ycond[t2015Q2, [idx_PZTEXP]] = np.nan
Shock = np.nan * np.ones((h_fore.sum(), n))  # h_fore is a boolean array indicating forecasts
# Initialize PredY_con for conditional density forecasts
PredY_con = np.nan * np.ones((len(YFore), YFore.shape[1], ndraws))
if runCF:
    for j in range(ndraws):  # Loop through the number of draws

        if (j % 1000) == 0:
            print(f"Generating conditional forecasts: {j} of {ndraws} draws...")
            sys.stdout.flush()

        # Extract the j-th draw for beta and sigma
        beta_j = bvar_results['mcmc']['beta'][:, :, j]
        Gamma = np.vstack((beta_j[1:, :], beta_j[0, :]))
        Su = bvar_results['mcmc']['sigma'][:, :, j]
        PredY_con[:, :, j] = bvar.VARcf_DKcks(Ycond, bvar_results['lags']['lags'], Gamma, Su, 0)

    print("\n Conditional Forecast generation complete.")

    # Prepare the data to be saved
    save_condi_forecast_data = {
        'PredY_con': PredY_con,
        'Ycond': Ycond
    }

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(condi_forecasts_file_path), exist_ok=True)
    # Save the results
    with open(condi_forecasts_file_path, 'wb') as f:
        pickle.dump(save_condi_forecast_data, f)
    print(f"Results saved to {condi_forecasts_file_path}")

else:
    # Load the results
    try:
        with open(condi_forecasts_file_path, 'rb') as f:
            # saves Pred_unc, DateAll, h_fore, ndraws, Spec, dataTransformed
            loaded_condi_forecast_data = pickle.load(f)

        # Extract variables
        PredY_con = loaded_condi_forecast_data['PredY_con']
        print("Data loaded successfully.")

    except FileNotFoundError:
        print(f"File not found: {condi_forecasts_file_path}")
    except KeyError as e:
        print(f"Missing key in loaded data: {e}")

# Compute the difference between conditional and unconditional forecasts
dY = PredY_con - PredY_unc

# Calculate quantiles of the difference
quantiles = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
dYQQ = np.quantile(dY, quantiles, axis=2)
# Transpose the result to match MATLAB's output format
dYQQ = np.transpose(dYQQ, (1, 2, 0))

############################ Plot scenario analyses ############################

if plot_scenario_analyses:
    plot_bvar.plot_scenario_analyses(DateAll, Spec, dYQQ, Shock, h_fore, plotDir, n, h, dots_per_inch)
    plot_bvar.create_tiled_plots(plotDir)


