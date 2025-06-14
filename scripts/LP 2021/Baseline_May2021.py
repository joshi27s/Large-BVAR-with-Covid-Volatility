# Preliminaries

import pandas as pd
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings("ignore")
import covbayesvar.large_bvar as bvar
#import large_bvar as bvar

# Configuration settings
vis = True  # Set to False to hide figures
estimateBVAR =  True
lags = 13  # Number of lags in VAR, if monthly data
Ndraws = 10000  # Number of draws in MCMC simulation
discard = 5000  # Number of initial draws to discard (burn-in period)
dots_per_inch = 600 # DPI to save high-resolution figures

# Create plot directory based on the COVID variable
results = 'results'
os.makedirs(results, exist_ok=True)

# Load data from the specified path
path = '/Users/sjoshis/Desktop/covbayesvar/scripts/Data/LP_replication_data.xlsx'
data = pd.read_excel(path)

# Extracting and transforming dates
dates = pd.to_datetime(data['DATE'])
data_array = data.drop(columns=['DATE']).to_numpy()
# Data transformations
# UNRATE (unemployment)
data['UNRATE'] = np.exp(data['UNRATE'] / 100)
# Real PCE: nominal PCE / PCE deflator
data['PCE_real'] = data['PCE'] / data['PCEPI']
# Real PCE services: nominal PCE services / PCE services deflator
data['PCE_services_real'] = data['PCES'] / data['DSERRG3M086SBEA']
# Selecting variables in the baseline model
indmacro = ['UNRATE', 'PAYEMS', 'PCE_real', 'PCE_services_real', 'PCEPI', 'DSERRG3M086SBEA', 'PCEPILFE']
# Y-axis labels for IRF and Forecast plots
YLABELirf = ["percentage points", "100 x log points", "100 x log points", "100 x log points", "100 x log points",
             "100 x log points", "100 x log points"]
YLABELfcst = ["percentage points", "index", "index", "index", "index", "index", "index"]

# Choice of estimation sample, constant or varying volatility, and forecasting period
T0 = data.index[(data['DATE'].dt.year == 1988) & (data['DATE'].dt.month == 12)][0]     # beginning of estimation sample
T1estim = data.index[(data['DATE'].dt.year == 2021) & (data['DATE'].dt.month == 5)][0]    # end of estimation sample

T1av = T1estim       # date of last available data for forecasting
Tend = T1estim       # date of last available data in the dataset

# Position of the Feb 2020 observation
Tfeb2020 = data.index[(data['DATE'].dt.year == 2020) & (data['DATE'].dt.month == 2)][0]
Tcovid = Tfeb2020 - T0 + 1           # first time period of COVID (March 2020; set to "None" if constant volatility)

Tjan2019 = Tfeb2020 - 13    # initial date for conditional forecast plots
TendFcst = Tfeb2020 + 22 + 6
# TendFcst = Tfeb2020 + 71   # end date for projections (June 2022)
hmax = TendFcst - T1av    # corresponding maximum forecasting horizon

# Monthly VAR estimation
Ylev = data.loc[T0:T1estim, indmacro]
Ylog_df = 100 * np.log(Ylev)
Ylog = Ylog_df.to_numpy()
Time = data['DATE'].iloc[T0:]
T, n = Ylog.shape

np.random.seed(10)            # random generator seed


############################ Estimate BVAR Model ############################

vintage_date = data.loc[T1estim, 'DATE'].strftime('%b_%Y')  # Format as 'Feb_2020'
bvar_res_file_path = f'./Results/BVAR_Covid_Results_{vintage_date}.pkl'

priors_params = {
    'lambda_mode': 0.2, #  "tightness" of the Minnesota prior: controls the scale of variances and covariances
    'miu_mode': 1, # mean reversion hyperparameter
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

if estimateBVAR:
    bvar_results = bvar.bvarGLP_covid(Ylog, lags=lags, mcmc=1, priors_params=priors_params,
                                      MCMCconst=1, MNpsi=0, sur=0, noc=0, Ndraws=Ndraws, Ndrawsdiscard=discard,
                                      hyperpriors=1, Tcovid=Tcovid)
    bvar_results = bvar.verify_bvar_results(bvar_results)
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(bvar_res_file_path), exist_ok=True)
    # Save the results
    with open(bvar_res_file_path, 'wb') as f:
        pickle.dump(bvar_results, f)

else:
    # Load the results
    with open(bvar_res_file_path, 'rb') as f:
        bvar_results = pickle.load(f)

########################## generalized IRFs to an "unemployment" shock ############################

# Compute the IRFs
H = 60
M = bvar_results['mcmc']['beta'].shape[2]
Dirf1 = np.zeros((H+1, Ylog.shape[1], M))

for jg in range(M):
    Dirf1[:, :, jg] = bvar.bvarIrfs(bvar_results['mcmc']['beta'][:, :, jg], bvar_results['mcmc']['sigma'][:, :, jg],
                                    1, H+1)

sIRF1 = np.sort(Dirf1, axis=2)

############# conditional forecasts ################

YYfcst = np.vstack([100 * np.log(data.loc[Tjan2019:T1av, indmacro].to_numpy()),
    np.full((hmax, n), np.nan)
])

# Conditioning scenario from Blue Chip
YYfcst[-hmax:, 0] = 4 + (5.8 - 4) * (0.85 ** np.arange(hmax))

TTfcst = YYfcst.shape[0]
M = bvar_results['mcmc']['beta'].shape[2]
DRAWSY = np.full((n, TTfcst, M), np.nan)

# Forecasts
for i in range(M):
    betadraw = bvar_results['mcmc']['beta'][:, :, i]
    G = np.linalg.cholesky(bvar_results['mcmc']['sigma'][:, :, i]).T

    if Tcovid is None:
        etapar = [1, 1, 1, 1]
        tstar = 1000000
    else:
        etapar = bvar_results['mcmc']['eta'][i, :]
        tstar = TTfcst - hmax + Tcovid - T

    varc, varZ, varG, varC, varT, varH = bvar.form_companion_matrices_covid(betadraw, G.T, etapar, tstar, n, lags,
                                                                            TTfcst)

    s00 = np.flip(YYfcst[:lags, :], axis=0).T.flatten().reshape(-1, 1)
    P00 = np.zeros((n * lags, n * lags))

    DrawStates, shocks = bvar.disturbance_smoother_var(
        YYfcst, varc, varZ, varG, varC, varT, varH, s00, P00, TTfcst, n, n * lags, n, 'kalman'
    )

    DRAWSY[:, :, i] = DrawStates[:n, :]

IRFA = DRAWSY[:n, :, :]
IRFAsorted = np.sort(IRFA, axis=2)

################## plot of conditional forecasts ##################
qqq = [0.025, 0.16, 0.5, 0.84, 0.975]

ColorCovid = (0.8941, 0.1020, 0.1098)
ColorBase = (44 / 255, 127 / 255, 184 / 255)
ColorGrey = (0.5, 0.5, 0.5)
results_data = {
    'DataMACRO': data,
    'n': n,
    'IRFA': IRFA,
    'IRFAsorted': IRFAsorted,
    'res': bvar_results,
    'Tcovid': Tcovid,
    'qqq': qqq,
    'sIRF1': sIRF1,
    'H': H,
    'M': M,
    'T1av': T1av,
    'Tend': Tend,
    'indmacro': indmacro,
    'ColorCovid': ColorCovid,
    'ColorBase': ColorBase,
    'ColorGrey': ColorGrey
}

vintage_date = data.loc[T1estim, 'DATE'].strftime('%b%Y')  # Format as 'Feb_2020'
# Save all variables to a pickle file
with open(f'{results}/Baseline_{vintage_date}.pkl', 'wb') as f:
    pickle.dump(results_data, f)






