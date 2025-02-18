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
import seaborn as sns
from PIL import Image
import covbayesvar.large_bvar as bvar
# import large_bvar as bvar

# Configuration settings
vis = True  # Set to False to hide figures
estimateBVAR = False
runUNC = False
runCF = False
plot_uncondi_forecasts = False
plot_joint_uncondi_forecasts = False
plot_scenario_analyses = False
plot_subplot_scenarios = False
plot_conditional_forecasts = True
lags = 4  # Number of lags in VAR, if monthly data
Ndraws = 40000  # Number of draws in MCMC simulation
discard = 20000  # Number of initial draws to discard (burn-in period)
dots_per_inch = 600 # DPI to save high-resolution figures

# Create plot directory based on the COVID variable
plotDir = 'Forecast Plots/'
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
pos = [str(i) for i, val in enumerate(Spec['Prior']) if val == 'WN']

# COVID specific settings
T0 = 0 # Index of start date of estimation
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

############################ Plot posterior distributions ############################
def plot_posterior_distribution(param_data, title, file_name, plot_dir, bins=50):
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 4)
    sns.histplot(param_data, bins=bins, ax=ax, color='red')
    ax.set_title(f'Posterior: {title}', fontsize=18, fontweight='bold')
    ax.set_ylabel('Count', fontsize=18, labelpad=10)  # Increased font size and added padding
    ax.grid(True, color='lightgrey', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='x', labelsize=18)  # Increase font size for x-ticks
    ax.tick_params(axis='y', labelsize=18)  # Increase font size for y-ticks
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))  # Reduce the number of x-ticks
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))  # Reduce the number of y-ticks
    plt.savefig(os.path.join(plot_dir, file_name), dpi=dots_per_inch)



# Plot eta
# Function to plot posterior distributions in a 2x2 grid
def plot_posterior_distributions_grid(param_data_list, titles, file_name, plot_dir, bins=50):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))  # Create a 2x2 grid of subplots
    axs = axs.flatten()  # Flatten the 2x2 array of axes for easier iteration

    for i, ax in enumerate(axs):
        sns.histplot(param_data_list[i], bins=bins, ax=ax)
        ax.set_title(f'Posterior: {titles[i]}', fontsize=18, fontweight='bold')
        ax.set_ylabel('Count', fontsize=18, labelpad=10)  # Increased font size and added padding
        ax.grid(True, color='lightgrey', linestyle='--', linewidth=0.5)
        ax.tick_params(axis='x', labelsize=18)  # Increase font size for x-ticks
        ax.tick_params(axis='y', labelsize=18)  # Increase font size for y-ticks
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))  # Reduce the number of x-ticks
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))  # Reduce the number of y-ticks

    plt.tight_layout()
    plt.savefig(os.path.join(plotDir, file_name), dpi=dots_per_inch)
    #plt.close(fig)  # Close the figure to free up memory

# Data and titles for the four eta parameters
param_data_list = [
    bvar_results['mcmc']['eta'][:, 0],
    bvar_results['mcmc']['eta'][:, 1],
    bvar_results['mcmc']['eta'][:, 2],
    bvar_results['mcmc']['eta'][:, 3]
]

titles = [r'$\eta_1$', r'$\eta_2$', r'$\eta_3$', r'$\eta_4$']
# Call the function to create a 2x2 grid of eta posteriors
plot_posterior_distributions_grid(param_data_list, titles, 'post_eta_grid.png', plotDir)
# Posterior distribution of the overall standard deviation of the Minnesota prior
plot_posterior_distribution(bvar_results['mcmc']['lambda'], r'$\lambda$', 'post_lambda.png', plotDir)


############################ Get unconditional forecasts ############################

h = 12  # Forecasting horizon: 12 quarters in the future

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

############################ Plot unconditional forecasts ############################

if plot_uncondi_forecasts:
    # Get the number of rows dynamically from PredY_unc
    num_rows, num_variables, _ = PredY_unc.shape
    # Initialize dPredY_unc with NaNs
    dPredY_unc = np.nan * np.ones((num_rows, num_variables, ndraws))
    # Compute the 12-month growth rates for the remaining rows
    dPredY_unc[4:, :, :] = PredY_unc[4:, :, :] - PredY_unc[:-4, :, :]
    QQ = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # Quantiles of interest
    plot_start = datetime(2019, 12, 1).date()  # The plots start from Jan 1, 2005
    # To calculate the end date for the plot, find the max date in DateAll and add 50 days
    plot_end = DateAll.max() + timedelta(days=50)

    # Define color codes
    blue = (55 / 255, 126 / 255, 184 / 255)  # A blue color
    red = (228 / 255, 26 / 255, 28 / 255)  # A red color

    # Loop through each variable
    for iVar in range(n):  # Assuming Spec has 40 rows, one for each variable

        # Set up the figure
        fig, ax = plt.subplots(figsize=(6, 3))  # Size in inches

        # Check the transformation type for the current variable
        if Spec.loc[iVar, 'Transformation'] == 'log':
            # Calculate quantiles for the 12-month log differences if transformation is 'log'
            yQQNew_transposed = np.squeeze(np.quantile(dPredY_unc[:, iVar, :], QQ, axis=1))
            yQQNew = yQQNew_transposed.T
            ax.set_title(f"{Spec.loc[iVar, 'SeriesName']} (4-quarter percent change)")
        else:
            # Calculate quantiles for the actual predictions if no log transformation
            yQQNew_transposed = np.squeeze(np.quantile(PredY_unc[:, iVar, :], QQ, axis=1))
            yQQNew = yQQNew_transposed.T
            ax.set_title(f"{Spec.loc[iVar, 'SeriesName']}")

        # Call the custom quantile plotting function
        bvar.quantile_plot(DateAll, yQQNew, show_plot=False)
        # Plot the actual data
        ax.plot(dates, yQQNew[:len(dates), 2], 'k-', linewidth=2)  # Black line for the actual data

        # Calculate the min and max while ignoring NaN values
        quantile_min = np.nanmin(yQQNew[:, [1, 3]])
        quantile_max = np.nanmax(yQQNew[:, [1, 3]])
        # Dynamically set y-axis limits based on the quantile data
        buffer = (quantile_max - quantile_min) * 0.1
        ax.set_ylim(quantile_min - buffer, quantile_max + buffer)
        ax.set_xticks(pd.date_range(start=plot_start, end=plot_end, periods=5))
        ax.set_xlim(plot_start, plot_end)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.set_xlabel('Year')
        ax.grid(True, color='lightgrey', linestyle='--', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=8.5)
        ax.tick_params(axis='x', rotation=25)  # Rotate x-axis labels by 25 degrees
        #plt.show()
        # Save the figure
        fig.savefig(f'{plotDir}/fore_{Spec["SeriesID"].iloc[iVar]}.png', bbox_inches='tight', dpi=dots_per_inch)
        # Close the figure to free memory
        plt.close(fig)

############################ Plot joint unconditional forecasts ############################
if plot_joint_uncondi_forecasts:
    # Define the jointMarginalCell equivalent in Python using a dictionary
    joint_marginal_data = {
        'mnemY1': ['GS1', 'GS10', 'FEDFUNDS', 'FEDFUNDS'],
        'mnemY2': ['UNRATE', 'UNRATE', 'GS1', 'UNRATE'],
        'y1lab': ['1-Yr Treasury (%)', '10-Yr Treasury (%)', 'Federal Funds Rate  (%)', 'Federal Funds Rate  (%)'],
        'y2lab': ['Unemployment Rate (%)', 'Unemployment Rate (%)', '1-Yr Treasury (%)', 'Unemployment Rate (%)'],
        'titlelab': ['1-Yr Treasury', '10-Yr Treasury', 'Federal Funds Rate', 'Federal Funds Rate'],
        'outlab': ['GS1 UNRATE', 'GS10 UNRATE', 'FEDFUNDS GS1', 'FERDFUNDS UNRATE'],
        'tranY1': ['lin', 'lin', 'lin', 'lin'],
        'tranY2': ['lin', 'lin', 'lin', 'lin']
    }

    # Convert the dictionary to a DataFrame
    joint_marginal_df = pd.DataFrame(joint_marginal_data)
    fore_series = pd.to_datetime(fore)
    yfore = fore_series.year
    mfore = fore_series.month

    # Estimate the model until September 1, 2024
    # Extract dates corresponding to December 2024
    hVec = fore[mfore == 12]
    # Line Width for Plotting
    LW = 1.5
    plt.close('all')

    # Iterate over the dates in hVec
    for t in hVec:
        # Convert t from 'DD-MMM-YYYY' to datetime.date
        t_datetime = pd.to_datetime(t, format='%d-%b-%Y').date()
        hRef_indices = np.where(DateAll == t_datetime)[0]
        hRef = hRef_indices[0]

        # Iterate over the rows in jointMarginalTable
        for j in range(len(joint_marginal_df)):
            # Close all existing plots
            plt.close('all')

            # Set labels
            y1lab = joint_marginal_df.loc[j, 'y1lab']
            y2lab = joint_marginal_df.loc[j, 'y2lab']
            tLab = joint_marginal_df.loc[j, 'titlelab']
            outLab = joint_marginal_df.loc[j, 'outlab']

            # Set data
            idxY1 = Spec[Spec['SeriesID'] == joint_marginal_df.loc[j, 'mnemY1']].index[0]
            idxY2 = Spec[Spec['SeriesID'] == joint_marginal_df.loc[j, 'mnemY2']].index[0]

            if joint_marginal_df.loc[j, 'tranY1'] == 'log':
                Y1 = dPredY_unc[hRef, idxY1, :].squeeze()
            else:
                Y1 = PredY_unc[hRef, idxY1, :].squeeze()

            if joint_marginal_df.loc[j, 'tranY2'] == 'log':
                Y2 = dPredY_unc[hRef, idxY2, :].squeeze()
            else:
                Y2 = PredY_unc[hRef, idxY2, :].squeeze()

            YY = np.column_stack((Y1, Y2))

            # Generate scatterplot of draws, with marginals on the side panels
            fig, ax = bvar.plot_joint_marginal(YY, None, y1lab, y2lab, vis, LW)
            t_datetime = pd.to_datetime(t, format='%d-%b-%Y')
            title_string = f"{t_datetime.strftime('%Y:%m')} | {dates.iloc[-1].strftime('%Y:%m')}"
            ax.set_title(title_string)
            plt.tight_layout()
            #plt.show()
            filename = f"{plotDir}/{outLab}_{t_datetime.strftime('%Y%m')}_nobin.png"
            plt.savefig(filename, dpi=dots_per_inch)


############################ Calculate conditional forecasts ############################

# Find indices of specific variables
idxCV1 = Spec.index[Spec['SeriesID'] == 'GDPC1'].tolist()[0]  # real GDP
# Create a matrix of NaNs to store shocks
# n is the number of variables and T is the length of the initial data
# Create a matrix for shocks
Shock = np.nan * np.ones((h_fore.sum(), n))  # h_fore is a boolean array indicating forecasts
Shock[0, idxCV1] = 1 # Apply shocks to % change in real GDP

Ycond = np.nan * np.ones((len(DateAll), n))
Ycond[:T, :] = data_transformed


# Initialize PredY_con for conditional density forecasts
PredY_con = np.nan * np.ones((len(DateAll), data_transformed.shape[1], ndraws))
condi_forecasts_file_path = f'./Results/Condi_Forecasts_{date_str}.pkl'
# Convert h_fore to a 1D boolean array
h_fore_1d = h_fore.reshape(-1).astype(bool)

if runCF:
    for j in range(ndraws):  # Loop through the number of draws

        if (j % 1000) == 0:
            print(f"Generating conditional forecasts: {j} of {ndraws} draws...")
            sys.stdout.flush()

        # Extract the j-th draw for beta and sigma
        Ycond[h_fore_1d, :] = PredY_unc[h_fore_1d, :, j] + Shock
        beta_j = bvar_results['mcmc']['beta'][:, :, j]
        Gamma = np.vstack((beta_j[1:, :], beta_j[0, :]))
        Su = bvar_results['mcmc']['sigma'][:, :, j]
        PredY_con[:, :, j] = bvar.VARcf_DKcks(Ycond, bvar_results['lags']['lags'], Gamma, Su, 0)

    print("\n Conditional Forecast generation complete.")

    # Prepare the data to be saved
    save_condi_forecast_data = {
        'PredY_con': PredY_con,
        'Ycond': Ycond,
        'Shock': Shock
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
quantiles = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
dYQQ = np.quantile(dY, quantiles, axis=2)
# Transpose the result to match MATLAB's output format
dYQQ = np.transpose(dYQQ, (1, 2, 0))

############################ Plot scenario analyses ############################

if plot_scenario_analyses:
    # Setting plot styles with LaTeX
    plt.rc('text', usetex=True)
    plt.rc('legend', fontsize='medium')
    plt.rc('axes', labelsize='medium')

    # Define plot variables
    plot_vars = [
        {
            'type': 'Prices',
            'Vars': [
                'PCE: Chain-Type Price Index',
                'PCE Excluding Food and Energy',
                'Crude Oil, spliced WTI and Cushing',
                'CPI-Urban: All Items Less Food and Energy'
            ]
        },
        {
            'type': 'Economic Activity',
            'Vars': [
                'Industrial Production Index',
                'Capacity Utilization: Manufacturing',
                'Housing Starts',
                'Real Private Residential Fixed Investment'
            ]
        },
        {
            'type': 'Real Output and Investment',
            'Vars': [
                'Real Gross Domestic Product',
                'Real Personal Consumption Expenditures',
                'Real Private Non-Residential Fixed Investment',
                'Real Disposable Personal Income'
            ]
        },
        {
            'type': 'Labor Market',
            'Vars': [
                'Civilian Unemployment Rate',
                'Business Sector: Real Compensation Per Hour',
                'All Employees, Total Nonfarm',
                'Real Government Consumption Expenditures and Gross Investment'

            ]
        },
        {
            'type': 'Interest Rates',
            'Vars': [
                '1 Year Treasury Bond Yield',
                '5 Year Treasury Bond Yield',
                '10-Year Treasury Note Yield'

            ]
        },
        {
            'type': 'Asset Prices and Economic Sentiment',
            'Vars': [

                'S&P 500 Index',
                'CBOE Volatility Index: VIX',
                'University of Michigan: Consumer Sentiment',
                'GDP Deflator'

            ]
        },
        {
            'type': 'International Trade and Bond Yields',
            'Vars': [
                'Real Exports of Goods and Services',
                'Real Imports of Goods and Services',
                'Moody Seasoned Aaa Corporate Bond Yield',
                'Moody Seasoned Baa Corporate Bond Yield'

            ]
        },

    ]

    # Assuming h_fore is a numpy array indicating forecast horizon
    iPlotStart = np.where(h_fore == 1)[0][0] - 1  # Index of start of plot
    iPlotEnd = np.where(h_fore == 1)[0][-1]  # Index of end of plot
    # Disable LaTeX rendering in Matplotlib
    plt.rc('text', usetex=False)

    # Before your plotting loop, set a larger figure size
    plt.rcParams['figure.figsize'] = [10, 10]  # Width, Height in inches

    # Loop over each category of variables
    for j, category in enumerate(plot_vars):

        plt.close('all')

        vars = category['Vars']

        # Create a figure with visibility controlled by 'vis'
        fig = plt.figure()
        fig.set_visible(vis)

        for k, var in enumerate(vars):

            idx = Spec.index[Spec['SeriesName'] == var][0]

            if k < 4:
                ax = fig.add_subplot(2, 2, k + 1)

                bvar.quantile_plot(DateAll, dYQQ[:, idx, :].squeeze(), run_scenario_analysis=False)
                ax.plot([DateAll.iloc[iPlotStart], DateAll.iloc[iPlotEnd]], [0, 0], linestyle='-', color='k')
                # Adding grid here
                ax.grid(True, color='lightgrey', linestyle='--', linewidth=0.5)

                titleLab = Spec.at[idx, 'SeriesName']
                transformation = Spec['Transformation'].iloc[idx]
                if transformation == 'log':
                    ax.set_title(f"{titleLab} (% change)", fontsize=10)
                else:
                    ax.set_title(f"{titleLab} (difference)", fontsize=10)

                # Adjust title length
                if len(ax.title.get_text()) > 80:
                    title_words = ax.title.get_text().split()
                    idxCut1 = max([i for i, x in enumerate(title_words) if len(' '.join(title_words[:i])) <= 37])
                    idxCut2 = max([i for i, x in enumerate(title_words) if len(' '.join(title_words[:i])) <= 37 * 2])

                    ax.set_title(' '.join(title_words[:idxCut1]) + '\n' +
                                 ' '.join(title_words[idxCut1:idxCut2]) + '\n' +
                                 ' '.join(title_words[idxCut2:]))

                elif len(ax.title.get_text()) > 37:
                    title_words = ax.title.get_text().split()
                    idxCut = max([i for i, x in enumerate(title_words) if len(' '.join(title_words[:i])) <= 37])
                    ax.set_title(' '.join(title_words[:idxCut]) + '\n' + ' '.join(title_words[idxCut:]), fontsize=10,
                                 fontweight='bold')
                else:
                    ax.set_title(ax.title.get_text() + '\n ', fontsize=10, fontweight='bold')

                ax.title.set_fontsize(fontsize=8)  # Adjust the font size as needed

            else:
                # For categories with more than 4 variables, handle the last two series differently
                # Plot bands using the 'quantilePlot' function
                # In Python, the color is passed as an RGB tuple normalized to [0, 1]
                color = (228 / 255, 26 / 255, 28 / 255)
                bvar.quantile_plot(DateAll, dYQQ[:, idx, :].squeeze(), run_scenario_analysis=True)

                # Set the title based on the category type
                if plot_vars[j]['type'] == 'Prices':
                    ax.set_title('CPI:All Items (blue), Core CPI (red) (% change)')
                elif plot_vars[j]['type'] == 'Borrowing Rates':
                    ax.set_title('5-Year (blue), 10-Year (red): Treasury Yields (%)')
                elif plot_vars[j]['type'] == 'Asset Prices and Money Supply':
                    ax.set_title('M1 (blue), M3 (red) (%)')

            # Plot dot marking impulse if condition is met
            if idx in np.where(~np.isnan(Shock).any(axis=0))[0] and Shock.shape[1] == n:
                yImpulse = dYQQ[-h:, idx, 0]
                yImpulse[np.isnan(Shock[:, idx])] = np.nan
                ax.scatter(DateAll[-h:], yImpulse, color='b', s=10)  # Adjust size 's' as needed

            ax.set_xlim([DateAll.iloc[iPlotStart], DateAll.iloc[iPlotEnd]])
            # Calculate the range of dates for x-ticks
            date_range = pd.date_range(start=DateAll.iloc[iPlotStart], end=DateAll.iloc[iPlotEnd], periods=3)
            # Set the calculated dates as x-ticks
            ax.set_xticks(date_range)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

        #plt.show()
        # Save the figure with a unique name based on the type of plot_vars
        fig.savefig(f"{plotDir}{plot_vars[j]['type']}.png", dpi=dots_per_inch, bbox_inches='tight')

        # Close the figure to free memory
        plt.close(fig)

############################ Create 4x4 subplots ############################
if plot_subplot_scenarios:
    tile_paths = [
        f'{plotDir}Real Output and Investment.png',
        f'{plotDir}Prices.png',
        f'{plotDir}Interest Rates.png',
        f'{plotDir}Asset Prices and Economic Sentiment.png'
    ]
    # Open the images and store them in a list
    images = [Image.open(path) for path in tile_paths]
    width, height = images[0].size

    # Create an output image with enough space to place the tile images
    # Here, we create a 2x2 tile, so we multiply the width and height by 2
    out_image = Image.new('RGB', (width * 2, height * 2))

    # Paste the images into the output image
    for i, img in enumerate(images):
        # Calculate the position of each image (top-left corner coordinates)
        x = (i % 2) * width
        y = (i // 2) * height
        out_image.paste(img, (x, y))

    # Display the composite image
    #out_image.show()

    # Save the composite image to a file
    out_image.save(f'{plotDir}/scorecardTiled1.png', 'PNG', dpi=(600, 600))

############################ Compare conditional and unconditional forecasts ############################

if plot_conditional_forecasts:
    QQ = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # Quantiles of interest

    # Find the indices for the start and end of the forecast period
    iPlotStart = np.where(h_fore == 1)[0][0] - 4*4
    iPlotEnd = np.where(h_fore == 1)[0][-1]

    blue = np.array([30, 90, 150]) / 255  # Blue color for unconditional
    red = np.array([200, 30, 30]) / 255  # Red color for conditional

    black = 'k'  # Black color

    # Loop through each variable
    for iVar in range(n):

        fig = plt.figure(figsize=(9, 5))
        fig.subplots_adjust(hspace=0.4)

        # First subplot for unconditional vs conditional forecasts
        ax1 = fig.add_subplot(2, 1, 1)
        yQQ_transposed = np.squeeze(np.quantile(PredY_unc[:, iVar, :], QQ, axis=1))
        yQQ = yQQ_transposed.T
        yQQ_con_transposed = np.squeeze(np.quantile(PredY_con[:, iVar, :], QQ, axis=1))
        yQQ_con = yQQ_con_transposed.T

        # Plot quantiles and median
        bvar.quantile_plot(DateAll, yQQ, show_plot=False, base_color=blue)
        ax1.plot(DateAll, yQQ_con, color=red, linewidth=1, linestyle=':')
        # Plot median for both unconditional and conditional forecasts
        ax1.plot(DateAll, yQQ_con[:, 2], color=red, linewidth=2)

        # Determine y-axis limits
        quantile_min = min(yQQ[0, :].min(), yQQ_con[0, :].min())  # Minimum of the lower quantiles
        quantile_max = max(yQQ[-1, :].max(), yQQ_con[-1, :].max())  # Maximum of the upper quantiles

        # Determine y-axis limits
        y_min = min(yQQ[0, :].min(), yQQ_con[0, :].min())  # Minimum of the lower quantiles
        y_max = max(yQQ[-1, :].max(), yQQ_con[-1, :].max())  # Maximum of the upper quantiles
        y_range = y_max - y_min  # Range of the data
        padding = y_range * 0.1  # Padding of 5% of the range

        ax1.set_ylim(y_min - padding, y_max + padding)  # Set y-axis limits with padding
        ax1.plot(dates, yQQ[:len(dates), 2], 'k-', linewidth=1.5)
        # Set the x-axis limit to the desired zoom range
        ax1.set_xlim([DateAll[iPlotStart], DateAll[iPlotEnd]])
        # Format the ticks to show the year and month, with less frequency to reduce clutter
        # Set the locator to month to have two ticks per year
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax1.set_xlabel('Year')
        ax1.grid(True, color='lightgrey', linestyle='--', linewidth=0.5)

        # Adjust y-axis limits to better fit the data range
        # Find the actual data range within the plotting window
        data_range_within_plot_window = np.concatenate([
            yQQ[iPlotStart:iPlotEnd, :].flatten(),
            yQQ_con[iPlotStart:iPlotEnd, :].flatten(),
        ])

        # Determine a reasonable range for the y-axis
        quantile_min = np.percentile(data_range_within_plot_window, 5)  # 5th percentile
        quantile_max = np.percentile(data_range_within_plot_window, 95)  # 95th percentile

        # Add a buffer to the range
        buffer = (quantile_max - quantile_min) * 0.5  # 5% buffer for better visibility
        ax1.set_ylim(quantile_min - buffer, quantile_max + buffer)

        # Redraw the current figure with the updated axes
        plt.draw()

        # Create dummy plots with NaN values for legend entries
        p1, = plt.plot(np.nan, np.nan, color=blue, linewidth=2, label='Unc.')
        p2, = plt.plot(np.nan, np.nan, color=red, linestyle=':', linewidth=2, label='Cond.')
        p3, = plt.plot(np.nan, np.nan, color=black, linewidth=1.5, label='Data')

        # Create the legend
        plt.legend(handles=[p1, p2, p3], loc='best')
        # Set font size for the axes
        ax = plt.gca()
        ax.tick_params(labelsize=8)
        ax1.set_title(f"{Spec.loc[iVar, 'SeriesName']} (log-level)"
                      if Spec.loc[iVar, 'Transformation'] == 'log' else f"{Spec.loc[iVar, 'SeriesName']} (level)")

        # Second subplot for differences
        ax2 = fig.add_subplot(2, 1, 2)
        dYQQ = yQQ_con - yQQ  # Difference between conditional and unconditional
        bvar.quantile_plot(DateAll, dYQQ, show_plot=False)

        ax2.plot([DateAll[iPlotStart], DateAll[iPlotEnd]], [0, 0], 'k-', linewidth=1.5)
        ax2.set_xlim([DateAll[iPlotStart], DateAll[iPlotEnd]])
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=12))  # Match the top subplot locator
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax2.set_xlabel('Year')
        ax2.grid(True, color='lightgrey', linestyle='--', linewidth=0.5)
        ax2.set_title('[Conditional] - [Unconditional]')
        #plt.show()
        fig.savefig(f'{plotDir}/subplot_{Spec["SeriesID"].iloc[iVar]}.png', bbox_inches='tight', dpi=dots_per_inch)
        # Close the figure to free memory
        plt.close(fig)
