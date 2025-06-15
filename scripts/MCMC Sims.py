import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings("ignore")
import covbayesvar.large_bvar as bvar
import pandas as pd

# Option to save simulation results
save_param_draws = False
# Directory setup
directory = 'MCMC Plots'
if not os.path.exists(directory):
    os.makedirs(directory)
filename = os.path.join(directory, 'simulation_results.pkl')

# Simulation parameters
T = 1000
c = np.array([0.5, -0.2])
A = np.array([[0.7, 0.1], [0.05, 0.95]])
sigma = np.array([[1, 0.3], [0.3, 1]])
np.random.seed(42)

# Generate simulated data
Y = np.zeros((T, len(c)))
for t in range(1, T):
    Y[t] = c + Y[t - 1] @ A + np.random.multivariate_normal(np.zeros(len(c)), sigma)

if save_param_draws:
    # Save simulation results
    resGLP = bvar.bvarGLP(Y, lags=1, Ndraws=100000, mcmc=1, MCMCconst=0, MNpsi=0, MNalpha=0, sur=0, noc=0, hyperpriors=1)
    with open(filename, 'wb') as f:
        pickle.dump(resGLP, f)

# Load simulation results
with open(filename, 'rb') as f:
    resGLP = pickle.load(f)

# True parameters for comparison
true_params = {'beta': A, 'sigma': sigma, 'c': c, 'lambda': 0.2}

# Function to calculate credible intervals and MSE
def calc_credible_intervals_and_mse(param_values, true_value):
    mean_estimate = np.mean(param_values)
    lower_bound = np.percentile(param_values, 2.5)
    upper_bound = np.percentile(param_values, 97.5)
    mse = np.mean((param_values - true_value) ** 2)
    return mean_estimate, lower_bound, upper_bound, mse

# Store results in a table format for beta, sigma, and lambda
results_beta = []
results_sigma = []
results_lambda = []

# Posterior analysis for beta parameters
for i in range(3): # 3 rows, including one row for the constant: c_1, c_2
    for j in range(2): # 2 variables
        param_values = resGLP['mcmc']['beta'][i, j, :].flatten()
        if i == 0:
            true_value = c[j]
        else:
            true_value = A[i - 1, j]
        mean_estimate, lower_bound, upper_bound, mse = calc_credible_intervals_and_mse(param_values, true_value)
        results_beta.append([f'Beta ({i + 1},{j + 1})', true_value, mean_estimate, lower_bound, upper_bound, mse])

# Posterior analysis for sigma parameters
for i in range(2):
    for j in range(2):
        param_values = resGLP['mcmc']['sigma'][i, j, :].flatten()
        true_value = sigma[i, j]
        mean_estimate, lower_bound, upper_bound, mse = calc_credible_intervals_and_mse(param_values, true_value)
        results_sigma.append([f'Sigma ({i + 1},{j + 1})', true_value, mean_estimate, lower_bound, upper_bound, mse])

# Posterior analysis for lambda parameter
param_values = resGLP['mcmc']['lambda'].flatten()
mean_estimate, lower_bound, upper_bound, mse = calc_credible_intervals_and_mse(param_values, true_params['lambda'])
results_lambda.append(['Lambda', true_params['lambda'], mean_estimate, lower_bound, upper_bound, mse])

# Create Pandas DataFrames for beta, sigma, and lambda results
df_beta = pd.DataFrame(results_beta, columns=['Parameter', 'True Value', 'Posterior Mean', '2.5% Credible Interval', '97.5% Credible Interval', 'MSE'])
df_sigma = pd.DataFrame(results_sigma, columns=['Parameter', 'True Value', 'Posterior Mean', '2.5% Credible Interval', '97.5% Credible Interval', 'MSE'])
df_lambda = pd.DataFrame(results_lambda, columns=['Parameter', 'True Value', 'Posterior Mean', '2.5% Credible Interval', '97.5% Credible Interval', 'MSE'])

# Round all values to 4 decimal places
df_beta = df_beta.round(4)
df_sigma = df_sigma.round(4)
df_lambda = df_lambda.round(4)

# Save the results to CSV files
df_beta.to_excel(os.path.join(directory, 'beta_posterior_results.xlsx'), index=False)
df_sigma.to_excel(os.path.join(directory, 'sigma_posterior_results.xlsx'), index=False)
df_lambda.to_excel(os.path.join(directory, 'lambda_posterior_results.xlsx'), index=False)

print("Results saved to CSV files.")

# Visualization (Beta)
fig, axs = plt.subplots(3, 2, figsize=(18, 12))

bins = 50
for i in range(3):
    for j in range(2):
        param_values = resGLP['mcmc']['beta'][i, j, :].flatten()
        counts, bin_edges = np.histogram(param_values, bins=bins)
        total_counts = np.sum(counts)
        probabilities = counts / total_counts
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        axs[i, j].bar(bin_centers, probabilities, width=np.diff(bin_edges), alpha=0.6, color='skyblue')

        # Determine true value
        if i == 0:
            true_value = c[j]
        else:
            true_value = A[i - 1, j]

        # Plot the true value (red dashed line)
        axs[i, j].axvline(x=true_value, color='r', linestyle='--', linewidth=2, label=f'True Value: {true_value:.3f}')

        # Plot the median of the posterior distribution
        median_value = np.median(param_values)
        axs[i, j].axvline(x=median_value, color='g', linestyle='-', linewidth=2, label=f'Median: {median_value:.3f}')

        # Set the title for each subplot with LaTeX formatting
        param_name = f'$\\hat{{\\beta}}_{{{i + 1}{j + 1}}}$'  # LaTeX formatted variable name with a hat symbol
        axs[i, j].set_title(param_name, fontsize=16)

        # Add a legend
        axs[i, j].legend()
        # Add a larger legend with bold font
        legend = axs[i, j].legend(
            fontsize=16,                    # Increase font size of legend text
            loc='upper right',               # Keep legend inside the plot
            borderpad=1,                  # Increase padding between legend text and the border
            handlelength=1,                 # Increase the length of the legend key lines
            handletextpad=1,              # Increase spacing between legend key lines and text
        )
        # After creating the plot
        axs[i, j].set_ylabel('Probability Density', fontsize=17)
        # Set larger tick font sizes and fewer ticks if space is an issue
        axs[i, j].tick_params(axis='x', labelsize=18)
        axs[i, j].tick_params(axis='y', labelsize=18)
        axs[i, j].locator_params(axis='x', nbins=5)  # Fewer x-ticks
        axs[i, j].locator_params(axis='y', nbins=5)  # Fewer y-ticks

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig(os.path.join(directory, 'beta_distribution.png'))

print("Beta distribution plots saved with median and true value lines.")

# Visualization (Sigma)
fig, axs = plt.subplots(2, 2, figsize=(16, 8))

for i in range(2):
    for j in range(2):
        param_values = resGLP['mcmc']['sigma'][i, j, :].flatten()
        counts, bin_edges = np.histogram(param_values, bins=bins)
        total_counts = np.sum(counts)
        probabilities = counts / total_counts
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Plot histogram as bars
        axs[i, j].bar(bin_centers, probabilities, width=np.diff(bin_edges), alpha=0.6, color='lightgreen')

        # True value line
        true_value = sigma[i, j]
        axs[i, j].axvline(x=true_value, color='r', linestyle='--', label=f'True Value: {true_value:.3f}')

        # Median line
        median_value = np.median(param_values)
        axs[i, j].axvline(x=median_value, color='g', linestyle='-', label=f'Median: {median_value:.3f}')

        # Title with LaTeX formatting
        param_name = f'$\\hat{{\\sigma}}_{{{i + 1}{j + 1}}}$'  # LaTeX formatted variable name with a hat symbol
        axs[i, j].set_title(param_name, fontsize=16)

        # Add a legend
        axs[i, j].legend()
        legend = axs[i, j].legend(
            fontsize=16,  # Increase font size of legend text
            loc='upper right',  # Keep legend inside the plot
            borderpad=1,  # Increase padding between legend text and the border
            handlelength=1,  # Increase the length of the legend key lines
            handletextpad=0.5,  # Increase spacing between legend key lines and text
        )
        # After creating the plot
        axs[i, j].set_ylabel('Probability Density', fontsize=17)
        # Set larger tick font sizes and fewer ticks if space is an issue
        axs[i, j].tick_params(axis='x', labelsize=18)
        axs[i, j].tick_params(axis='y', labelsize=18)
        axs[i, j].locator_params(axis='x', nbins=5)  # Fewer x-ticks
        axs[i, j].locator_params(axis='y', nbins=5)  # Fewer y-ticks


# Adjust layout and save the figure
plt.tight_layout()
plt.savefig(os.path.join(directory, 'sigma_distribution.png'))

print("Sigma distribution plots saved with median and true value lines.")

# Visualization (Lambda)
plt.figure(figsize=(5, 4))

# Extract lambda parameter values from the MCMC results
param_values = resGLP['mcmc']['lambda'].flatten()

# Compute histogram
counts, bin_edges = np.histogram(param_values, bins=bins)
total_counts = np.sum(counts)
probabilities = counts / total_counts
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
plt.bar(bin_centers, probabilities, width=np.diff(bin_edges), alpha=0.6, color='salmon')
# Plot true value (red dashed line)
plt.axvline(x=true_params['lambda'], color='r', linestyle='--', label=f"True Value: {true_params['lambda']:.3f}")
# Plot median value (green solid line)
median_value = np.median(param_values)
plt.axvline(x=median_value, color='g', linestyle='-', label=f"Median: {median_value:.3f}")

# Title and axis labels
param_name = f'$\\hat{{\\lambda}}$'  # LaTeX formatted variable name with a hat symbol
plt.title(param_name, fontsize=12)
plt.xlabel('Value', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)  # Updated y-axis label to show probability density
# Adjust x-tick and y-tick font sizes and reduce the number of ticks
plt.tick_params(axis='x', labelsize=12)  # Increase font size for x-ticks
plt.tick_params(axis='y', labelsize=12)  # Increase font size for y-ticks
plt.locator_params(axis='x', nbins=5)    # Reduce the number of x-ticks
plt.locator_params(axis='y', nbins=5)    # Reduce the number of y-ticks
# Add legend
plt.legend(fontsize=12)

# Tight layout and save the figure
plt.tight_layout()
plt.savefig(os.path.join(directory, 'lambda_distribution.png'))

print("Lambda distribution plot saved with median and true value lines.")

#Function to plot trace plots
def plot_trace(param_samples, param_name, directory):
    plt.figure(figsize=(10, 4))
    plt.plot(param_samples, alpha=0.7)
    plt.title(f"Trace plot for {param_name}")
    plt.xlabel("MCMC Iteration")
    plt.ylabel(f"{param_name}")
    plt.grid(True, color='lightgrey', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(directory, f'trace_{param_name}.png'))
    #plt.close()

# Trace plots for beta parameters
for i in range(3):
    for j in range(2):
        param_values = resGLP['mcmc']['beta'][i, j, :].flatten()
        param_name = f'$\\hat{{\\beta}}_{{{i + 1}{j + 1}}}$'  # LaTeX formatted variable name with a hat symbol
        plot_trace(param_values, param_name, directory)

# Trace plots for sigma parameters
for i in range(2):
    for j in range(2):
        param_values = resGLP['mcmc']['sigma'][i, j, :].flatten()
        param_name = f'$\\hat{{\\sigma}}_{{{i + 1}{j + 1}}}$'  # LaTeX formatted variable name with a hat symbol
        plot_trace(param_values, param_name, directory)

# Trace plot for lambda parameter
param_values = resGLP['mcmc']['lambda'].flatten()
param_name = f'$\\hat{{\\lambda}}$'  # LaTeX formatted variable name with a hat symbol
plot_trace(param_values, param_name, directory)