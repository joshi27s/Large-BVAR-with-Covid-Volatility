
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import matplotlib.ticker as ticker
import covbayesvar.large_bvar as bvar
# import large_bvar as bvar



sns.set_theme(style="white")
plot_dir = 'Descriptives Plots/'
# Load data from the specified path: change path as needed
path = '/Users/sjoshis/Desktop/covbayesvar/scripts/Data/Replication_Data.xlsx'

os.makedirs(plot_dir, exist_ok=True)
vis = 'on'  # Set to 'off' to hide figures

data = pd.read_excel(path, sheet_name="Medium Data")
Spec = pd.read_excel(path, sheet_name="Descriptive")
dates = data['Date'].values
data_array = data.drop(columns=['Date']).values
data_transformed = bvar.transform_data(Spec, data_array)
T, n = data_transformed.shape
print(Spec.iloc[:, 1:])
xl = [dates[0], dates[-1]]

for j_var in range(n):
    f, axs = plt.subplots(2, 1, figsize=(6, 3), dpi=100)

    # Untransformed
    axs[0].plot(dates, data_array[:, j_var], linewidth=1)
    axs[0].set_xlim(xl)
    axs[0].xaxis.set_major_locator(plt.matplotlib.dates.AutoDateLocator())
    axs[0].set_title(Spec.SeriesName[j_var], fontsize=5)
    axs[0].tick_params(axis='both', labelsize=5)
    axs[0].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m/%d/%y'))
    # Set the date format to mmm-yyyy (e.g., Jun-2023)
    axs[0].xaxis.set_major_locator(mdates.AutoDateLocator())
    axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    plt.setp(axs[0].get_xticklabels(), rotation=25, ha='right', fontsize=5) # rotate x-axis labels by 25 degree

    # Add grid lines
    axs[0].grid(True, which='both', linestyle='--', alpha=0.7)

    # Transformed
    axs[1].plot(dates, data_transformed[:, j_var], linewidth=1)
    axs[1].set_xlim(xl)
    axs[1].set_title(f"Transformed ({Spec.Transformation[j_var]})", fontsize=5)
    axs[1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    plt.setp(axs[1].get_xticklabels(), rotation=25, ha='right', fontsize=5)
    axs[1].grid(True, which='both', linestyle='--', alpha=0.7)
    axs[1].tick_params(axis='y', labelsize=5)
    axs[1].yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    plt.subplots_adjust(hspace=0.6)

    # Save figure
    plt.savefig(plot_dir + 'subplots_' + Spec.SeriesID[j_var] + '.png', dpi=600)
    plt.close()


################################### Interactive plots ###################################

# Calculate the total number of rows for subplots (each variable will have two rows)
total_rows = n * 2
subplot_height = 500  # Set the height for each subplot row
total_height = total_rows * subplot_height  # Total height for the figure
vertical_spacing = 0.005  # Adjust this value as needed

# Create a figure with multiple rows
fig = make_subplots(rows=total_rows, cols=1, shared_xaxes=True, vertical_spacing= vertical_spacing)

for j_var in range(n):
    row_orig = j_var * 2 + 1
    row_trans = j_var * 2 + 2

    # Add traces for original and transformed data
    fig.add_trace(go.Scatter(x=dates, y=data_array[:, j_var], mode='lines'), row=row_orig, col=1)
    fig.add_trace(go.Scatter(x=dates, y=data_transformed[:, j_var], mode='lines'), row=row_trans, col=1)

    # Add subplot titles using annotations
    fig.add_annotation(
        text=Spec.SeriesName[j_var],
        xref='paper', yref=f'y{row_orig}',
        x=0.5, y=1, showarrow=False,
        font=dict(size=10),
        xanchor='center', yanchor='bottom'
    )
    fig.add_annotation(
        text='Transformed (' + Spec.Transformation[j_var] + ')',
        xref='paper', yref=f'y{row_trans}',
        x=0.5, y=1, showarrow=False,
        font=dict(size=10),
        xanchor='center', yanchor='bottom'
    )

# Update the layout with the desired font sizes and ensure dates are displayed on x-axis
fig.update_layout(
    height=total_height,
    showlegend=False,
    title_text="Descriptives",
    title_font_size=20,
    font=dict(size=12),
)

# Adjust x-axis for dates
for j_var in range(n):
    fig.update_xaxes(tickformat='%d/%m/%Y', row=(j_var+1)*2, col=1)


# Save all subplots to a single HTML file
fig.write_html(f"{plot_dir}Descriptives.html")