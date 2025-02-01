# Preliminaries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import covbayesvar.large_bvar as bvar
from PIL import Image

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
            'GDP Deflator',
            'S&P 500 Index',
            'CBOE Volatility Index: VIX',
            'University of Michigan: Consumer Sentiment'

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


def plot_scenario_analyses(DateAll, Spec, dYQQ, Shock, h_fore, plotDir, n, h, dots_per_inch, vis=True):
    """
    Plots scenario analyses based on specified variables and saves the plots to a directory.
    """
    # Setting plot styles with LaTeX
    plt.rc('text', usetex=True)
    plt.rc('legend', fontsize='medium')
    plt.rc('axes', labelsize='medium')
    iPlotStart = np.where(h_fore == 1)[0][0] - 2  # Index of start of plot
    iPlotEnd = np.where(h_fore == 1)[0][-1] -1   # Index of end of plot
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


def create_tiled_plots(plotDir):
    """
    Creates a tiled plot combining individual scenario plots.
    """
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
