
import pandas as pd


# Load the original CSV file
file_path = ('/Users/sjoshis/Desktop/covbayesvar/scripts/Data/current.csv')
df_csv = pd.read_csv(file_path)

# monthly data from FRED-MD
series_list = [

    ######## Output and Income ##############

    'INDPRO',  # Industrial Production: Total Index
    'CUMFNS',  # Capacity Utilization: Total Index
    'HOUST',  # New Privately-Owned Housing Units Started: Total Units
    'W875RX1',  # Real Personal Income ex. Current Transfer Receipts
    'DPCERA3M086SBEA',  # Real personal consumption expenditures (chain-type quantity index)
    'CMRMTSPLx', # Real Manufacturing and Trade Industries Sales

    ############ Labor Market ###############

    'PAYEMS',  # All employees, total nonfarm
    'UNRATE',  # Unemployment Rate
    'CES0600000008',  # Average Hourly Earnings of Production and Nonsupervisory Employees, Goods-Producing
    'CLAIMSx', # Initial Claims

    ########## Prices ###################

    'CPIAUCSL',  # CPI-Urban, All items
    'CPIULFSL',  # CPI-Urban: All Items Less Food
    'PCEPI',  # PCE: Chain-type Price Index
    'DSERRG3M086SBEA',  # PCE Services: Chain-Type Price Index
    'PPICMM',  # PPI by Commodity: Metals and Metal Products
    'OILPRICEx',  # Crude Oil, spliced WTI and Cushing

    ######### Interest Rates ##################

    'GS10',  # 10 Year Treasury Bond Yield,
    'GS1',  # 1 Year Treasury Bond Yield,
    'GS5',  # 5 Year Treasury Bond Yield
    'FEDFUNDS',  # Federal Funds Rate
    'AAA',  # Moody's Seasoned Aaa Corporate Bond Yield
    'BAA',  # Moody's Seasoned Baa Corporate Bond Yield
    'M2REAL',  # Real M2 Money Stock

    #########  Stock Market ##############

    'S&P 500',  # S&P 500 Index
    'VIXCLSx',  # CBOE Volatility Index: VIX
    
    'EXJPUSx',  # Japan / US Foreign Exchange Rate
    'EXUSUKx',  # US / UK Foreign Exchange Rate
    'EXCAUSx'  # Canada / US Foreign Exchange Rate
]

# Filter the columns based on the series list and keep the date column
selected_columns = ['sasdate'] + [col for col in series_list if col in df_csv.columns]
df_filtered = df_csv[selected_columns]
df_filtered = df_filtered.drop(0)
df_filtered['sasdate'] = pd.to_datetime(df_filtered['sasdate'])
df_filtered.rename(columns={'sasdate': 'Date'}, inplace=True)
df_filtered = df_filtered.dropna(subset=selected_columns[1:], how='any').reset_index(drop=True)
df_filtered_filled = df_filtered.ffill()
df_filtered_filled = df_filtered_filled.bfill()

# Python interpreted dates of 21st century (e.g., 2062 instead of 1962).
# Since this didn't match the expected historical context of the data,
# the code corrects these future dates by subtracting 100 years, effectively shifting them back to the 20th century.
current_year = pd.Timestamp.today().year
df_filtered_filled['Date'] = \
    df_filtered_filled['Date'].apply(lambda x: x - pd.DateOffset(years=100) if x.year > current_year else x)

output_path = 'Replication_Data.xlsx'
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    df_filtered_filled.to_excel(writer, sheet_name='Medium Data', index=False)