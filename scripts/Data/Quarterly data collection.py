
import pandas as pd


# Load the original CSV file
file_path = ('/Users/sjoshis/Desktop/Fall 2022/ECON527 Macroeconometrics/BVAR of US Economy/Python '
             'Codes/Data/current_QD.csv')
df_csv = pd.read_csv(file_path)
# Remove the row where 'sasdate' equals 'transform'
df_csv = df_csv[df_csv['sasdate'] != 'transform']


# monthly data from FRED-MD
series_list = [

    ######## Output and Income ##############
    'GDPC1',  # Real Gross Domestic Product (Billions of Chained 2012 Dollars)
    'PCECC96',  # Real Personal Consumption Expenditures (Billions of Chained 2012 Dollars)
    'DPIC96',  # Real Disposable Personal Income (Billions of Chained 2012 Dollars)
    'PRFIx',  # Real Private Residential Fixed Investment (Billions of Chained 2012 Dollars)
    'PNFIx',  # Real Private Non-Residential Fixed Investment (Billions of Chained 2012 Dollars)
    'GCEC1', # Real Government Consumption Expenditures and Gross Investment
    'INDPRO',  # Industrial Production: Total Index
    'CUMFNS',  # Capacity Utilization: Total Index
    'HOUST',  # New Privately-Owned Housing Units Started: Total Units

    ############ Labor Market ###############

    'PAYEMS',  # All employees, total nonfarm
    'UNRATE',  # Unemployment Rate
    'RCPHBS',  # Business Sector: Real Compensation Per Hour (Index 2012=100)

    ########## Prices ###################
    'GDPCTPI', #  Gross Domestic Product: Chain-type Price Index (Index 2012=100) This is GDP Deflator
    'PCECTPI',  # PCE: Chain-type Price Index (Index 2012=100)
    'PCEPILFE',  # PCE Excluding Food and Energy (Index 2012=100)
    'CPIAUCSL',  # CPI-Urban, All items
    'CPIULFSL',  # CPI-Urban: All Items Less Food
    'OILPRICEx',  # Crude Oil, spliced WTI and Cushing

    ######### Interest Rates ##################

    'GS10',  # 10 Year Treasury Bond Yield,
    'GS1',  # 1 Year Treasury Bond Yield,
    'GS5',  # 5 Year Treasury Bond Yield
    'AAA',  # Moody's Seasoned Aaa Corporate Bond Yield
    'BAA',  # Moody's Seasoned Baa Corporate Bond Yield

    ############### International Trade ################
    'EXPGSC1',  # Real Exports of Goods and Services (Billions of Chained 2012 Dollars)
    'IMPGSC1',  # Real Imports of Goods and Services (Billions of Chained 2012 Dollars)

    #########  Stock Market ##############

    'S&P 500',  # S&P 500 Index
    'VIXCLSx',  # CBOE Volatility Index: VIX
    'UMCSENTx'  # University of Michigan: Consumer Sentiment  (Index 1st Quarter 1966=100)

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

output_path = 'Replication_Quarterly_Data.xlsx'
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    df_filtered_filled.to_excel(writer, sheet_name='Medium Data', index=False)