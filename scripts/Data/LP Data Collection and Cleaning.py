from fredapi import Fred
from datetime import datetime
from pandas_datareader import DataReader

fred = Fred(api_key='write_your_api_key_here')
file_name = 'LP_data.xlsx'
file_path = '/Users/sjoshis/Desktop/covbayesvar/scripts/Data/'
full_path = f"{file_path}/{file_name}"

# monthly data from FRED-MD
series_list = [
    'CPIAUCSL',  # Consumer Price Index for All Urban Consumers: All Items
    'DDURRG3M086SBEA',  # Personal consumption expenditures: Durable goods (chain-type price index)
    'DGDSRG3M086SBEA',  # Personal consumption expenditures: goods (chain-type price index)
    'DNDGRG3M086SBEA',  # Personal consumption expenditures: Non-durable goods (chain-type price index)
    'DPCCRC1M027SBEA',  # Personal consumption expenditures excluding food and energy (Billions of dollars)
    'DSERRG3M086SBEA',  # Personal consumption expenditures: Services (chain-type price index)
    'INDPRO',  # Industrial Production Index, Index 2017=100
    'PAYEMS',  # All employees, total nonfarm, Thousands of Persons
    'PCE',  # Personal consumption expenditures, Billions of dollars
    'PCEDG',  # Personal Consumption Expenditures: Durable Goods
    'PCEND',  # Personal Consumption Expenditures: Non-Durable Goods
    'PCEPI',  # Personal consumption expenditures: Chain-type price index
    'PCEPILFE',  # Personal consumption expenditures: Chain-type price index excluding food and energy
    'PCES',  # Personal Consumption Expenditures: Services, Billions of Dollars
    'UNRATE',  # Civilian Unemployment Rate


]
data = DataReader(series_list, "fred" ,start=datetime(1947, 1, 1),
                  end=datetime(2025, 2, 1))
# Convert the index (DATE) to string format without time
#data.index = data.index.strftime('%Y-%m-%d')
# Saving data to Excel file
data.to_excel(full_path)
print(f"Data saved successfully at {full_path}")


