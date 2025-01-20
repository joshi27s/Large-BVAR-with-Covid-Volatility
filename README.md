# covbayesvar: A Python Package for Large Bayesian VAR Models with COVID Volatility


![PyPI - Python Version](https://img.shields.io/pypi/pyversions/covbayesvar)
![PyPI Version](https://img.shields.io/pypi/v/covbayesvar)
[![Documentation Status](https://readthedocs.org/projects/large-bvar-with-covid-volatility/badge/?version=latest)](https://large-bvar-with-covid-volatility.readthedocs.io/en/latest/?badge=latest)
![Project Status](https://img.shields.io/badge/status-active-brightgreen)



`covbayesvar` is a Python package designed to estimate large Bayesian Vector Autoregressive (BVAR) models, incorporating COVID-induced volatility. 
It is a tool to forecast macroeconomic and financial data, leveraging advanced econometric methods to address 
the unique challenges posed by extreme observations and structural shifts during the pandemic. 

This package facilitates various tasks, including:
- **Estimating BVAR Models with COVID volatility**: Implements hierarchical priors to shrink parameters in high-dimensional systems and adjust for the heightened volatility and structural changes caused by the COVID-19 pandemic.
- **Forecasting**: Generates both unconditional (baseline), impulse response functions, and joint distribution of variables at any forecast horizon.
- **Scenario Analysis**: Explores the effects of hypothetical economic shocks on macroeconomic variables by constructing forecasts conditional on shocks and scenario analyses.
- **Entropic Tilting**: Anchoring forecasts to long-term policy targets, such as inflation or unemployment rates.
- **Versatility**: The package can be applied to diverse sets of monthly and quraterly datasets to answer policy-related questions, making it a valuable resource for researchers, policymakers, and financial analysts.
- **Extensive Documentation**: Includes detailed programming examples, documentation, and Google Colab notebooks to guide users through various use cases.


The methodology draws from seminal works in the field of macroeconometrics, including Giannone, Lenza, and Primiceri (2015), Lenza and Primiceri (2021), and Crump et al. (2021).


For a complete overview, refer to the following
- [Research paper](https://drive.google.com/drive/folders/1tKcULsaeg_ch-nMa-kWJ9D2VPIsYazwV)
- [Official documentation](https://large-bvar-with-covid-volatility.readthedocs.io/en/latest/large_bvar.html#module-covbayesvar.large_bvar)
- [Github repository](https://github.com/joshi27s/Large-BVAR-with-Covid-Volatility/tree/main?tab=readme-ov-file)
- [`covbayesvar` package](https://pypi.org/project/covbayesvar/)

## Installation

You can install the package directly from PyPI using pip:

```bash
pip install covbayesvar
```



### Applications

Examples of questions this package and Google Colab python files can answer:
- Assessing the impact of supply, and demand, simulating the response of changes in any fiscal and monetary policy, etc
- Forecasting inflation, unemployment, and other macroeconomic and financial indicators.
- Evaluating economic resilience under stress scenarios and extreme uncertainty.

---

## Cloning the Repository

To get started, clone the repository to your local machine:

```bash
git clone https://github.com/joshi27s/Large-BVAR-with-Covid-Volatility.git
cd Large-BVAR-with-Covid-Volatility
```
## Installing from Source

If you'd like to install the latest version of the package from the source code, follow these steps:

1. Ensure you have Python 3.8 or higher installed.
2. Create a virtual environment and activate it:

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install the package and its dependencies:
```bash
pip install .
```

## Requirements

The package requires the following Python libraries:

- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `seaborn`
- `openpyxl`
- `plotly`

You can also install all dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```


## How to Execute the Model?

In addition to the Google Colab notebooks, the **`scripts`** folder in the GitHub repository contains Python files to run the model in the following order. For a deeper understanding of the code, refer to the accompanying research paper.

1. **`Descriptives.py`**:
   - Generates time series graphs of level and transformed data.
   - Transforms are applied only to variables not measured in percentage terms.

2. **`main.py`**:
   - Runs the BVAR model, including:
     - Unconditional (baseline) forecasts.
     - Conditional forecasts with shocks applied to two variables.
     - Scenario analysis and joint predictive densities.

3. **`Entropic_Tilting.py`**:
   - Anchors the long-run baseline forecasts to long-run targets mentioned in the Summary of Economic Projections (SEP) by the FOMC.

4. **`MCMC_Sims.py`** (optional):
   - Tests the performance of the BVAR model using a simple Monte Carlo simulation.


## Citation

If you use `covbayesvar` in your research, please cite the following works that inspired its methodology:

- Giannone, D., Lenza, M., & Primiceri, G. (2015). **Prior Selection for Vector Autoregressions**.
- Lenza, M., & Primiceri, G. (2021). **How to Estimate a VAR after March 2020**.
- Crump, R. K., Eusepi, S., Giannone, D., et al. (2021). **A Large Bayesian VAR of the United States Economy**.


You can additionally cite the software repository itself:

```bibtex
@misc{joshi_2025_covbayesvar_repo,
  author = {Sudiksha Joshi and contributors},
  title = {covbayesvar: A Python Package for Large Bayesian VAR Models with COVID Volatility},
  url = {https://github.com/joshi27s/Large-BVAR-with-Covid-Volatility},
  year = {2025}
}
```

## Contributions

You are welcome to contribute in any capacity to aid in expanding the project from creating an issue, reporting
a bug, suggesting an improvement, to forking the repository and creating a pull request to the development branch.

