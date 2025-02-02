# covbayesvar: A Python Package for Large Bayesian VAR Models with COVID Volatility


![PyPI - Python Version](https://img.shields.io/pypi/pyversions/covbayesvar)
![PyPI Version](https://img.shields.io/pypi/v/covbayesvar)
[![Documentation Status](https://readthedocs.org/projects/large-bvar-with-covid-volatility/badge/?version=latest)](https://large-bvar-with-covid-volatility.readthedocs.io/en/latest/?badge=latest)
![Project Status](https://img.shields.io/badge/status-active-brightgreen)



`covbayesvar` is a Python package designed to estimate large Bayesian Vector Autoregressive (BVAR) models, incorporating COVID-induced volatility. 
It is a tool to forecast monthly and quarterly macroeconomic and financial data, leveraging advanced econometric methods to address 
the unique challenges posed by extreme observations and structural shifts during the pandemic. 

This package facilitates various tasks, including:
- **Estimating BVAR Models with COVID volatility**: Implements hierarchical priors to shrink parameters in high-dimensional systems and adjust for the heightened volatility and structural changes caused by the COVID-19 pandemic.
- **Forecasting**: Generates both unconditional (baseline), impulse response functions, and joint distribution of variables at any forecast horizon.
- **Scenario Analysis**: Explores the effects of hypothetical economic shocks on macroeconomic variables by constructing forecasts conditional on shocks and scenario analyses.
- **Entropic Tilting**: Anchoring forecasts to long-term policy targets, such as inflation or unemployment rates.
- **Versatility**: The package can be applied to diverse sets of monthly and quarterly datasets to answer policy-related questions, making it a valuable resource for researchers, policymakers, and financial analysts.
- **Extensive Documentation**: Includes detailed programming examples, documentation, and Google Colab notebooks to guide users through various use cases.


The methodology draws from seminal works in the field of macroeconometrics, 
including Giannone, Lenza, and Primiceri (2015), Banbura et. al. (2015), 
Crump et al. (2021), and Lenza and Primiceri (2022).


For a complete overview, refer to the following
- [Research paper and Google Colab jupyter notebook files](https://drive.google.com/drive/folders/1tKcULsaeg_ch-nMa-kWJ9D2VPIsYazwV)
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
- Assess the impact of supply, and demand, simulating the response of changes in any fiscal and monetary policy, etc
- Forecast inflation, unemployment, and other macroeconomic and financial indicators.
- Evaluate economic resilience under stress scenarios and extreme uncertainty.
- Examine the presence of structural breaks during and after COVID pandemic

---
## Ways to replicate the results of the research paper

To replicate the results presented in the accompanying research paper, you can 
choose between two primary methods: **running Jupyter notebooks in Google Colab** 
or **executing Python scripts locally**. 
Each method has its own advantages and limitations, which are outlined below to 
help you decide based on your preferences and computational resources.

### **1. Run the Jupyter Notebooks in Google Colab (Not Recommended for higher-frequency data)**

This method involves using pre-configured Jupyter notebooks that are optimized 
for Google Colab, a cloud-based environment that allows you to run Python code 
without setting up anything on your local machine.

**Pros:**
- User-Friendly Interface: don't need to set up an environment or install.
- Suitable for preliminary analyses or demonstrations.
- No Local Resource Dependency: Does not consume your local machine’s CPU or RAM.
- Pre-Configured Environment: Dependencies are often pre-installed or easy to install.
- Quick Prototyping: Ideal for testing small code snippets or analyzing quarterly-frequency datasets.

**Cons:**
- Slower Performance: Especially when running the model at the monthly level.
- Session Time Limits: Colab imposes usage limits, especially for free and Pro accounts.
- Limited Memory: May lead to out-of-memory errors with large Bayesian monthly VAR models.

**How to Use:**
- Open the provided Jupyter notebooks in Google Colab.
- Execute cells sequentially
---

### **2. Run Python Scripts Locally (Recommended method)**

This method involves cloning the repository to your local machine, 
setting up a virtual environment, and executing Python scripts found in 
the `scripts` folder.

**Pros:**
- Faster Execution: Leverages your local hardware, speeding computations, especially with monthly datasets.
- Persistent Environment: No risk of session timeouts as with Colab.
- Easier Debugging: easier to debug, modify code and unit-test functions.

**Cons:**
- Initial Setup Required: Requires cloning the repository, creating virtual environments, and installing dependencies.
- Hardware Limitations: Performance depends on your local machine’s CPU, RAM, and GPU capabilities.

**How to Use:**
- Follow the steps outlined below on how to proceed with running the python scripts locally.
 

## Cloning the Repository

To get started, clone the repository to your local machine:

```bash
git clone https://github.com/joshi27s/Large-BVAR-with-Covid-Volatility.git Large_BVAR
pwd # prints the present working directory to know where the Large_BVAR folder was cloned
cd Large_BVAR
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

The package requires the following Python packages:

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

### Using monthly data

In addition to the Google Colab notebooks, the **`scripts`** folder in the GitHub 
repository contains Python files to run the model on monthly data in the following order. 
To have a deeper understanding of the code, refer to the accompanying research paper,
and the official documentation of the functions. The package relies on functions
stored in the `large_bvar` module that you can find in the `covbayesvar` folder.
In particular, section 4, 5 and 6 elaborate the method and results obtained from the 
following scripts

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
   
```bash
# example of how to run the scripts via terminal/command prompt
python Descriptives.py
python main.py
python entropicTilting.py
```

### Using quarterly data

Navigate to the **`scripts/Quarterly`** folder and run the
following scripts that contain examples and results described in section 7 of
the research paper:

1. **`Descriptives.py`**

2. **`main_quarterly.py`**

3. **`entropicTilting.py`**

4. **`increaseGDP_longRun.py`**

5. **`increase1Y_unrestricted.py`**

6. **`increase1Y_Cholesky.py`**

7. **`financialTurmoil.py`**

8. **`structuralBreak.py`**

9. **`increaseAAA_restricted.py`**

10. **`increaseOilPrice_unrestricted.py`**


## Citation

If you use `covbayesvar` in your work, please cite the following research papers that inspired its methodology:

1. Banbura, M., Giannone, D., & Lenza, M. (2015). Conditional forecasts and scenario analysis with vector autoregressions for large cross-sections. *International Journal of Forecasting, 31*(3), 739–756. https://doi.org/10.1016/j.ijforecast.2014.08.013

2. Giannone, D., Lenza, M., & Primiceri, G. E. (2015). Prior selection for vector autoregressions. *The Review of Economics and Statistics, 97*(2), 436–451. https://doi.org/10.1162/REST_a_00483

3. Crump, R. K., Eusepi, S., Giannone, D., Qian, E., & Sbordone, A. M. (2021). A large Bayesian VAR of the United States economy. *NY Fed Staff Report.* https://www.newyorkfed.org/research/staff_reports/sr976

4. Lenza, M., & Primiceri, G. (2022). How to estimate a VAR after March 2020. *Journal of Applied Econometrics, 37*(4), 688–699.https://doi.org/10.1002/jae.2895


You can additionally cite the software repository itself:

```bibtex
@misc{joshi_sudiksha_2025_covbayesvar,
  author = {Sudiksha Joshi},
  title = {covbayesvar: A Python Package for Large Bayesian VAR Models with COVID Volatility},
  url = {https://github.com/joshi27s/Large-BVAR-with-Covid-Volatility},
  year = {2025}
}
```

## Contributions

You are welcome to contribute in any capacity to aid in expanding the project from creating an issue, reporting
a bug, suggesting an improvement, to forking the repository and creating a pull request to the development branch.

