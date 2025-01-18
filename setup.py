from setuptools import setup, find_packages

setup(
    name="covbayesvar",
    version="0.1.8",
    description="This package has functions to estimate large BVAR models with covid volatility, plot conditional and "
                "unconditional forecasts, scenario analyses, and impulse response functions, and joint distribution of "
                "forecasts using the methods established in Giannone et al. (2015), Lenza and Primiceri (2021), "
                "and Crump et. al (2021).",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Sudiksha Joshi",
    author_email="joshi27s@uw.edu",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3"
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "matplotlib"
    ]
)
