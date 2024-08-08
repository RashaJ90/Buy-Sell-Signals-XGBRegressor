# Buy/Sell Signals Prediction Using LSTM - XGBRegressor

Predicting Stock Returns and Generating Buy/Sell Signals with Machine Learning and Neural Networks.

## Intro

Predicting buy and sell signals in financial markets is essential for informed trading decisions, achieved by forecasting close prices and returns. This project utilizes machine learning models like XGBoost and LSTM to predict stock returns and generate trading signals using S&P 500 data from 2002 to 2023, including significant events like the 2007–2009 financial crisis and the COVID-19 pandemic.

* Feature engineering of technical indicators: trend identification, momentum assessment, volatility, and volume.
* Anomaly detection using Z-score as a feature.
* Walk Forward cross-validation using sktime.SlidingWindowSplitter.
* Results evaluation using Sharpe Ratio, Maximum Drawdown (MDD), and Treynor Ratio.

## Features:
* ATR: Average True Range
* Bollinger Band
* EMA: Exponential Moving Average
* MACD: Moving Average Convergence Divergence
* DMI: Directional Movement Index
* STOCH: Stochastic Oscillator
* RSI: Relative Strength Index
* Williams %R

## Environment Setup

I am using Micromamba, here's the installation link - https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html

Homebrew
```bash
brew install micromamba
```

If you want to quickly use micromamba in an ad-hoc usecase, you can run
```bash
export MAMBA_ROOT_PREFIX=/some/prefix  # optional, defaults to ~/micromamba
eval "$(./bin/micromamba shell hook -s posix)"
# Linux/bash:
./bin/micromamba shell init -s bash -p ~/micromamba  # this writes to your .bashrc file
# sourcing the bashrc file incorporates the changes into the running session.
# better yet, restart your terminal!
source ~/.bashrc

# macOS/zsh:
./micromamba shell init -s zsh -p ~/micromamba
source ~/.zshrc
```

Activate Micromamba 
```bash
micromamba activate  # this activates the base environment
```

clone the repository
```bash
https://github.com/RashaJ90/Buy-Sell-Signals-XGBRegressor.git
```

Create a conda environment after opening the repository
```bash
micromamba env create -f investor.yaml
```

```bash
micromamba activate Investor
```

Install the requirements
```bash
pip install -r requirements.txt
```

## Data
Install yfinance using this Link instructions -  https://pypi.org/project/yfinance/


## WorkFlow

This section is made to instruct user of how to train, predict, and see results in wandb / tensoboard.

## Project Results

This section is made to add results for the best performance  

