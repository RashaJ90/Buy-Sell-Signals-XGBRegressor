from pathlib import Path
import os

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
print(f"DEBUG: BASE DIR in settings.py: {BASE_DIR}")

DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR.joinpath("data")))

ANALYSIS_DIR = BASE_DIR.joinpath("analysis")
CONF_DIR = ANALYSIS_DIR.joinpath("conf")
COMMON_DIR = ANALYSIS_DIR.joinpath("common")
INVEST_DIR = ANALYSIS_DIR.joinpath("invest")
SRC_DIR = ANALYSIS_DIR.joinpath("src")

# Data directories
RAW_DATA_DIR = DATA_DIR.joinpath("raw")
PROCESSED_DATA_DIR = DATA_DIR.joinpath("processed")
TESTING_DATA_DIR = DATA_DIR.joinpath("testing")
TRAINING_DATA_DIR = DATA_DIR.joinpath("training")
DATA_COLUMNS = ["open", "high", "low", "close", "volume"]

DATA_SOURCE = ["yfinance", "quandl" ,"alpha_vantage", "fred"]

# RESULTS directories
RESULTS_DIR = BASE_DIR.joinpath("results")
MODELS_DIR = RESULTS_DIR.joinpath("models")
SHAP_DIR = RESULTS_DIR.joinpath("shap")