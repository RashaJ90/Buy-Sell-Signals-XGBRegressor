from pathlib import Path
import os

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR.joinpath("data")))
RAW_DATA_DIR = DATA_DIR.joinpath('raw')



# Results Paths

# For data-split
TRAIN_RATIO = 0.85
VAL_RATIO = 0.15
TEST_RATIO = 0.0

# for data training
BATCH_SIZE = 16 
EPOCHS = 60
