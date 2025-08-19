import argparse
import sys
import os
import logging

# Add the project root to the Python path to enable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from analysis.src.models.lstm import run_lstm_pipeline
from analysis.src.models.xgboost import run_xgboost_pipeline

logger = logging.getLogger(__name__)

def train(model_type: str, **kwargs):
    """
    Main training function to select and run a model training pipeline.

    Args:
        model_type (str): The type of model to train (e.g., 'lstm', 'xgboost').
        **kwargs: Model-specific hyperparameters passed from the command line.
    """
    logger.info(f"--- Starting Training for {model_type.upper()} Model ---")

    if model_type.lower() == 'lstm':
        run_lstm_pipeline( # change into json
            hidden_size=kwargs.get('hidden_size', 50),
            num_layers=kwargs.get('num_layers', 2),
            num_epochs=kwargs.get('num_epochs', 20),
            lr=kwargs.get('lr', 0.001)
        )
    elif model_type.lower() == 'xgboost':
        run_xgboost_pipeline(# change into json
            n_estimators=kwargs.get('n_estimators', 1000),
            max_depth=kwargs.get('max_depth', 5),
            learning_rate=kwargs.get('lr', 0.05)
        )
    else:
        logger.error(f"Error: Model type '{model_type}' is not supported.")
        logger.error("Supported models are: 'lstm', 'xgboost'")

def main():

    parser = argparse.ArgumentParser(description="Train a specified model.")
    parser.add_argument(
        'model_type',
        type=str,
        choices=['lstm', 'xgboost'],
        help="The type of model to train."
    )
    # LSTM arguments
    parser.add_argument('--hidden_size', type=int, default=50, help='LSTM: Number of features in the hidden state.')
    parser.add_argument('--num_layers', type=int, default=2, help='LSTM: Number of recurrent layers.')
    parser.add_argument('--num_epochs', type=int, default=20, help='LSTM: Number of training epochs.')

    # XGBoost arguments
    parser.add_argument('--n_estimators', type=int, default=1000, help='XGBoost: Number of boosting rounds.')
    parser.add_argument('--max_depth', type=int, default=5, help='XGBoost: Maximum tree depth.')

    # Shared arguments
    parser.add_argument('--lr', type=float, help='Learning rate. Default is 0.001 for LSTM, 0.05 for XGBoost.')

    args = parser.parse_args()

    # Set default learning rate if not provided, based on model type
    if args.lr is None:
        args.lr = 0.001 if args.model_type == 'lstm' else 0.05
        logger.info(f"Learning rate not specified. Using default for {args.model_type}: {args.lr}")

    train(args.model_type, **vars(args))

if __name__ == "__main__":
    main()
