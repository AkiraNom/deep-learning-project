from datetime import datetime
import os

import torch

from utils import FinancialDataHandler
from model import PredictionModel

def main():
    """
    Main function to train and evaluate the prediction model
    """
    print("Starting prediction model training and evaluation with PyTorch")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Create directories
    model_save_path =f'./model/{timestamp}'

    os.makedirs(f'{model_save_path}', exist_ok=True)
    os.makedirs(f'{model_save_path}/data', exist_ok=True)
    os.makedirs(f'{model_save_path}/models', exist_ok=True)
    os.makedirs(f'{model_save_path}/results', exist_ok=True)
    os.makedirs(f'{model_save_path}/plots', exist_ok=True)

    # Optimized hyperparameters
    optimized_params = {
        'dropout_rate': 0.2,
        'learning_rate': 0.0005,
        'batch_size': 32,
        'hidden_size_1': 128,
        'hidden_size_2': 64,
        'hidden_size_3': 32
    }

    # name and ticker
    ticker_info = {'EURUSD': 'EURUSD=X'}

    # date range
    start_date = '2010-01-01'
    end_date = None

    # Fetch and preprocess data
    data_handler = FinancialDataHandler(
        ticker_info=ticker_info,
        start_date=start_date,
        end_date=end_date,
        window_size=30,
        train_split=0.8,
        model_save_path=model_save_path
    )

    processed_data = data_handler.fetch_and_preprocess_data()

    # Initialize model with optimized parameters
    model = PredictionModel(
        ticker_info,
        processed_data,
        start_date,
        end_date,
        device,
        model_save_path,
        optimized_params=optimized_params
    )

    # Build model
    model.build_model()

    # Train model
    model.train_model(epochs=100, patience=20)

    # Evaluate model
    model.evaluate_model()

    # Make predictions with uncertainty
    mean_predictions, std_predictions = model.predict_with_uncertainty(n_samples=100)

    # Visualize predictions
    model.visualize_predictions(mean_predictions, std_predictions)

    # Generate report
    model.generate_report()

    print(f"{list(ticker_info.keys())[0]} prediction model training and evaluation completed")
    print(f"Results and visualizations saved to {model_save_path}/ directory")



if __name__ == "__main__":
  main()
