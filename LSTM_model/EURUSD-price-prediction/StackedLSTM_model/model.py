from datetime import datetime
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score
import scipy.stats as stats
import seaborn as sns

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series data
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class AttentionLayer(nn.Module):
    """
    Attention mechanism for LSTM
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, hidden_size)
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        # attention_weights shape: (batch_size, seq_len, 1)

        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        # context_vector shape: (batch_size, hidden_size)

        return context_vector, attention_weights


class StackedLSTMModel(nn.Module):
    """
    Stacked LSTM model with attention mechanism
    """
    def __init__(self, input_size, hidden_size_1=128, hidden_size_2=64, hidden_size_3=32,
                 dropout_rate=0.2, output_size=1):

        super().__init__()

        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.hidden_size_3 = hidden_size_3
        self.dropout_rate = dropout_rate

        # First LSTM layer
        self.lstm1 = nn.LSTM(input_size, hidden_size_1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)

        # Second LSTM layer
        self.lstm2 = nn.LSTM(hidden_size_1, hidden_size_2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Attention mechanism
        self.attention = AttentionLayer(hidden_size_2)

        # Third LSTM layer
        self.lstm3 = nn.LSTM(hidden_size_2 * 2, hidden_size_3, batch_first=True)
        self.dropout3 = nn.Dropout(dropout_rate)

        # Output layer
        self.fc = nn.Linear(hidden_size_3, output_size)

    def forward(self, x, apply_dropout=False):
        # x shape: (batch_size, seq_len, input_size)

        # First LSTM layer
        lstm1_out, _ = self.lstm1(x)
        if apply_dropout:
            lstm1_out = self.dropout1(lstm1_out)
        else:
            lstm1_out = lstm1_out * (1 - self.dropout_rate)

        # Second LSTM layer
        lstm2_out, _ = self.lstm2(lstm1_out)
        if apply_dropout:
            lstm2_out = self.dropout2(lstm2_out)
        else:
            lstm2_out = lstm2_out * (1 - self.dropout_rate)

        # Attention mechanism
        context_vector, attention_weights = self.attention(lstm2_out)
        context_vector_expanded = context_vector.unsqueeze(1).expand(-1, lstm2_out.size(1), -1)
        combined = torch.cat((lstm2_out, context_vector_expanded), dim=2)

        # Third LSTM layer
        lstm3_out, _ = self.lstm3(combined)
        if apply_dropout:
            lstm3_out = self.dropout3(lstm3_out)
        else:
            lstm3_out = lstm3_out * (1 - self.dropout_rate)

        # ooutput at last time step
        last_time_step = lstm3_out[:, -1, :]

        # Output layer
        output = self.fc(last_time_step)

        return output, attention_weights


class PredictionModel:

    def __init__(self, ticker, data, start_date, end_date, device, save_model_path= '.',optimized_params=None):
        """
        Initialize the model

        Parameters:
        -----------
        ticker : dict
            Dictionary containing ticker name and ticker symbol (yfinance)
        device : torch.device
            Device to run the model on
        data : dict
            Dictionary containing preprocessed data
        date : tuple
            Tuple containing start and end dates
        save_model_path : str
            Path to save the model, defaults to current directory
        optimized_params : dict
            Dictionary containing model parameters
        """

        self.ticker_name = list(ticker.keys())[0]
        self.ticker = list(ticker.values())[0]
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.data = data
        self.device = device
        self.model_save_path = save_model_path

        # Initialize data containers
        self.model = None
        self.train_losses = []
        self.val_losses = []
        self.predictions = None
        self.metrics = None

        # Default hyperparameters
        self.params = {
            'dropout_rate': 0.2,
            'learning_rate': 0.0005,
            'batch_size': 32,
            'hidden_size_1': 128,
            'hidden_size_2': 64,
            'hidden_size_3': 32
        }

        # Update params with opitmized params if provided
        if optimized_params:
          self.params.update(optimized_params)

        print(f"Initialized Model with parameters: {self.params}")

    def build_model(self):
        """
        Build the stacked LSTM model with optimized hyperparameters

        Returns:
        --------
        model : StackedLSTMModel
            Built PyTorch model
        """
        if self.data is None:
            print("Data not available. Please run fetch_and_preprocess_data first.")
            return None

        input_size = self.data['X_train'].shape[2]
        print(f"Building stacked LSTM model with input size {input_size}")

        # Create model
        model = StackedLSTMModel(
            input_size=input_size,
            hidden_size_1=self.params['hidden_size_1'],
            hidden_size_2=self.params['hidden_size_2'],
            hidden_size_3=self.params['hidden_size_3'],
            dropout_rate=self.params['dropout_rate']
        ).to(self.device)

        # Print model summary
        print(model)
        print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

        self.model = model

        return model

    def train_model(self, epochs=100, patience=20):
        """
        Train the model with optimized hyperparameters

        Parameters:
        -----------
        epochs : int
            Maximum number of epochs
        patience : int
            Patience for early stopping

        Returns:
        --------
        model : StackedLSTMModel
            Trained PyTorch model
        """
        if self.model is None:
            print("Model not built. Please run build_model first.")
            return None

        if self.data is None:
            print("Data not available. Please run fetch_and_preprocess_data first.")
            return None

        print(f"Training model with {self.data['X_train'].shape[0]} samples")

        # Create data loaders
        train_dataset = TimeSeriesDataset(self.data['X_train'], self.data['y_train'])
        val_size = int(0.2 * len(train_dataset))
        train_size = len(train_dataset) - val_size

        train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_subset,
            batch_size=self.params['batch_size'],
            shuffle=True
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=self.params['batch_size'],
            shuffle=False
        )

        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.params['learning_rate'])

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )

        # Early stopping variables
        best_val_loss = float('inf')
        early_stopping_counter = 0
        best_model_state = None

        # Training loop
        for epoch in range(epochs):
            # Train mode
            self.model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()

                outputs, _ = self.model(batch_X, apply_dropout=True)

                loss = criterion(outputs, batch_y)

                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_X.size(0)

            train_loss = train_loss / len(train_loader.dataset)
            self.train_losses.append(train_loss)

            # Validation
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                    outputs, _ = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item() * batch_X.size(0)

            val_loss = val_loss / len(val_loader.dataset)
            self.val_losses.append(val_loss)

            scheduler.step(val_loss)

            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
                best_model_state = self.model.state_dict()

                # Save best model
                torch.save(
                    self.model.state_dict(),
                    os.path.join(f'{self.model_save_path}/models', "best_model.pth")
                )
                print(f"Model saved at epoch {epoch+1} with validation loss: {val_loss:.6f}")
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        # Save final model
        torch.save(
            self.model.state_dict(),
            os.path.join(f'{self.model_save_path}/models', "final_model.pth")
        )

        # Plot training history
        self._plot_training_history()

        return self.model

    def _plot_training_history(self):
        """
        Plot training history
        """
        if not self.train_losses or not self.val_losses:
            print("No training history available.")
            return

        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(f'{self.model_save_path}/plots', 'training_history.png'))
        plt.close()

    def evaluate_model(self):
        """
        Evaluate the model

        Returns:
        --------
        metrics : dict
            Dictionary of evaluation metrics
        """

        if self.model is None:
            print("Model not available. Please run build_model and train_model first.")
            return None

        if self.data is None:
            print("Data not available. Please run fetch_and_preprocess_data first.")
            return None

        print(f"Evaluating model on {self.data['X_test'].shape[0]} test samples")

        test_dataset = TimeSeriesDataset(self.data['X_test'], self.data['y_test'])
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.params['batch_size'],
            shuffle=False
        )

        # Evaluation
        self.model.eval()
        criterion = nn.MSELoss()
        mae_criterion = nn.L1Loss()

        test_loss = 0.0
        test_mae = 0.0
        preds_list = []
        targetst_list = []

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                outputs, _ = self.model(batch_X)

                loss = criterion(outputs, batch_y)
                mae = mae_criterion(outputs, batch_y)

                test_loss += loss.item() * batch_X.size(0)
                test_mae += mae.item() * batch_X.size(0)

                preds_list.extend(outputs.cpu().numpy())
                targetst_list.extend(batch_y.cpu().numpy())

        test_loss = test_loss / len(test_loader.dataset)

        preds_list = np.array(preds_list).flatten()
        targetst_list = np.array(targetst_list).flatten()

        self.predictions = preds_list

        # Calculate metrics
        test_mae = test_mae / len(test_loader.dataset)
        rmse = np.sqrt(test_loss)

        # Calculate direction accuracy
        direction_actual = np.diff(targetst_list)
        direction_pred = np.diff(preds_list)
        direction_accuracy = np.mean((direction_actual > 0) == (direction_pred > 0))

        # Calculate R-squared
        r2 = r2_score(targetst_list, preds_list)

        metrics = {
            'mse': float(test_loss),
            'mae': float(test_mae),
            'rmse': float(rmse),
            'direction_accuracy': float(direction_accuracy),
            'r2': float(r2)
        }

        self.metrics = metrics

        # Save metrics
        with open(os.path.join(f'{self.model_save_path}/results', 'evaluation_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)

        print(f"Evaluation metrics: {metrics}")

        self._residual_analysis()
        # self.test_residual_analysis()

        return metrics

    def predict_with_uncertainty(self, n_samples=100):
        """
        Make predictions with uncertainty estimation using Monte Carlo Dropout

        Parameters:
        -----------
        n_samples : int
            Number of Monte Carlo samples

        Returns:
        --------
        mean_predictions : numpy.ndarray
            Mean predicted values

        std_predictions : numpy.ndarray
            Standard deviation of predictions
        """

        if self.model is None:
            print("Model not available. Run build_model and train_model first.")
            return None, None

        if self.data is None:
            print("Data not available. Run fetch_and_preprocess_data first.")
            return None, None

        print(f"Making predictions with uncertainty for {self.data['X_test'].shape[0]} samples")

        X_test_tensor = torch.tensor(self.data['X_test'], dtype=torch.float32).to(self.device)

        self.model.train()  # Set to train mode to enable dropout

        # Monte Carlo sampling
        predictions = []

        with torch.no_grad():
            for i in range(n_samples):
                outputs, _ = self.model(X_test_tensor, apply_dropout=True)
                predictions.append(outputs.cpu().numpy())

                if (i+1) % 10 == 0:
                    print(f"Completed {i+1}/{n_samples} Monte Carlo samples")

        # Calculate mean and standard deviation
        predictions = np.array(predictions)
        mean_predictions = np.mean(predictions, axis=0)
        std_predictions = np.std(predictions, axis=0)

        # Save predictions
        np.save(os.path.join(f'{self.model_save_path}/results', 'mean_predictions.npy'), mean_predictions)
        np.save(os.path.join(f'{self.model_save_path}/results', 'std_predictions.npy'), std_predictions)

        return mean_predictions, std_predictions

    def visualize_predictions(self, mean_predictions=None, std_predictions=None):
        """
        Visualize model predictions

        Parameters:
        -----------
        mean_predictions : numpy.ndarray
            Mean predicted values from Monte Carlo Dropout

        std_predictions : numpy.ndarray
            Standard deviation of predictions from Monte Carlo Dropout
        """

        if self.data is None:
            print("Data not available. Please run fetch_and_preprocess_data first.")
            return

        if self.predictions is None and mean_predictions is None:
            print("No predictions available. Run evaluate_model or predict_with_uncertainty first.")
            return

        print("Visualizing predictions")

        test_dates = [datetime.strptime(d, '%Y-%m-%d') for d in self.data['metadata']['test_dates']]
        plt.figure(figsize=(12, 6))

        # Plot actual values
        plt.plot(test_dates, self.data['y_test'], label='Actual', color='blue')

        if mean_predictions is not None and std_predictions is not None:

            plt.plot(test_dates, mean_predictions, label='Predicted', color='red', linestyle='--')
            plt.fill_between(
                test_dates,
                mean_predictions.flatten() - 2 * std_predictions.flatten(),
                mean_predictions.flatten() + 2 * std_predictions.flatten(),
                color='red',
                alpha=0.2,
                label='95% Confidence Interval'
            )
        else:
            plt.plot(test_dates, self.predictions, label='Predicted', color='red', linestyle='--')

        plt.title(f'{self.ticker_name} - Forex Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Normalized Price')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)

        if len(test_dates) > 20:
            plt.xticks(test_dates[::len(test_dates)//10])

        plt.tight_layout()
        plt.savefig(os.path.join(f'{self.model_save_path}/plots', 'predictions.png'))
        plt.close()

        # Plot prediction error
        plt.figure(figsize=(12, 6))

        if mean_predictions is not None:
            prediction_error = mean_predictions.flatten() - self.data['y_test']
        else:
            prediction_error = self.predictions - self.data['y_test']

        plt.plot(test_dates, prediction_error, color='green')
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title('Prediction Error')
        plt.xlabel('Date')
        plt.ylabel('Error (Predicted - Actual)')
        plt.grid(True)
        plt.xticks(rotation=45)

        # If there are too many dates, show only a subset
        if len(test_dates) > 20:
            plt.xticks(test_dates[::len(test_dates)//10])

        plt.tight_layout()
        plt.savefig(os.path.join(f'{self.model_save_path}/plots', 'prediction_error.png'))
        plt.close()

    def _residual_analysis(self):
        """
        Perform residual analysis on the model predictions
        """

        if self.model is None:
            print("Model not available. Please run build_model and train_model first.")
            return None

        if self.data is None:
            print("Data not available. Please run fetch_and_preprocess_data first.")
            return None

        if self.predictions is None:
            print("No predictions available. Please run evaluate_model or predict_with_uncertainty first.")
            return None

        residuals = self.data['y_test'] - self.predictions
        mean_residuals = np.mean(residuals)
        sd_residuals = np.std(residuals)
        rmse = np.sqrt(np.mean(residuals**2))
        index = [datetime.strptime(d, '%Y-%m-%d') for d in self.data['metadata']['test_dates']]

        fig, ax = plt.subplots(1, 2, figsize=(15,8))

        sns.histplot(residuals, bins=50, ax=ax[0])
        ax[0].axvline(mean_residuals + sd_residuals, color='grey', linestyle='--', linewidth=2)
        ax[0].axvline(mean_residuals - sd_residuals, color='grey', linestyle='--', linewidth=2)
        ax[0].axvline(mean_residuals, color='black', linewidth=3)
        ax[0].text(x=mean_residuals, y=5, s= f'Mean: {mean_residuals: .2f}')
        ax[0].text(x=mean_residuals + sd_residuals, y=3, s= f'Std: {sd_residuals: .2f}')
        ax[0].set_title('Residual Distribution')
        ax[0].set_xlabel('Residuals')
        ax[0].set_ylabel('Counts')

        qq = stats.probplot(residuals, dist="norm", plot=None)
        ax[1].scatter(qq[0][0], qq[1][1] + qq[1][0]*qq[0][0], label='Fitted Line')
        ax[1].scatter(qq[0][0], qq[0][1], label='Predicted')
        ax[1].set_title('QQ plots')
        ax[1].set_xlabel('Theoretical quantiles')
        ax[1].set_ylabel('Ordered values')

        fig.suptitle("Residual Analysis")

        fig.tight_layout()
        fig.savefig(os.path.join(f'{self.model_save_path}/plots', 'residual_analysis.png'))
        plt.close()

    def generate_report(self):
        """
        Generate a comprehensive report of the model performance

        Returns:
        --------
        report : dict
            Dictionary containing the report
        """
        if self.data is None or self.model is None or self.metrics is None:
            print("Data, model, or metrics not available. Complete the training and evaluation first.")
            return None

        print("Generating comprehensive report")

        # Create report
        report = {
            'model_summary': {
                'ticker' : self.ticker_name,
                'ticker_symbol' : self.ticker,
                'input_shape': (self.data['X_train'].shape[1], self.data['X_train'].shape[2]),
                'hyperparameters': self.params,
                'training_samples': self.data['X_train'].shape[0],
                'test_samples': self.data['X_test'].shape[0]
            },
            'evaluation_metrics': self.metrics,
            'training_period': {
                'start_date': self.start_date,
                'end_date': self.end_date
            },
            'features_used': self.data['metadata']['features']
        }

        # Save report to file
        with open(os.path.join(f'{self.model_save_path}/results', 'model_report.json'), 'w') as f:
            json.dump(report, f, indent=4)

        print(f"Report generated and saved to {self.model_save_path}/results/model_report.json")
        return report
