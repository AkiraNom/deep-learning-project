from datetime import datetime
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import yfinance as yf
import pickle

np.random.seed(42)
# torch.manual_seed(42)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(42)

class FinancialDataHandler:
    def __init__(self,
                 ticker_info={'EURUSD':'EURUSD=X'},
                 start_date='2020-01-01',
                 end_date=None,
                 window_size=30,
                 train_split=0.8,
                 model_save_path='.'):
        """
        Initialize the model

        Parameters:
        -----------
        ticker : dict
            Dictionary containing ticker name and ticker symbol
        start_date : str
            Start date for data collection (YYYY-MM-DD)
        end_date : str
            End date for data collection (YYYY-MM-DD), defaults to current date
        window_size : int
            Size of the input window (how many past days to use)
        train_split : float
            Proportion of data to use for training, defaults to 0.8
        model_save_path : str
            Path to save the model, defaults to current directory

        """
        self.ticker_name = list(ticker_info.keys())[0]
        self.ticker = list(ticker_info.values())[0]
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.window_size = window_size
        self.train_split = train_split
        self.model_save_path = model_save_path

        # data containers
        self.raw_data = None
        self.data = None

    def fetch_data(self):
        """
        Fetch data from Yahoo Finance and save the raw data to a CSV file

        Returns:
        --------
        pd.DataFrame
            Raw data
        """
        print(f"Fetching {self.ticker_name} data from {self.start_date} to {self.end_date}")

        try:
            # Yahoo Finance ticker for EUR/USD
            self.raw_data = yf.download(
                self.ticker,
                start=self.start_date,
                end=self.end_date,
                interval="1d",
                progress=False
            )

            if self.raw_data.empty:
                print(f"No data found for {self.ticker}")
                return None

            # Save raw data to CSV
            csv_path = os.path.join(f'{self.model_save_path}/data', f"{self.ticker}_raw.csv")
            self.raw_data.to_csv(csv_path)
            print(f"Raw data saved to {csv_path}")

            return self.raw_data

        except Exception as e:
            print(f"Error fetching {self.ticker} data: {e}")
            return None

    def fetch_and_preprocess_data(self):
        """
        Fetch data and calculate technical indicators, then preprocess the data

        Returns:
        --------
        pd.DataFrame
            Preprocessed data
        """

        # Fetch data
        data = self.fetch_data()

        # Calculate technical indicators
        df = self._calculate_technical_indicators(data)

        # Preprocess data
        processed_data = self._preprocess_data(df)

        self.data = processed_data

        return processed_data

    def _calculate_technical_indicators(self, df):
        """
        Calculate technical indicators

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing forex data

        Returns:
        --------
        pd.DataFrame
            DataFrame with added technical indicators
        """
        print("Calculating technical indicators")

        # Make a copy to avoid modifying the original dataframe
        df = df.copy()

        # Moving averages
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()

        # Bollinger Bands (20-day, 2 standard deviations)
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        df['BB_std'] = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
        df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']

        # RSI (Relative Strength Index) - 14 days
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD (Moving Average Convergence Divergence)
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']

        # Price changes
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_5d'] = df['Close'].pct_change(periods=5)

        # Volatility indicators
        df['HL_Diff'] = df['High'] - df['Low']
        df['HL_Diff_Pct'] = (df['High'] - df['Low']) / df['Low']

        # ATR (Average True Range) - 14 days
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.rolling(14).mean()

        # Drop NaN values
        df = df.dropna()

        print("Technical indicators calculated")

        return df

    def _preprocess_data(self, df):
        """
        Preprocess data for LSTM model

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing forex data with technical indicators

        Returns:
        --------
        dict
            Dictionary containing preprocessed data
        """
        print("Preprocessing data")

        # Select features
        features = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'MA5', 'MA10', 'MA20',
            'BB_upper', 'BB_middle', 'BB_lower',
            'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
            'Price_Change', 'Price_Change_5d',
            'HL_Diff', 'HL_Diff_Pct', 'ATR'
        ]

        # Scale the features
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df[features])
        scaled_df = pd.DataFrame(scaled_data, index=df.index, columns=features)

        # Create sequences for LSTM
        X, y = [], []
        for i in range(len(scaled_df) - self.window_size):
            X.append(scaled_df.iloc[i:i+self.window_size].values)

            # target: close price after the window_size
            close_idx = features.index('Close')
            y.append(scaled_df.iloc[i+self.window_size, close_idx])

        X = np.array(X)
        y = np.array(y)

        # Split into train and test sets
        split_idx = int(len(X) * self.train_split)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Save dates for visualization
        dates = df.index[self.window_size:].tolist()
        train_dates = dates[:split_idx]
        test_dates = dates[split_idx:]

        # Save preprocessed data
        np.save(os.path.join(f'{self.model_save_path}/data', "X_train.npy"), X_train)
        np.save(os.path.join(f'{self.model_save_path}/data', "y_train.npy"), y_train)
        np.save(os.path.join(f'{self.model_save_path}/data', "X_test.npy"), X_test)
        np.save(os.path.join(f'{self.model_save_path}/data', "y_test.npy"), y_test)

        # Save metadata
        metadata = {
            'features': features,
            'close_idx': close_idx,
            'train_dates': [d.strftime('%Y-%m-%d') for d in train_dates],
            'test_dates': [d.strftime('%Y-%m-%d') for d in test_dates]
        }

        with open(os.path.join(f'{self.model_save_path}/data', "metadata.pth"), 'wb') as f:
            pickle.dump(metadata, f)

        with open(os.path.join(f'{self.model_save_path}/data', "scaler.pth"), 'wb') as f:
            pickle.dump(scaler, f)

        print(f"Preprocessed data saved. X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'metadata': metadata,
            'scaler': scaler,
            'raw_df': df
        }
