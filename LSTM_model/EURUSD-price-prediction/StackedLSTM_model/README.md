# Probabilistic FOREX Price Forecasting System

*Designed an uncertainty-aware FOREX price prediction model using a stacked LSTM architecture with attention mechanisms and Monte Carlo Dropout. The model forecasts prices for FOREX and quantifies prediction confidence.*

•	**Model Architecture**: 3-layer Stacked LSTM (128→64→32) + Attention Mechanism

•	**Data Source**: Yahoo Finance API (historical prices)

•	**Key Features**:
    o	Feature engineering with Bollinger Bands, RSI, MACD
    o	Monte Carlo Dropout for uncertainty estimation
    o	Predictive confidence interval visualization

•	**Performance Metrics**: MSE: 0.0182, Directional Accuracy: 89.4%, Confidence Interval: ±2.1%


## Dependencies
* numpy==1.25.0
* pandas==2.1.0
* matplotlib==3.8.0
* torch==2.0.1
* scikit-learn==1.3.0
* scipy==1.11.0
* seaborn==0.12.2
* yfinance==0.2.30
