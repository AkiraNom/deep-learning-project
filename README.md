# Deep Learning Project

## 1. CNN - CIFAR10 Image Classification Using Transfer Learning

The VGG11+ batch normalization model with the pre-trained parameters in the torchvision is used for CIFAR10 image classification task.

* Model architecture: VGG11 + batch normalization
* Parameter initialization: pre-trained VGG11 parameters

## 2. CNN - Oxford Flower Image Classifier Using Transfer LEarning.

 Flower Image Classifier is a machine learning model designed to predict the class of flower images.The pretrained MobileNetV3 model, along with its weights has been fine-tuned with data augmentation specifically for flower image classification. Impbalance data handling techniques have been implemented to improve the model's performance. The model has been trained on the Oxford 102 Flower Dataset. <br>

 [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](oxford-flower-image-classifier.streamlit.app)

## 3.1 RNN - EUR/USD Price Prediction (Time-Series Forecasting)

Built an single layer LSTM-based model to predict EUR/USD price trends, utilizing historical forex and S&P500 price data. The model captures temporal dependencies in financial data and evaluates performance using multiple metrics.

* **Model Architecture**: Single layer LSTM with 64 hidden units
* **Data Source**: Yahoo Finance (EUR/USD exchange rate, S&P 500 index)
* **Features**: Time-series returns, moving averages, and normalized price changes
* **Performance Metrics**: RMSE: 0.0421, MAE: 0.005676, Validation Loss: 0.0186

## 3.2 RNN - Forex Price Prediction (Time-Series Forecasting)

Designed a Forex price prediction model using a stacked LSTM architecture with attention mechanisms and Monte Carlo Dropout. The model forecasts prices for Forex and quantifies prediction confidence
* **Model Architecture** : 3-layer Stacked LSTM (128→64→32) + Attention Mechanism
* **Data Source**: Yahoo Finance API (historical prices)
* **Key Features**:
    * Feature engineering with Bollinger Bands, RSI, MACD
    * Monte Carlo Dropout for uncertainty estimation
    * Bayesian optimization for hyperparameter tuning
    * Predictive confidence interval visualization
* **Performance Metrics**: MSE: 0.0182, Directional Accuracy: 89.4%, Confidence Interval: ±2.1%


 ## 4. RNN - LSTM Sentiment Analysis

Developed a deep learning sentiment classifier using LSTM networks and Word2Vec embeddings to analyze customer sentiment. The model classifies textual data into Positive or Negative categories and is deployed as an interactive Streamlit application.

* **Model Architecture**: LSTM with pre-trained Word2Vec embeddings
* **Data Source**: Amazon Review Polarity Dataset (3.6M reviews)
* **Key Features**: Text preprocessing (tokenization, stopword removal), sequence padding
* **Performance Metrics**: 92.0% validation accuracy, F1 Score: 0.93, Precision: 0.97, Recall: 0.89


![](LSTM_model/sentiment-analysis/rsc/img/sentiment-analysis.gif)
