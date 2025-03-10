# Deep Learning Project

## 1. CNN - CIFAR10 Image Classification Using Transfer Learning

The VGG11+ batch normalization model with the pre-trained parameters in the torchvision is used for CIFAR10 image classification task.

* Model architecture: VGG11 + batch normalization
* Parameter initialization: pre-trained VGG11 parameters

## 2. CNN - Oxford Flower Image Classifier Using Transfer LEarning.

 Flower Image Classifier is a machine learning model designed to predict the class of flower images.The pretrained MobileNetV3 model, along with its weights has been fine-tuned with data augmentation specifically for flower image classification. Impbalance data handling techniques have been implemented to improve the model's performance. The model has been trained on the Oxford 102 Flower Dataset. <br>

 [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](oxford-flower-image-classifier.streamlit.app)

## 3. RNN - LSTM Time Series Prediction

The LSTM model is used to predict a forex future price movement.

 * One feature input : EURUSD daily close price
 * Two feature inputs : EURUSD and SP500 daily close price
