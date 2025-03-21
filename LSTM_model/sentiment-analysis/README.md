## Sentiment Analysis Model

This project implements a Sentiment Analysis model using Long Short Term Memory (LSTM) networks and Word2Vec embeddings. It is desinged for classifying texutal data, such as customer reviews, into binary sentiment classes (e.g. Positive or Negative).<br>

### Features

* **Model Architecture**: The LSTM module is built using Pytorch and includes customizable options for:
    * Bidirectional LSTMs for enhanced contextual understanding
    * Dropout layers to mitigate overfitting

* **WordEmbedding**: Utilize Pre-trained Word2Vec embeddings (GoogleNews-vectors-negative300.bin) for better language understanding

### Model Characteristics

* Model Architecture: LSTM
* Framework: PyTorch
* Training Data: Amazon Review Polarity Dataset
* Number of Classes: 2 (Positive, Negative)
* Optimizer: Adam
* Learning Rate: Adjusted dynamically using CosineAnnealingLR
* Loss Function: Binary Crossentropy

### Dataset
The model is trained and evaluated on the Amazon Review Polarity Dataset, which contains labeled customer reviews divided into two sentiment classes.

### Dependencies
 * Python
 * Numpy
 * PyTorch
 * Gensim
