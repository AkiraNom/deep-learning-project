import numpy as np

from gensim.models import KeyedVectors
import torch
from torch import nn

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, embedding_matrix=None, dropout=0.3, num_layers=1, bidirectional=False):

        super().__init__()
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1

        # embedding layer (vocab_size x embedding_dim)
        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False) # if freeze=True, no update
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        input_size = embedding_dim

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size*self.num_directions, output_size)

    def forward(self, x):
        x = self.embedding(x)
        output_seq, _ = self.rnn(x)
        output_seq = output_seq[:, -1, :]
        out = self.fc(output_seq)
        out = self.dropout(out)
        return nn.Sigmoid()(out)

    @staticmethod
    def load_model(PATH, vocab ,full_model=True):
        """
        Load a pre-trained model and change the classifier to the number of classes in the dataset.
        """
        if full_model:
            model = torch.load(PATH, weights_only=False, map_location=torch.device('cpu'))

        else:
            #model parameters
            num_classes = 1 # binary classes: 0 for negative, 1 for positive
            embedding_dim = 300
            hidden_size = 64
            output_size = num_classes
            num_layers = 2

            model = LSTM(len(vocab), embedding_dim, hidden_size, output_size, embedding_matrix=None, num_layers=num_layers, bidirectional=True)
            model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))

        return model

    @staticmethod
    def predict_class(model, text, device='cpu'):
        """
        Predict sentiment of the customer comments

        return the probabilities of the sentiment (Positive or Negative)
        """

        model.eval()

        with torch.no_grad():
            text = (text).to(device)
            prob = model(text)
            if prob > 0.5:
                return 1, prob
            else:
                return 0, prob

    @staticmethod
    def embedding_matrix(vocab, word2vec):
        """
        Create an embedding matrix from the word2vec model
        """

        unk_vectors = torch.from_numpy(np.mean(word2vec.vectors, axis=0))

        # create an embedding matrix
        embedding_matrix = torch.zeros(len(vocab), word2vec.vector_size)

        for idx, word in enumerate(vocab):
            if word in word2vec:
                embedding_matrix[idx] = torch.from_numpy(word2vec[word])
            else:
                embedding_matrix[idx] = unk_vectors

        return embedding_matrix


class Word2Vec:
    def __init__(self, model_path):

        self.word2vec = KeyedVectors.load_word2vec_format(model_path, binary=True)

