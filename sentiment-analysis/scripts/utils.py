import torch
from torchtext.data.utils import get_tokenizer

class Processing:
    @staticmethod
    def load_vocab_library(vectors_path):
        """
        Load the pretrained word vectors
        """
        vocab = torch.load(vectors_path)

        return vocab

    @staticmethod
    def set_tokenizer():

        tokenizer = get_tokenizer("basic_english")

        return tokenizer

    @staticmethod
    def tokinize_text(text, tokinizer):
        """
        Tokenize the text
        """
        tokenized_text = tokinizer(text)

        return tokenized_text

    @staticmethod
    def map_tokens_to_indices(tokenized_text, vocab):
        """
        Map the tokens to their indices in the vocabulary
        """

        indices = [vocab[token] for token in tokenized_text]

        return indices

    @classmethod
    def preprocess_texts(cls, text, tokenizer, vocab):
        """
        Transform the text into tokens and map the tokens to indices in the vocabulary.
        """
        tokenized_text = Processing.tokinize_text(text, tokenizer)
        indices = Processing.map_tokens_to_indices(tokenized_text, vocab)

        input_tensor = torch.tensor(indices).unsqueeze(0)

        return input_tensor


