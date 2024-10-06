import nltk

nltk.download('punkt')
from nltk.tokenize import word_tokenize
from torch.nn import Embedding


def tokenize_processing(intput_text):
    '''
    gets input string as input and returns
    embeddings of each token as the output
    '''
    tokenize_input = word_tokenize(input_text)
    vocab = {}
    for i, token in enumerate(tokenize_input):
        if token not in vocab:
            vocab[token]=i
    return vocab , Embedding(num_embeddings=len(vocab), embedding_dim=256)


input_text = 'I have participated in multiple projects, delivering Machine Learning models that consistently achieve high precision and recall in production environments.'
target_text = 'Ich habe an mehreren Projekten mitgewirkt und Machine-Learning-Modelle bereitgestellt, die in Produktionsumgebungen durchgängig eine hohe Präzision und Trefferquote erreichen.'

vocab_input = tokenize_processing(input_text)
vocab_output = tokenize_processing(target_text)
