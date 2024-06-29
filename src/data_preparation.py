import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_data(data_dir="data/imdb_reviews/aclImdb"):
    def load_texts_and_labels(dir_name):
        texts, labels = [], []
        for label_type in ["pos", "neg"]:
            dir_path = os.path.join(data_dir, dir_name, label_type)
            for fname in os.listdir(dir_path):
                if fname.endswith(".txt"):
                    with open(os.path.join(dir_path, fname), encoding="utf-8") as f:
                        texts.append(f.read())
                    labels.append(1 if label_type == "pos" else 0)
        return texts, labels

    train_texts, train_labels = load_texts_and_labels("train")
    test_texts, test_labels = load_texts_and_labels("test")

    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(train_texts)
    X_train = tokenizer.texts_to_sequences(train_texts)
    X_test = tokenizer.texts_to_sequences(test_texts)

    max_length = 500
    X_train = pad_sequences(X_train, maxlen=max_length)
    X_test = pad_sequences(X_test, maxlen=max_length)
    
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)

    word_index = tokenizer.word_index

    return X_train, X_test, y_train, y_test, word_index, max_length