from reformat_pd import load_parquet
import pandas as pd
import numpy as np
import tensorflow as tf
import tflearn
import re

from collections import Counter
from sklearn.model_selection import train_test_split
from tflearn.data_utils import to_categorical
from nltk.stem.snowball import RussianStemmer
from nltk.tokenize import TweetTokenizer

HOME_DIR = '../'
VOCAB_SIZE = 5000


test_set = load_parquet(HOME_DIR + 'test.parquet')
train_set = load_parquet(HOME_DIR + 'train.parquet')

stemer = RussianStemmer()
regex = re.compile('[^а-яА-Я ]')
stem_cache = {}


def get_stem(token):
    stem = stem_cache.get(token, None)
    
    if stem:
        return stem
    
    token = regex.sub('', token).lower()
    stem = stemer.stem(token)
    stem_cache[token] = stem
    
    return stem


stem_count = Counter()
tokenizer = TweetTokenizer()


def count_unique_tokens_in_texts(texts):
    for _, text_series in texts.iterrows():
        text = text_series['description']
        tokens = tokenizer.tokenize(text)
        for token in tokens:
            stem = get_stem(token)
            stem_count[stem] += 1


count_unique_tokens_in_texts(train_set['text_fields'])

vocab = sorted(stem_count, key=stem_count.get, reverse=True)[:VOCAB_SIZE]
print(vocab[:100])

idx = 2
print(f'stem: {vocab[idx]}, count: {stem_count.get(vocab[idx])}')

token_2_idx = {vocab[i]: i for i in range(VOCAB_SIZE)}
len(token_2_idx)


def text_to_vector(text):
    vector = np.zeros(VOCAB_SIZE, dtype=np.int_)
    for token in tokenizer.tokenize(text):
        stem = get_stem(token)
        idx = token_2_idx.get(stem, None)

        if idx is not None:
            vector[idx] = 1

    return vector


text_vectors = np.zeros(
    (len(train_set), VOCAB_SIZE), dtype=np.int_)

texts = []
for i, (_, text) in enumerate(train_set.iterrows()):
    texts.append(text['text_fields']['description'])
    text_vectors[i] = text_to_vector(text['text_fields']['description'])

labels = np.zeros(len(train_set), dtype=np.int_)

X = text_vectors
y = to_categorical(labels, 2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


def build_model(learning_rate=0.1):
    tf.reset_default_graph()

    net = tflearn.input_data([None, VOCAB_SIZE])
    net = tflearn.fully_connected(net, 125, activation='ReLU')
    net = tflearn.fully_connected(net, 25, activation='ReLU')
    net = tflearn.fully_connected(net, 2, activation='softmax')
    regression = tflearn.regression(
        net,
        optimizer='sgd',
        learning_rate=learning_rate,
        loss='categorical_crossentropy')
    model = tflearn.DNN(net)

    return model


model = build_model(learning_rate=0.75)
model.fit(
    X_train,
    y_train,
    validation_set=0.1,
    show_metric=True,
    batch_size=128,
    n_epoch=30)


predictions = (np.array(model.predict(X_test))[:, 0] >= 0.5).astype(np.int_)
accuracy = np.mean(predictions == y_test[:, 0], axis=0)
print("Accuracy: ", accuracy)


def test_text(text):
    text_vector = text_to_vector(text, True)
    positive_prob = model.predict([text_vector])[0][1]
    print(f'Original text: {text}')
    print(f'P(positive) = {positive_prob}. Result:', 'Positive' if positive_prob > 0.5 else 'Negative')


def test_text_number(idx):
    test_text(texts[idx])


test_text_number(120705)
