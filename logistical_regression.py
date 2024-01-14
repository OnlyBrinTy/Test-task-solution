import os
import random
import pickle

import joblib
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import pandas as pd
import numpy as np


def get_dataset(df, df_type: str, dataset_size=float('inf')) -> tuple:
    # filename to save dataset
    if df_type == 'train':
        x_cache_filename = f'data/x_train_cache_{dataset_size}.npy'
    elif df_type == 'test':
        x_cache_filename = f'data/x_test_cache_{dataset_size}.npy'

    if dataset_size >= len(df):
        dataset_size = 'full'
    else:
        df = df.sample(dataset_size, random_state=0)

    if not os.path.exists(x_cache_filename):
        img_datapoints = []
        text_datapoints = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            product_id = row['product_id']
            image_emb = np.load(f'data/image_embs/{df_type}/{product_id}.jpg.npy')
            text_emb = np.load(f'data/text_embs/{df_type}/{product_id}.npy')
            img_datapoints.append(image_emb)
            text_datapoints.append(text_emb)

        X_img = np.stack(img_datapoints)
        X_text = np.stack(text_datapoints)
        scaler = preprocessing.StandardScaler().fit(X_img)
        X_img_scaled = scaler.transform(X_img)

        X = np.concatenate((X_text, X_img_scaled), axis=1)

        np.save(x_cache_filename, X)

        return X, df, dataset_size

    return np.load(x_cache_filename), df, dataset_size


def train():
    df_train = pd.read_csv('data/train.csv')
    X, df_train, ds_size = get_dataset(df_train, 'train')

    categories = list(df_train.category_id)
    labels = np.unique(df_train.category_id).tolist()
    reverse_index = {labels[i]: i for i in range(len(labels))}

    y = np.array([reverse_index[cat] for cat in categories])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

    model = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        penalty='l2',
        C=1,
        max_iter=1000,
        verbose=1
    )

    model.fit(X_train, y_train)

    base_filename = f'models/logreg_{ds_size}'
    pickle.dump(model, open(f'{base_filename}.pkl', 'wb'))

    score = model.score(X_test, y_test)
    print(score)


def predict():
    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')

    X, df_test, ds_size = get_dataset(df_test, 'test')
    labels = np.unique(df_train.category_id).tolist()

    with open(f"models/logreg_{ds_size}.pkl", 'rb') as file:
        model = pickle.load(file)

    predictions = model.predict(X)

    category_ids = [labels[idx] for idx in predictions]
    result_df = pd.DataFrame(
        {
            "product_id": df_test["product_id"],
            "predicted_category_id": category_ids
        }
    )
    result_df.to_parquet(f'../result.parquet')


if __name__ == "__main__":
    # train()
    predict()
