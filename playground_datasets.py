import json
import re

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

from datasets import load_dataset, Dataset

SEPARATOR = '</s>'

"""
    Before starting create folders: data, models.
    Inside data create folders text_embs and image_embs. 
    Also add folders test and train in each of them. 
"""


def parquets_to_csvs():
    train_parquet = pd.read_parquet('../train.parquet')
    test_parquet = pd.read_parquet('../test.parquet')

    train_parquet.to_csv('data/train.csv')
    test_parquet.to_csv('data/test.csv')


def remove_html(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    
    return cleantext


def flatten_text_fields(values: dict) -> str:
    result = [values['title']]

    description = values['description']
    soup = BeautifulSoup(description, "html.parser")
    elements = list(soup.find_all("p"))

    if len(elements) == 0:
        result.append(remove_html(description))
    else:
        for child in elements:
            block_text = child.text.strip()

            if 'http' in block_text or not block_text:
                continue

            result.append(block_text)

    result.extend(values['attributes'])

    if values['filters']:
        result.append(', '.join(values['filters'].keys()))

    return SEPARATOR.join(result)


def convert_text_fields(values: list) -> list:
    result = []
    for value in tqdm(values):
        json_value = json.loads(value)
        flatten_text = flatten_text_fields(json_value)
        result.append(flatten_text)
    return result


def main():
    parquets_to_csvs()

    df = pd.read_csv('data/train.csv')
    texts = convert_text_fields(df.text_fields)
    df['text'] = texts

    df.to_csv('data/full.csv')
    pivot = np.random.rand(len(df)) < 0.95
    train = df[pivot]
    test = df[~pivot]

    train.to_csv('data/pro_train.csv')
    test.to_csv('data/pro_test.csv')


if __name__ == "__main__":
    main()
