import json
import pandas as pd
import re


def load_parquet(file_path):
    df = pd.read_parquet(file_path, engine='fastparquet')
    descriptions = []

    for product_info in df['text_fields']:
        product_info = json.loads(product_info)
        product_info['description'] = re.sub('<.*?>', '', product_info['description'])

        descriptions.append(product_info)

    df['text_fields'] = descriptions

    print('Data parsing is done')

    return df
