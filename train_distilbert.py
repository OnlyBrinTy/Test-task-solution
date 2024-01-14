import os

import torch
import numpy as np

from simpletransformers.classification import ClassificationModel
from sklearn import metrics

from file_utils import load_csv

model_output_path = 'models/distilb'
model_path = 'models'
data_path = 'data'

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main(train=True):
    print(f'Using {device}')
    # model_name 'sberbank-ai/ruBert-base'
    model_name = os.path.join(model_path, 'distilb')
    config = {
        'classifier_dropout': 0.3,
        'hidden_dropout_prob': 0.1,
        'attention_probs_dropout_prob': 0.1,
        'num_attention_heads': 12,
        'num_hidden_layers': 12
    }

    args = {
        'config': config,
        'manual_seed': 0,
        'process_count': 1,
        'num_train_epochs': 3,
        'batch_size': 32,
        'learning_rate': 1e-5,
        'regression': False,
        'fp16': False,
        'use_multiprocessing': False,
        'use_multiprocessing_for_evaluation': False,
        'overwrite_output_dir': True,
        'save_steps': 0,
        'save_model_every_epoch': False,
    }

    train_df = load_csv(os.path.join(data_path, 'pro_train.csv'))
    test_df = load_csv(os.path.join(data_path, f'pro_test.csv'))

    ar = np.array(list(train_df['category_id']) + list(test_df['category_id']))
    labels = np.unique(ar).tolist()
    reverse_index = {labels[i]: i for i in range(len(labels))}

    # add labels to dfs
    train_df['labels'] = train_df.apply(lambda row: reverse_index[row['category_id']], axis=1)
    test_df['labels'] = test_df.apply(lambda row: reverse_index[row['category_id']], axis=1)

    model = ClassificationModel(
        # "bert",
        "distilbert",
        model_name,
        use_cuda=(device == 'cuda'),
        num_labels=len(labels),
        args=args,
    )

    if train:
        model.train_model(
            train_df,
            output_dir=model_output_path,
            eval_df=test_df
        )

        model.save_model(output_dir=model_output_path, model=model.model)


if __name__ == '__main__':
    main()
