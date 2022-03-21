import os
import sys

import numpy as np
import pandas as pd

input_data_path = '/opt/ml/processing/input_data/'
processed_data_path = '/opt/ml/processing/processed_data'


def feature_engineer_split(train_df, train_text_df, train_split, valid_split, random_seed):

    all_entities = []
    for ii, i in enumerate(train_text_df.iterrows()):
        if ii%100==0: print(ii,', ',end='')
        total = i[1]['text'].split().__len__()
        entities = ["O"] * total
        for j in train_df[train_df['id'] == i[1]['id']].iterrows():
            discourse = j[1]['discourse_type']
            list_ix = [int(x) for x in j[1]['predictionstring'].split(' ')]
            entities[list_ix[0]] = f"B-{discourse}"
            for k in list_ix[1:]: entities[k] = f"I-{discourse}"
        all_entities.append(entities)
    train_text_df['entities'] = all_entities

    print(train_text_df.shape)
    print(train_text_df.head())

    IDS = train_df.id.unique()

    np.random.seed(random_seed)
    train_idx = np.random.choice(np.arange(len(IDS)), int(train_split * len(IDS)), replace=False)
    valid_idx = np.setdiff1d(np.arange(len(IDS)), train_idx)

    data = train_text_df[['id', 'text', 'entities']]
    train_dataset = data.loc[data['id'].isin(IDS[train_idx]), ['text', 'entities']].reset_index(drop=True)
    valid_dataset = data.loc[data['id'].isin(IDS[valid_idx])].reset_index(drop=True)

    print("FULL Dataset: {}".format(data.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(valid_dataset.shape))

    return train_dataset, valid_dataset


def main():
    print("Processing Started")

    # Convert command line args into a map of args
    args_iter = iter(sys.argv[1:])
    args = dict(zip(args_iter, args_iter))

    print('Received arguments {}'.format(args))
    print('Reading input data from {}'.format(input_data_path))

    input_files = [file for file in os.listdir(input_data_path) if file.endswith('.csv')]
    print('Available input text files: {}'.format(input_files))

    train_split = float(args['train-split-percentage'])
    valid_slit = float(args['validation-split-percentage'])
    random_seed = int(args['random-seed'])

    train_df = pd.read_csv(os.path.join(input_data_path, 'train.csv'))
    train_text_df = pd.read_csv(os.path.join(input_data_path, 'train_text_df.csv'))

    train_df_ner, valid_df_ner = feature_engineer_split(train_df, train_text_df, train_split, valid_slit, random_seed)

    train_output_file = os.path.join(processed_data_path, 'train_NER' + '.csv')
    valid_output_file = os.path.join(processed_data_path, 'valid_NER' + '.csv')
    train_df_ner.to_csv(train_output_file)
    valid_df_ner.to_csv(valid_output_file)

    output_files = [file for file in os.listdir(processed_data_path) if file.endswith('.' + 'csv')]
    print('Available output text files: {}'.format(output_files))

    print("Processing Complete")


if __name__ == "__main__":
    main()