import os
import sys
from datetime import datetime
import pandas as pd
from datetime import datetime
from time import strftime
import time
import re


input_data_path = '/opt/ml/processing/input_data/'
processed_data_path = '/opt/ml/processing/processed_data'


event_time_feature_name = "EventTime"


def cast_object_to_string(data_frame):
    for label in data_frame.columns:
        if data_frame.dtypes[label] == "object":
            data_frame[label] = data_frame[label].astype("str").astype("string")


def feature_engineer_split(train_df, timestamp, train_split, valid_split, test_split):
    train_df.discourse_id = train_df.discourse_id.astype(int).astype(str).astype('string')

    # This dataframe doesn't have the date type included, so we have to create it
    # Accepted date formats: yyyy-MM-dd'T'HH:mm:ssZ, yyyy-MM-dd'T'HH:mm:ss.SSSZ
    train_df[event_time_feature_name] = pd.Series([timestamp] * len(train_df), dtype="string")

    train_df.discourse_text = train_df.discourse_text.apply(lambda text: re.sub('[^A-Za-z0-9]+', ' ', text))

    cutoff = int(len(train_df) * (train_split))
    id_at_position = train_df.id[cutoff]
    count = 0
    while id_at_position == train_df.id[cutoff + count]:
        count += 1
    index_valid_df = cutoff + count

    cutoff = int(len(train_df) * (train_split + valid_split))
    id_at_position = train_df.id[cutoff]
    count = 0
    while id_at_position == train_df.id[cutoff + count]:
        count += 1
    index_test_df = cutoff + count

    valid_df = train_df[index_valid_df:index_test_df].copy()
    test_df = train_df[index_test_df:].copy()
    train_df = train_df[:index_valid_df].copy()

    train_df['split_type'] = 'train'
    valid_df['split_type'] = 'validation'
    test_df['split_type'] = 'test'

    cast_object_to_string(train_df)
    cast_object_to_string(valid_df)
    cast_object_to_string(test_df)

    return train_df, valid_df, test_df


def main():
    print("Processing Started")

    # Convert command line args into a map of args
    args_iter = iter(sys.argv[1:])
    args = dict(zip(args_iter, args_iter))

    print('Received arguments {}'.format(args))
    print('Reading input data from {}'.format(input_data_path))

    print("Got Args: {}".format(args))

    input_files = [file for file in os.listdir(input_data_path) if file.endswith('train.csv')]
    print('Available input text files: {}'.format(input_files))

    train_split = float(args['train-split-percentage'])
    valid_slit = float(args['validation-split-percentage'])
    test_split = float(args['test-split-percentage'])

    train_df = pd.read_csv(os.path.join(input_data_path, 'train.csv'))

    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    print(timestamp)

    train_df, valid_df, test_df = feature_engineer_split(train_df, timestamp, train_split, valid_slit, test_split)

    train_output_file = os.path.join(processed_data_path, 'train' + '.csv')
    valid_output_file = os.path.join(processed_data_path, 'valid' + '.csv')
    test_output_file = os.path.join(processed_data_path, 'test' + '.csv')

    train_df.to_csv(train_output_file)
    valid_df.to_csv(valid_output_file)
    test_df.to_csv(test_output_file)

    output_files = [file for file in os.listdir(processed_data_path) if file.endswith('.' + 'csv')]
    print('Available output text files: {}'.format(output_files))

    print("Processing Complete")


if __name__ == "__main__":
    main()