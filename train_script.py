import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

import argparse
import os
import joblib


if __name__ == "__main__":
    print("Training Started")
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here. In this simple example we are just including one hyperparameter.
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--fit_prior", type=bool, default=True)

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    # https://github.com/aws/sagemaker-training-toolkit/blob/master/ENVIRONMENT_VARIABLES.md
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--num-cpus", type=str, default=os.environ["SM_NUM_CPUS"])

    args = parser.parse_args()
    print("Got Args: {}".format(args))

    train_df = pd.read_csv(os.path.join(args.train, 'train.csv'))

    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(train_df.discourse_type.to_numpy())

    # hyperparameters
    alpha = args.alpha
    fit_prior = args.fit_prior

    # Now use scikit-learn to train the model.
    clf = Pipeline([
        ("tf-idf", TfidfVectorizer()),
        ("clf", MultinomialNB(alpha=alpha, fit_prior=fit_prior))
    ])

    clf = clf.fit(train_df.discourse_text, train_labels_encoded)

    # Print the coefficients of the trained classifier, and save the coefficients
    joblib.dump(clf, os.path.join(args.model_dir, "model.joblib"))

    print("Training Completed")


def model_fn(model_dir):
    """Deserialized and return fitted model
    Note that this should have the same name as the serialized model in the main method
    """
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf