from sagemaker.sklearn import SKLearn

DUMMY_IAM_ROLE = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'


def do_inference_on_local_endpoint(predictor, validation_input):
    pass


def main():

    print('Starting model training.')
    print('Note: if launching for the first time in local mode, container image download might take a few minutes to complete.')

    sklearn = SKLearn(
        entry_point="train_script.py",
        framework_version="0.23-1",
        instance_type="local",
        role=DUMMY_IAM_ROLE,
        hyperparameters={"alpha": 1.0, "fit_prior": True},
    )

    # accepts s3:// or file://
    train_input = "file://./processed_data/train.csv"
    valid_input = "file://./processed_data/valid.csv"

    sklearn.fit({"train": train_input},
                wait=True,
                logs=True)

    print('Completed model training')

    print('Deploying endpoint in local mode')
    # https://sagemaker.readthedocs.io/en/stable/api/inference/predictors.html
    predictor = sklearn.deploy(initial_instance_count=1, instance_type='local')

    # https://sagemaker.readthedocs.io/en/stable/api/inference/transformer.html
    # https://sagemaker-examples.readthedocs.io/en/latest/sagemaker_batch_transform/index.html
    # sklearn.transformer()

    do_inference_on_local_endpoint(predictor, valid_input)


    print('About to delete the endpoint to stop paying (if in cloud mode).')
    predictor.delete_endpoint(predictor.endpoint_name)


if __name__ == "__main__":
    main()