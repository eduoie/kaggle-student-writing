from sagemaker.local import LocalSession
from sagemaker.pytorch import PyTorch


def main():

    sagemaker_session = LocalSession()
    sagemaker_session.config = {'local': {'local_code': True}}

    # For local training a dummy role will be sufficient
    role = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'

    print('Starting model training')
    print('Note: if launching for the first time in local mode, container image download might take a few minutes to complete.')
    estimator = PyTorch(entry_point='train_transformer_script.py',
                                source_dir='./src_transformers', # the source dir will be fully packaged as an artifact, so beware to choose a proper directory
                                role=role,
                                framework_version='1.9',
                                py_version='py38',
                                instance_count=1,
                                instance_type='local',
                                hyperparameters={
                                    'epochs': 1,
                                })

    estimator.fit('file://./processed_data_transformer/')


    # print('Deploying local mode endpoint')
    # predictor = estimator.deploy(initial_instance_count=1, instance_type='local')
    #
    # do_inference_on_local_endpoint(predictor, test_loader)
    #
    # predictor.delete_endpoint(predictor.endpoint)

if __name__ == "__main__":
    main()