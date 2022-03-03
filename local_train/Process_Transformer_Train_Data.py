from sagemaker.local import LocalSession
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor

sagemaker_session = LocalSession()
sagemaker_session.config = {'local': {'local_code': True}}

# For local training a dummy role will be sufficient
role = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'

MODEL_NAME = 'google/bigbird-roberta-base'
train_split_percentage = 0.9
validation_split_percentage = 0.1
random_seed = 42

processor = SKLearnProcessor(framework_version='0.23-1',
                             instance_count=1,
                             instance_type='local',
                             role=role,
                             )

print('Starting processing job.')
processor.run(code='processing_transformer_data_script.py',
              inputs=[ProcessingInput(
                  source='./input_data/',
                  destination='/opt/ml/processing/input_data/')],
              outputs=[ProcessingOutput(
                  output_name='processed_data_transformer',
                  source='/opt/ml/processing/processed_data/')],
              arguments=[
                  "model-name",
                  MODEL_NAME,
                  "train-split-percentage",
                  str(train_split_percentage),
                  "validation-split-percentage",
                  str(validation_split_percentage),
                  "random-seed",
                  str(random_seed)
              ],
              )

preprocessing_job_description = processor.jobs[-1].describe()
output_config = preprocessing_job_description['ProcessingOutputConfig']

print(output_config)

# for output in output_config['Outputs']:
#     if output['OutputName'] == 'word_count_data':
#         word_count_data_file = output['S3Output']['S3Uri']

# print('Output file is located on: {}'.format(word_count_data_file))
