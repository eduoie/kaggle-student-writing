# About

A project to put prediction models in production for the Kaggle competition 'Feedback Prize - Evaluating Student Writing'

A very simple model using Naive Bayes has been created to make a simple end-to-end sample of data transformation, creation of a feature store, model training and deployment, and a pipeline to tie these steps together.

As the next step, this task is repeated using a more complex transformer model.

The transformer model and processing has been taken from some of the available notebooks of the competition:
* https://www.kaggle.com/cdeotte/pytorch-bigbird-ner-cv-0-615
* For a visual explanation, check: https://www.kaggle.com/thedrcat/visual-tutorial-ner-chunking-token-classification/notebook

# Preparation

For local development, a new conda environment has been created 

Follow instructions at `SKLearnProcessor_local_processing.py`:
```
1. Install required Python packages:
     pip install boto3 sagemaker pandas scikit-learn
     pip install 'sagemaker[local]'

     ...have Docker ready, etc
```

## Cloud development

The project is designed to be run using AWS services, specifically on AWS SageMaker

## Local mode development

When working with SageMaker script mode, it is extremelly helpful to be able to debug and test the scripts offline on a local computer.

https://github.com/aws-samples/amazon-sagemaker-local-mode

https://aws.amazon.com/blogs/machine-learning/use-the-amazon-sagemaker-local-mode-to-train-on-your-notebook-instance/

Check `tutorial_scikit_learn_local_processing` and `local_train` folders

For the transformers model processing, do `pip install torch` on the proper conda environment

**Environment variables for scripts:**

https://github.com/aws/sagemaker-training-toolkit/blob/master/ENVIRONMENT_VARIABLES.md

Other references:

https://github.com/aws-samples/amazon-sagemaker-local-mode/tree/main/scikit_learn_script_mode_local_training_and_serving

[Inference Pipeline with Scikit-learn and Linear Learner](https://github.com/aws/amazon-sagemaker-examples/blob/main/sagemaker-python-sdk/scikit_learn_inference_pipeline/Inference%20Pipeline%20with%20Scikit-learn%20and%20Linear%20Learner.ipynb)

In many cases, when the trained model is used for processing real time or batch prediction requests, the model receives data in a format which needs to pre-processed (e.g. featurized) before it can be passed to the algorithm. In the following notebook, we will demonstrate how you can build your ML Pipeline leveraging the Sagemaker Scikit-learn container and SageMaker Linear Learner algorithm & after the model is trained, deploy the Pipeline (Data preprocessing and Lineara Learner) as an Inference Pipeline behind a single Endpoint for real time inference and for batch inferences using Amazon SageMaker Batch Transform.

