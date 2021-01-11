
# Capstone Project - Azure Machine Learning Engineer

This is the Capstone for the nanodegree - Azure Machine Learning Engineer. This is an opportunity to use the knowledge we have obtained from this program to solve an interesting problem. In this project, we will create two models: one using Automated ML (AutoML) and one customized model whose hyperparameters are tuned using HyperDrive. We will then compare the performance of both the models and deploy the best performing model.

This project will demonstrate our ability to use an external dataset in the workspace, train a model using the different tools available in the AzureML framework as well as our ability to deploy the model as a web service.

## Project Set Up and Installation

### How to set up this project in Azure ML:

  - We start by creating a new workspace in Azure ML Studio. Then we create a compute instance where we can run our jupyter notebook(s).
  - We then upload the starter files aka the jupyter notebooks for the two models that we'll be creating for the project.
  - We import all the dependencies in the first cell. This will include but not be limited to, Workspace, Experiment, TabularDatasetFactory, Dataset, AutoMLConfig, AmlCompute, ComputeTarget and ComputeTargetException.
  - We create a new compute cluster of STANDARD_D2_V2 virtual machine size and 4 maximum number of nodes or use an existing one with the same configurations.
  - We select an appropriate dataset for the task and register it in the auto ml workspace. If already registered, we use the existing one since we might be doing this multiple times.
  - We then split the dataset into train and test and then treat it so it's compatible with azureml and then upload it to the source directory and directly use it from there.

## Dataset

### Overview
# *TODO*: Explain about the data you are using and where you got it from.

### Task
# *TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.

### Access
  - We select an appropriate dataset for the task and register it in the auto ml workspace. If already registered, we use the existing one since we might be doing this multiple times.
  - We then split the dataset into train and test and then treat it so it's compatible with azureml and then upload it to the source directory and directly use it from there.
## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment
### Auto ML Settings:
We'll be using:
  - experiment_timeout_minutes : 25, Maximum amount of time in minutes that all iterations combined can take before the experiment terminates. If not specified, the default experiment timeout is 6 days.
  - max_concurrent_iterations": 4, Represents the maximum number of iterations that would be executed in parallel. The default value is 1. We have used 4 for this experiment.
  - n_cross_validations": 5, Represents how many cross validations to perform when user validation data is not specified.
  - primary_metric": 'accuracy', It's the metric that Automated Machine Learning will optimize for model selection. Automated Machine Learning collects more metrics than it can optimize. We can use the functiom 'azureml.train.automl.utilities.get_primary_metrics' to get a list of valid metrics for a given task.
 
### Auto ML Config:
We'll be using:
  - task = 'classification', It's the type of task to run. Values can be 'classification', 'regression', or 'forecasting' depending on the type of automated ML problem to solve.
  - debug_log = 'automl_errors.log', It's the log file to write debug information to. If not specified, 'automl.log' is used.
  - compute_target=compute_target, It's the Azure Machine Learning compute target to run the Automated Machine Learning experiment on. It has the details of the compute cluster we had created in the beginning.
  - training_data = train_data, It's the training data to be used within the experiment. It should contain both training features and a label column (optionally a sample weights column). If training_data is specified, then the label_column_name parameter must also be specified. We created this data from the original dataset in the beginning as well.
  - label_column_name = 'target', It's the name of the label column used in the training_data. This parameter is applicable to training_data and validation_data parameters.

According to microsoft documentation, some other parameters that can also be used for automl settings and config are:
  - iterations: The total number of different algorithm and parameter combinations to test during an automated ML experiment. If not specified, the default is 1000 iterations.
  - featurization: 'auto' / 'off' / FeaturizationConfig Indicator for whether featurization step should be done automatically or not, or whether customized featurization should be used. If the input data is sparse, featurization cannot be turned on.
  - validation_data: This is the validation data to be used within the experiment. It should contain both training features and label column (optionally a sample weights column). If validation_data is specified, then training_data and label_column_name parameters must be specified.
  - weight_column_name: It's the name of the sample weight column. Automated ML supports a weighted column as an input, causing rows in the data to be weighted up or down. If the input data is from a pandas.DataFrame which doesn't have column names, column indices can be used instead, expressed as integers. This parameter is applicable to training_data and validation_data parameters.
  - cv_split_column_names: List of names of the columns that contain custom cross validation split. Each of the CV split columns represents one CV split where each row are either marked 1 for training or 0 for validation. This parameter is applicable to training_data parameter for custom cross validation purposes. 
  - enable_dnn: Tells the run whether to include DNN based models during model selection. The default is False.

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
