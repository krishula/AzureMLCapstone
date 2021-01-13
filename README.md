
# Capstone Project - Azure Machine Learning Engineer

This is the Capstone for the nanodegree - Azure Machine Learning Engineer. This is an opportunity to use the knowledge we have obtained from this program to solve an interesting problem. In this project, we will create two models: one using Automated ML (AutoML) and one customized model whose hyperparameters are tuned using HyperDrive. We will then compare the performance of both the models and deploy the best performing model.

This project will demonstrate our ability to use an external dataset in the workspace, train a model using the different tools available in the AzureML framework as well as our ability to deploy the model as a web service.

## Project Set Up and Installation

### How to set up this project in Azure ML:

  - We started by creating a new workspace in Azure ML Studio. Then we created a compute instance where we can run our jupyter notebook(s).
  - We then uploaded the starter files aka the jupyter notebooks for the two models that we'll be creating for the project.
  - We imported all the dependencies in the first cell. This will include but not be limited to, Workspace, Experiment, TabularDatasetFactory, Dataset, AutoMLConfig, AmlCompute, ComputeTarget and ComputeTargetException.
  - We created a new compute cluster of STANDARD_D2_V2 virtual machine size and 4 maximum number of nodes or use an existing one with the same configurations.
  - We selected an appropriate dataset for the task.

## Dataset

### Overview
The dataset used in this project is the **Cleveland Heart Disease dataset** taken from the **UCI repository**.

It consists of 303 individuals data. There are 14 columns in the dataset, which are described below.
  - Age: displays the age of the individual.
  - Sex: displays the gender of the individual using the following format :1 = male, 0 = female
  - Chest-pain type: displays the type of chest-pain experienced by the individual using the following format :
    1 = typical angina, 2 = atypical angina, 3 = non — anginal pain, 4 = asymptotic
  - Resting Blood Pressure: displays the resting blood pressure value of an individual in mmHg (unit)
  - Serum Cholestrol: displays the serum cholesterol in mg/dl (unit)
  Fasting Blood Sugar: compares the fasting blood sugar value of an individual with 120mg/dl.
    If fasting blood sugar > 120mg/dl then : 1 (true), else : 0 (false)
  - Resting ECG : displays resting electrocardiographic results. 0 = normal, 1 = having ST-T wave abnormality, 2 = left ventricular hyperthrophy
  - Max heart rate achieved : displays the max heart rate achieved by an individual.
  - Exercise induced angina : 1 = yes, 0 = no
  - ST depression induced by exercise relative to rest: displays the value which is an integer or float.
  - Peak exercise ST segment : 1 = upsloping, 2 = flat, 3 = downsloping
  - Number of major vessels (0–3) colored by flourosopy : displays the value as integer or float.
  - Thal : displays the thalassemia : 3 = normal, 6 = fixed defect, 7 = reversible defect
  - Diagnosis of heart disease : Displays whether the individual is suffering from heart disease or not : 0 = absence, 1, 2, 3, 4 = present.

### Task

We applied Machine Learning approaches using Auto ML and Hyperparameter sampling to classify whether a person is suffering from heart disease or not. We also determined the best model, deployed and used it to see how the model is working.
The dataset consists of 13 features and we used all of them for the task.

### Access
  - We downloaded the dataset from the UCI repository and uploaded it in this github repository. We then access it directly from there.
  - The dataset is loaded using TabularDatasetFactory() function available in azure-ml.
  
## Automated ML

### Auto ML Settings:
For this task we have used:
  - enable_early_stopping: True, To decide whether to enable early termination if the score is not improving in the short term. The default is False.
  - primary_metric: 'accuracy', It's the metric that Automated Machine Learning will optimize for model selection. Automated Machine Learning collects more metrics than it can optimize. We can use the functiom 'azureml.train.automl.utilities.get_primary_metrics' to get a list of valid metrics for a given task.
  - featurization: 'auto', FeaturizationConfig Indicator for whether featurization step should be done automatically or not, or whether customized featurization should be used. If the input data is sparse, featurization cannot be turned on.
  - experiment_timeout_minutes : 20, Maximum amount of time in minutes that all iterations combined can take before the experiment terminates. If not specified, the default experiment timeout is 6 days.
 
### Auto ML Config:
For this task we have used:
  - task = 'classification', It's the type of task to run. Values can be 'classification', 'regression', or 'forecasting' depending on the type of automated ML problem to solve.
  - compute_target=compute_target, It's the Azure Machine Learning compute target to run the Automated Machine Learning experiment on. It has the details of the compute cluster we had created in the beginning.
  - blocked_models = ['KNN','LinearSVM'], A list of algorithms to ignore for an experiment. If enable_tf is False, TensorFlow models are included in blocked_models.
  - enable_onnx_compatible_models = True, Decides whether to enable or disable enforcing the ONNX-compatible models. The default is False.
  - debug_log = 'automl_errors.log', It's the log file to write debug information to. If not specified, 'automl.log' is used.
  - training_data = train_data, It's the training data to be used within the experiment. It should contain both training features and a label column (optionally a sample weights column). If training_data is specified, then the label_column_name parameter must also be specified. We created this data from the original dataset in the beginning as well.
  - label_column_name = 'target', It's the name of the label column used in the training_data. This parameter is applicable to training_data and validation_data parameters.
  - max_concurrent_iterations = 4, Represents the maximum number of iterations that would be executed in parallel. The default value is 1. We have used 4 for this experiment.

According to microsoft documentation, some other parameters that can also be used for automl settings and config are:
  - n_cross_validations: Represents how many cross validations to perform when user validation data is not specified.
  - iterations: The total number of different algorithm and parameter combinations to test during an automated ML experiment. If not specified, the default is 1000 iterations.
  - validation_data: This is the validation data to be used within the experiment. It should contain both training features and label column (optionally a sample weights column). If validation_data is specified, then training_data and label_column_name parameters must be specified.
  - weight_column_name: It's the name of the sample weight column. Automated ML supports a weighted column as an input, causing rows in the data to be weighted up or down. If the input data is from a pandas.DataFrame which doesn't have column names, column indices can be used instead, expressed as integers. This parameter is applicable to training_data and validation_data parameters.
  - cv_split_column_names: List of names of the columns that contain custom cross validation split. Each of the CV split columns represents one CV split where each row are either marked 1 for training or 0 for validation. This parameter is applicable to training_data parameter for custom cross validation purposes. 
  - enable_dnn: Tells the run whether to include DNN based models during model selection. The default is False.

### Results
The best model had shown an accuracy of 85% but the classification report says that it has an accuracy of 92%.

![alt txt](https://github.com/krishula/AzureMLCapstone/blob/main/Screenshots/Screen%20Shot%202021-01-11%20at%209.06.49%20PM.png)

Parameters: 
  - AUC_macro - 91.96%
  - AUC_micro - 91.63%
  - AUC_weighted - 91.96%
The model could be improved by leaving out one or two features that might not be as helpful. More data cleaning and prep before running the automl can achieve that. Another thing that can help is running the automl for longer than the current settings.

Following are screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters:

**Run Details**

![alt txt](https://github.com/krishula/AzureMLCapstone/blob/main/Screenshots/AutoML1.png)

![alt txt](https://github.com/krishula/AzureMLCapstone/blob/main/Screenshots/AutoML2.png)

**Best Model**

Auto ML run created many models as we can see in the scrren shots below. The best model is "Voting Ensemble" with an accuarcy of 85%.

![alt txt](https://github.com/krishula/AzureMLCapstone/blob/main/Screenshots/AutoML3.png)

![alt txt](https://github.com/krishula/AzureMLCapstone/blob/main/Screenshots/AutoML4.png)

**Best Model Parameters**

Other than accuracy we found the following parameter values as shown in the screenshot below:
  - AUC_macro - 91.96%
  - AUC_micro - 91.63%
  - AUC_weighted - 91.96%

![alt txt](https://github.com/krishula/AzureMLCapstone/blob/main/Screenshots/AutoML5.png)

**Best model and run_id**

Here are the details of the best model, such as run id:

![alt txt](https://github.com/krishula/AzureMLCapstone/blob/main/Screenshots/AutoML6.png)

**Fitted Model - Best Estimator from the model**

Here is a sneak peek of the fitted model, we can see the hyperparameters our auto ml model has generated:

![alt txt](https://github.com/krishula/AzureMLCapstone/blob/main/Screenshots/AutoML7.png)

![alt txt](https://github.com/krishula/AzureMLCapstone/blob/main/Screenshots/AutoML8.png)

## Hyperparameter Tuning

We have created a logistic-regression model from scikit-learn and used hyperparameters 'Inverse of Regularization Strength' and 'Maximum number of iterations to converge' for parameter sampling. The sampling method used is RandomSampling which supports both discrete and continuous values. In this sampling method, the values are selected randomly from a defined search space. It also supports early termination of low-performance runs. 
Early stopping policy used is Bandit Policy which takes care of the computational efficiency.

policy = BanditPolicy(evaluation_interval=2, slack_factor=0.1)

Parameters used for ramdom sampling (ps):
  - C: The inverse of the reqularization strength. **{'--C': uniform(0.1, 1)}**
  - max_iter: Maximum number of iterations to converge. **{'--max_iter': quniform(100, 1500, 100)}**

Estimator: est = SKLearn("./", compute_target=aml_compute, entry_script="train.py", pip_packages=['sklearn'])

We configured the hyperdrive run as follows:
  - **estimator = est**, An estimator that will be called with sampled hyperparameters.
  - **hyperparameter_sampling = ps**, The hyperparameter sampling space.
  - **policy = policy**, States the early termination policy to be used. If None - the default, no early termination policy will be used. 
  - **primary_metric_name = 'Accuracy'**, It's the name of the primary metric reported by the experiment runs.
  - **primary_metric_goal = PrimaryMetricGoal.MAXIMIZE**, Takes a value of either PrimaryMetricGoal.MINIMIZE or PrimaryMetricGoal.MAXIMIZE. This parameter determines if the primary metric is to be minimized or maximized when evaluating runs.
  - **max_total_runs = 15**, This is the maximum total number of runs to create. This is the upper bound; there may be fewer runs when the sample space is smaller than this value. If both max_total_runs and max_duration_minutes are specified, the hyperparameter tuning experiment terminates when the first of these two thresholds is reached.
  - **max_concurrent_runs = 4**, The maximum number of runs to execute concurrently. If None, all runs are launched in parallel. The number of concurrent runs is gated on the resources available in the specified compute target. Hence, you need to ensure that the compute target has the available resources for the desired concurrency.

Other parameters that can also be used are:
  - run_config: It's the object for setting up configuration for script/notebook runs.
  - pipeline: It's the pipeline object for setting up configuration for pipeline runs. The pipeline object will be called with the sample hyperparameters to submit pipeline runs.

### Results
Hyperdrive run produced a best model with 87% accuracy.
Parameters of the model were:
  - Inverse of Regularization Strength, C = 0.61
  - Maximum number of iterations to converge, max_iter = 1200
  
### Future Improvements
  - Model can be improved by using a less aggressive early stopping policy like truncation stopping policy. It cancels a percentage of lowest performing runs at each evaluation interval. Runs are compared using the primary metric. This can help up the number of models with higher accuracy.
  - We can use Gridsampling instead of random sampling since it uses all the possible values from the search space. It might get us a model with even better accuracy.
  - Data denoising can also be performed to cut down on the values for the datasets that might deviate the models from better accuracies. 
  - We can try using the timeit() function to test out how fast the models respond to network deployments and result outputs.
  
Following are screenshots of the `RunDetails` widget as well as a screenshot of the best model:

**Run Details**

After configuring auto ml we submit the run.
remote_run = experiment.submit(automl_config, show_output = True)
We have flagged "show_output" = True to see the details as shown below. 

![alt txt] (https://github.com/krishula/AzureMLCapstone/blob/main/Screenshots/Hyperdrive1.png)

![alt txt] (https://github.com/krishula/AzureMLCapstone/blob/main/Screenshots/Hyperdrive2.png)

**Best Model**

![alt txt] (https://github.com/krishula/AzureMLCapstone/blob/main/Screenshots/Hyperdrive3.png)

## Model Deployment

We decide to deploy the best Auto-ml Model. We registered and then deployed it by creating score.py and using it in inference_cofig.
The model endpoint was queried by sending a post request to the model over the REST url.
**resp = requests.post(scoring_uri, test_sample, headers=headers)**
Screenshots for the successfully deployed model and the service request as follows:

**Deployed Mode**

![alt txt] (https://github.com/krishula/AzureMLCapstone/blob/main/Screenshots/Hyperdrive4.png)
![alt txt] (https://github.com/krishula/AzureMLCapstone/blob/main/Screenshots/Hyperdrive5.png)

**Service Request**

![alt txt] (https://github.com/krishula/AzureMLCapstone/blob/main/Screenshots/Hyperdrive6.png)


## Screen Recording
Screencast: https://youtu.be/ASd24ITsWbA
A script is also included in the folder.
