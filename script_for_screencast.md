Welcome to the capstone project of machine learning engineer nanodegree with Azure ML.
  - We have used the Cleveland Heart Disease dataset taken from the UCI repository to create a machine learning model that can predict whether or not a person has heart disease.
  - First screen shows all the experiments that are active at the moment. It includes the auto ml run that we created in experiment 1 and the hyperdrive run for hyperparameter sampling that we created for experiment 2.
  - We select the experiment “capstone-automl” and see that the run has been completed. Among a list of models, we can see the best model turned out to be “Voting Ensemble” with an accuracy of about 85%. After running the auto-ml evaluation we saw that our model has an accuracy of 92%.

![alt txt](https://github.com/krishula/AzureMLCapstone/blob/main/Screenshots/Screen%20Shot%202021-01-11%20at%209.06.49%20PM.png)
 
 - Moving on to the hyperdrive run, we can see that it has been successfully completed as well.
 - We have later deployed the best auto ml model a web service on Azure container instancesand it can be seen in the video that it has been succefully deployed.
 - Included in the video is an example of a service request made to the deployed model using azure ml and returns the value '1', predicting that the person queried for might have a heart disease.

Thank you!
