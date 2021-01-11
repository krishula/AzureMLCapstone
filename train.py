from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

run = Run.get_context()

def clean_data(data):
    x_df = data.to_pandas_dataframe().dropna()

    categorical_val = []
    for column in x_df.columns:
        if len(x_df[column].unique()) <= 10:
            categorical_val.append(column)

    categorical_val.remove('target')
    x_df = pd.get_dummies(x_df, columns = categorical_val)

    s_sc = StandardScaler()
    col_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    x_df[col_to_scale] = s_sc.fit_transform(x_df[col_to_scale])


    y_df = x_df.target
    x_df = x_df.drop('target', axis=1)
    return x_df, y_df

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))
    
    factory = TabularDatasetFactory()
    url = 'https://raw.githubusercontent.com/krishula/AzureMLCapstone/main/heart.csv'
    trainds = factory.from_delimited_files(url)

    x, y = clean_data(trainds)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 1) 

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
    joblib.dump(model, 'outputs/heart-diesease-hyper-model.joblib')

if __name__ == '__main__':
    main()
