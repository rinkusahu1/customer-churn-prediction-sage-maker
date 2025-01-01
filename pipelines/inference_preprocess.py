from __future__ import print_function

import time
import sys
from io import StringIO
import os
import shutil

import argparse
import csv
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder

from sagemaker_containers.beta.framework import (
    content_types, encoders, env, modules, transformer, worker)

# Since we get a headerless CSV file we specify the column names here.
feature_columns_names = ['age', 'job', 'marital', 'education', 'default', 
                         'housing', 'loan','contact', 'month', 'day_of_week', 
                         'campaign', 'pdays', 'previous', 'poutcome','no_previous_contact',
                         'not_working']

label_column = 'rings'

feature_columns_dtype = {
    'age': 'int64',
    'job': 'object',
    'marital': 'object',
    'education': 'object',
    'default': 'object',
    'housing': 'object',
    'loan': 'object',
    'contact': 'object',
    'month': 'object',
    'day_of_week': 'object',
    'campaign': 'int64',
    'pdays': 'int64',
    'previous': 'int64',
    'poutcome': 'object',
    'no_previous_contact': 'int64',
    'not_working': 'int64'
}



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])

    args = parser.parse_args()

    print("saved model!")


def input_fn(input_data, content_type):
    """Parse input data payload

    We currently only take csv input. Since we need to process both labelled
    and unlabelled data we first determine whether the label column is present
    by looking at how many columns were provided.
    """
    if content_type == 'text/csv':
        # Read the raw input data as CSV.
        df = pd.read_csv(StringIO(input_data))
        print(df.columns)
        return df
    else:
        raise ValueError("{} not supported by script!".format(content_type))


def output_fn(prediction, accept):
    """Format prediction output

    The default accept/content-type between containers for serial inference is JSON.
    We also want to set the ContentType or mimetype as the same value as accept so the next
    container can read the response payload correctly.
    """
    accept = 'text/csv'
    print(accept)
    print(prediction)
    return worker.Response(encoders.encode(prediction, accept), mimetype=accept)


def predict_fn(input_data, model):
    # Indicator variable to capture when pdays takes a value of 999
    input_data["no_previous_contact"] = np.where(input_data["pdays"] == 999, 1, 0)
    
    # Indicator for individuals not actively employed
    input_data["not_working"] = np.where(
        np.in1d(input_data["job"], ["student", "retired", "unemployed"]), 1, 0
    )
    df_model_data = input_data
    
    bins = [18, 30, 40, 50, 60, 70, 90]
    labels = ['18-29', '30-39', '40-49', '50-59', '60-69', '70-plus']
    
    df_model_data['age_range'] = pd.cut(df_model_data.age, bins, labels=labels, include_lowest=True)
    df_model_data = pd.concat([df_model_data, pd.get_dummies(df_model_data['age_range'], prefix='age', dtype=int)], axis=1)
    df_model_data.drop('age', axis=1, inplace=True)
    df_model_data.drop('age_range', axis=1, inplace=True)
    
    minMaxScaler , encoder = model
    
    scaled_features =  minMaxScaler.feature_names_in_
    df_model_data[scaled_features] = minMaxScaler.transform(df_model_data[scaled_features])
    
    categorical_columns = encoder.feature_names_in_
    # Fit and transform the categorical columns
    one_hot_encoded = encoder.transform(df_model_data[categorical_columns])
    
    # Create a DataFrame with the encoded columns
    one_hot_df = pd.DataFrame(one_hot_encoded, 
                          columns=encoder.get_feature_names_out(categorical_columns))
    
    # Concatenate the one-hot encoded columns with the original DataFrame
    df_model_data = pd.concat([df_model_data.drop(categorical_columns, axis=1), one_hot_df], axis=1)
    
    return df_model_data

def model_fn(model_dir):
    """Deserialize fitted model
    """
    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    onehot_encoder = joblib.load(os.path.join(model_dir, 'onehot_encoder.joblib'))
    return scaler,onehot_encoder