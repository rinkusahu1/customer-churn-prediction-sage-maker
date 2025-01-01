import pandas as pd
import numpy as np
import mlflow
from mlflow.data.pandas_dataset import PandasDataset
from time import gmtime, strftime
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
import joblib
def preprocess(
    input_data_s3_path,
    output_s3_prefix,
    tracking_server_arn,
    experiment_name=None,
    pipeline_run_name=None,
    run_id=None,
):
    try:
        suffix = strftime('%d-%H-%M-%S', gmtime())
        mlflow.set_tracking_uri(tracking_server_arn)
        experiment = mlflow.set_experiment(experiment_name=experiment_name if experiment_name else f"{preprocess.__name__ }-{suffix}")
        pipeline_run = mlflow.start_run(run_name=pipeline_run_name) if pipeline_run_name else None            
        run = mlflow.start_run(run_id=run_id) if run_id else mlflow.start_run(run_name=f"processing-{suffix}", nested=True)

        # Load data
        df_data = pd.read_csv(input_data_s3_path, sep=";")

        input_dataset = mlflow.data.from_pandas(df_data, source=input_data_s3_path)
        mlflow.log_input(input_dataset, context="raw_input")
        
        target_col = "y"
    
        # Indicator variable to capture when pdays takes a value of 999
        df_data["no_previous_contact"] = np.where(df_data["pdays"] == 999, 1, 0)
    
        # Indicator for individuals not actively employed
        df_data["not_working"] = np.where(
            np.in1d(df_data["job"], ["student", "retired", "unemployed"]), 1, 0
        )
    
        # remove data not used for the modelling
        df_model_data = df_data.drop(
            ["duration", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"],
            axis=1,
        )
    
        bins = [18, 30, 40, 50, 60, 70, 90]
        labels = ['18-29', '30-39', '40-49', '50-59', '60-69', '70-plus']
    
        df_model_data['age_range'] = pd.cut(df_model_data.age, bins, labels=labels, include_lowest=True)
        df_model_data = pd.concat([df_model_data, pd.get_dummies(df_model_data['age_range'], prefix='age', dtype=int)], axis=1)
        df_model_data.drop('age', axis=1, inplace=True)
        df_model_data.drop('age_range', axis=1, inplace=True)

        # Scale features
        scaled_features = ['pdays', 'previous', 'campaign']
        scaler = MinMaxScaler()
        df_model_data[scaled_features] = scaler.fit_transform(df_model_data[scaled_features])
        
        categorical_columns = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week", "poutcome"]
        
        # Fit the OneHotEncoder
        encoder = OneHotEncoder(sparse_output=False,dtype=int)
        
        # Fit and transform the categorical columns
        one_hot_encoded = encoder.fit_transform(df_model_data[categorical_columns])
        
        # Create a DataFrame with the encoded columns
        one_hot_df = pd.DataFrame(one_hot_encoded, 
                              columns=encoder.get_feature_names_out(categorical_columns))
        
        # Concatenate the one-hot encoded columns with the original DataFrame
        df_model_data = pd.concat([df_model_data.drop(categorical_columns, axis=1), one_hot_df], axis=1)
        # Combine with other columns
        df_model_data['y'] = df_model_data['y'].map({'yes': 1, 'no': 0})
        
        cols = ['y'] + [col for col in df_model_data.columns if col != 'y']
        df_model_data = df_model_data[cols]

        model_dataset = mlflow.data.from_pandas(df_data)
        mlflow.log_input(model_dataset, context="model_dataset")
        
        # Shuffle and split the dataset
        train_data, validation_data, test_data = np.split(
            df_model_data.sample(frac=1, random_state=1729),
            [int(0.7 * len(df_model_data)), int(0.9 * len(df_model_data))],
        )
    
        print(f"## Data split > train:{train_data.shape} | validation:{validation_data.shape} | test:{test_data.shape}")
    
        mlflow.log_params(
            {
                "full_dataset": df_model_data.shape,
                "train": train_data.shape,
                "validate": validation_data.shape,
                "test": test_data.shape
            }
        )

        # Set S3 upload path
        train_data_output_s3_path = f"{output_s3_prefix}/train/train.csv"
        validation_data_output_s3_path = f"{output_s3_prefix}/validation/validation.csv"
        test_x_data_output_s3_path = f"{output_s3_prefix}/test/test_x.csv"
        test_y_data_output_s3_path = f"{output_s3_prefix}/test/test_y.csv"
        baseline_data_output_s3_path = f"{output_s3_prefix}/baseline/baseline.csv"

        encoder_path = f"{output_s3_prefix}/artifacts_model/onehot_encoder.joblib"
        scaler_path = f"{output_s3_prefix}/artifacts_model/scaler.joblib"
        
        # Upload datasets to S3
        train_data.to_csv(train_data_output_s3_path, index=False, header=False)
        validation_data.to_csv(validation_data_output_s3_path, index=False, header=False)
        test_data[target_col].to_csv(test_y_data_output_s3_path, index=False, header=False)
        test_data.drop([target_col], axis=1).to_csv(test_x_data_output_s3_path, index=False, header=False)
        
        # We need the baseline dataset for model monitoring
        df_model_data.drop([target_col], axis=1).to_csv(baseline_data_output_s3_path, index=False, header=False)

        joblib.dump(encoder, encoder_path)
        joblib.dump(scaler, scaler_path)
        print("## Data processing complete. Exiting.")
        
        return {
            "train_data":train_data_output_s3_path,
            "validation_data":validation_data_output_s3_path,
            "test_x_data":test_x_data_output_s3_path,
            "test_y_data":test_y_data_output_s3_path,
            "baseline_data":baseline_data_output_s3_path,
            "experiment_name":experiment.name,
            "pipeline_run_id":pipeline_run.info.run_id if pipeline_run else ''
        }

    except Exception as e:
        print(f"Exception in processing script: {e}")
        raise e
    finally:
        mlflow.end_run()