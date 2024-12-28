
import argparse
import json
import logging
import os
import pandas as pd
import pickle as pkl

from sagemaker_containers import entry_point
from sagemaker_xgboost_container.data_utils import get_dmatrix
from sagemaker_xgboost_container import distributed

from sklearn.metrics import roc_auc_score

import xgboost as xgb
import mlflow

from time import gmtime, strftime

suffix = strftime('%d-%H-%M-%S', gmtime())

user_profile_name = os.getenv('USER', 'sagemaker')
experiment_name = os.getenv('MLFLOW_EXPERIMENT_NAME')
region = os.getenv('REGION')

mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_ARN'))
mlflow.set_experiment(experiment_name=experiment_name if experiment_name else f"train-{suffix}")

def _xgb_train(params, dtrain, dval, evals, num_boost_round, model_dir, is_master):
    """Run xgb train on arguments given with rabit initialized.

    This is our rabit execution function.

    :param args_dict: Argument dictionary used to run xgb.train().
    :param is_master: True if current node is master host in distributed training,
                        or is running single node training job.
                        Note that rabit_run includes this argument.
    """
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        evals=evals,
        num_boost_round=num_boost_round
    )

    val_auc = roc_auc_score(dval.get_label(), booster.predict(dval))
    train_auc = roc_auc_score(dtrain.get_label(), booster.predict(dtrain))
    mlflow.log_params(params)
    mlflow.log_metrics({"validation_auc":val_auc, "train_auc":train_auc})
    # emit training metrics - SageMaker collects them from the log stream
    print(f"[0]#011train-auc:{train_auc}#011validation-auc:{val_auc}")
    
    if is_master:
        model_location = model_dir + '/xgboost-model'
        pkl.dump(booster, open(model_location, 'wb'))
        print("Stored trained model at {}".format(model_location))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Hyperparameters are described here.
    parser.add_argument('--max_depth', type=int)
    parser.add_argument('--eta', type=float)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--gamma', type=int)
    parser.add_argument('--min_child_weight', type=float)
    parser.add_argument('--subsample', type=float)
    parser.add_argument('--colsample_bytree', type=float)
    parser.add_argument('--verbosity', type=int)
    parser.add_argument('--objective', type=str)
    parser.add_argument('--num_round', type=int)
    parser.add_argument('--early_stopping_rounds', type=int)
    parser.add_argument('--tree_method', type=str, default="auto")
    parser.add_argument('--predictor', type=str, default="auto")

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output_data_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--sm_hosts', type=str, default=os.environ.get('SM_HOSTS'))
    parser.add_argument('--sm_current_host', type=str, default=os.environ.get('SM_CURRENT_HOST'))
    parser.add_argument('--sm_training_env', type=str, default=os.environ.get('SM_TRAINING_ENV'))
    
    print("main function")
    args, _ = parser.parse_known_args()

    # Get SageMaker host information from runtime environment variables
    sm_hosts = json.loads(args.sm_hosts)
    sm_current_host = args.sm_current_host
    dtrain = get_dmatrix(args.train, 'CSV')
    dval = get_dmatrix(args.validation, 'CSV')

    watchlist = [(dtrain, 'train'), (dval, 'validation')] if dval is not None else [(dtrain, 'train')]

    # get SageMaker enviroment setup
    sm_training_env = json.loads(args.sm_training_env)
    
    # enable auto logging
    mlflow.xgboost.autolog(log_model_signatures=False, log_datasets=False)

    train_hp = {
        'max_depth': args.max_depth,
        'eta': args.eta,
        'gamma': args.gamma,
        'min_child_weight': args.min_child_weight,
        'subsample': args.subsample,
        'verbosity': args.verbosity,
        'objective': args.objective,
        'tree_method': args.tree_method,
        'predictor': args.predictor,
    }

    xgb_train_args = dict(
        params=train_hp,
        dtrain=dtrain,
        dval=dval,
        evals=watchlist,
        num_boost_round=args.num_round,
        model_dir=args.model_dir)

    with mlflow.start_run(
        run_name=f"container-training-{suffix}",
        description="xgboost running in SageMaker container in script mode"
    ) as run:

        mlflow.set_tags(
            {
                'mlflow.user':user_profile_name,
                'mlflow.source.type':'JOB',
                'mlflow.source.name': f"https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/jobs/{sm_training_env['job_name']}" if sm_training_env['current_host'] != 'sagemaker-local' else sm_training_env['current_host']
            }
        )
    
        if len(sm_hosts) > 1:
            # Wait until all hosts are able to find each other
            entry_point._wait_hostname_resolution()
    
            # Execute training function after initializing rabit.
            distributed.rabit_run(
                exec_fun=_xgb_train,
                args=xgb_train_args,
                include_in_training=(dtrain is not None),
                hosts=sm_hosts,
                current_host=sm_current_host,
                update_rabit_args=True
            )
        else:
            # If single node training, call training method directly.
            if dtrain:
                xgb_train_args['is_master'] = True
                _xgb_train(**xgb_train_args)
            else:
                raise ValueError("Training channel must have data to train model.")

# Return model object
def model_fn(model_dir):
    """Deserialize and return fitted model.

    Note that this should have the same name as the serialized model in the _xgb_train method
    """
    model_file = 'xgboost-model'
    booster = pkl.load(open(os.path.join(model_dir, model_file), 'rb'))
    return booster
