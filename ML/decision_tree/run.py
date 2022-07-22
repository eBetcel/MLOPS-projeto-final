"""
Creator: Ivanovitch Silva
Date: 26 Jan. 2022
Implement a pipeline component to train a decision tree model.
"""

import argparse
import logging
import os

import yaml
import tempfile
import mlflow
from mlflow.models import infer_signature
from sklearn.impute import SimpleImputer

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import r2_score as R2
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from scipy.stats import gaussian_kde
import numpy as np
import wandb
RANDOM = 40028922

# configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt='%d-%m-%Y %H:%M:%S')

# reference for a logging obj
logger = logging.getLogger()


def process_args(args):

    # project name comes from config.yaml >> project_name: week_08_example_04
    run = wandb.init(job_type="train")

    logger.info("Downloading and reading train artifact")
    local_path = run.use_artifact(args.train_data).file()
    df_train = pd.read_csv(local_path)

    # Spliting train.csv into train and validation dataset
    logger.info("Spliting data into train/val")
    # split-out train/validation and test dataset
    x_train, x_val, y_train, y_val = train_test_split(df_train.drop(labels="price",axis=1),
                                                    df_train["price"],
                                                    test_size=0.20,
                                                    random_state=RANDOM,
                                                    shuffle=True
                                                 )
    
    logger.info("x train: {}".format(x_train.shape))
    logger.info("y train: {}".format(y_train.shape))
    logger.info("x val: {}".format(x_val.shape))
    logger.info("y val: {}".format(y_val.shape))

    logger.info("Removal Outliers")
    # temporary variable
    x = x_train.select_dtypes("int64").copy()

    # identify outlier in the dataset
    lof = LocalOutlierFactor()
    outlier = lof.fit_predict(x)
    mask = outlier != -1

    logger.info("x_train shape [original]: {}".format(x_train.shape))
    logger.info("x_train shape [outlier removal]: {}".format(x_train.loc[mask,:].shape))

    # dataset without outlier, note this step could be done during the preprocesing stage
    x_train = x_train.loc[mask,:].copy()
    y_train = y_train[mask].copy()

    logger.info("Encoding Target Variable")
    logger.info("Pipeline generation")
    
    # Get the configuration for the pipeline
    with open(args.model_config) as fp:
        model_config = yaml.safe_load(fp)
        
    # Add it to the W&B configuration so the values for the hyperparams
    # are tracked
    wandb.config.update(model_config)
    # The full pipeline 
    pipe = Pipeline(steps = [("regressor",DecisionTreeRegressor(**model_config["decision_tree"])) ]
                   )

    # training 
    logger.info("Training")
    pipe.fit(x_train,y_train)

    # predict
    logger.info("Infering")
    predict = pipe.predict(x_val)
    
    # Evaluation Metrics
    logger.info("Evaluation metrics")
    # Metric: AUC
    aux = MAE(y_val, predict)
    run.summary["MAE"] = aux
    
    # Metric: Accuracy
    aux = R2(y_val, predict)
    run.summary["R2"] = aux

    font = {'weight' : 'bold',
            'size'   : 10}

    plt.rc('font', **font)

    # Calculate the point density
    xy = np.vstack([y_val,predict])
    z = gaussian_kde(xy)(xy)

    fig_real_predict, ax_real_predict = plt.subplots(figsize=(8, 8), dpi=120)
    ax_real_predict.scatter(y_val,predict, c=z, s=10)
    ax_real_predict.plot(y_val,y_val,color='red')
    ax_real_predict.set_xlabel("Real")
    ax_real_predict.set_ylabel("Prediction")
    ax_real_predict.set_title("Real x Prediction density")
    

    fig_tree, ax_tree = plt.subplots(1,1, figsize=(30, 10))
    plot_tree(pipe["regressor"], 
              filled=True, 
              rounded=True, 
              feature_names=x_val.columns, ax=ax_tree,max_depth=3,fontsize=14)
    
    # Uploading figures
    logger.info("Uploading figures")
    run.log(
        {
            "confusion_matrix": wandb.Image(fig_real_predict),
            "tree": wandb.Image(fig_tree)
        }
    )
    
    # Export if required
    if args.export_artifact != "null":
        export_model(run, pipe, x_val, predict, args.export_artifact)

        
def export_model(run, pipe, x_val, val_pred, export_artifact):

    # Infer the signature of the model
    signature = infer_signature(x_val, val_pred)

    with tempfile.TemporaryDirectory() as temp_dir:

        export_path = os.path.join(temp_dir, "model_export")

        mlflow.sklearn.save_model(
            pipe, # our pipeline
            export_path, # Path to a directory for the produced package
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            signature=signature, # input and output schema
            input_example=x_val.iloc[:2], # the first few examples
        )

        artifact = wandb.Artifact(
            export_artifact,
            type="model_export",
            description="Decision Tree pipeline export",
        )
        
        # NOTE that we use .add_dir and not .add_file
        # because the export directory contains several
        # files
        artifact.add_dir(export_path)

        run.log_artifact(artifact)

        # Make sure the artifact is uploaded before the temp dir
        # gets deleted
        artifact.wait()        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Decision Tree",
        fromfile_prefix_chars="@",
    )
    
    parser.add_argument(
        "--train_data",
        type=str,
        help="Fully-qualified name for the training data artifact",
        required=True,
    )

    parser.add_argument(
        "--model_config",
        type=str,
        help="Path to a YAML file containing the configuration for the Decision Tree",
        required=True,
    )

    parser.add_argument(
        "--export_artifact",
        type=str,
        help="Name of the artifact for the exported model. Use 'null' for no export.",
        required=False,
        default="null"
    )

    ARGS = parser.parse_args()

    process_args(ARGS)
