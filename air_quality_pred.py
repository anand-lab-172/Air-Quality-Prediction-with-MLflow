"""
importing the required libraries and modules
"""
import pandas as pd, numpy as np, warnings, mlflow, seaborn as sns, matplotlib.pyplot as plt, os, shutil
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, log_loss, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import RidgeClassifier, PassiveAggressiveClassifier, Perceptron
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from mlflow.tracking import MlflowClient
warnings.simplefilter("ignore")


def load_data(file_path):
    """
    Loading the data and splitting the data into train and test sets
    Using SMOTE to balance the data (oversampling the minority class)
    """
    df = pd.read_csv(file_path)
    x = df.drop('Air Quality', axis=1)
    y = df['Air Quality']
    smote = SMOTE()
    x_resampled, y_resampled = smote.fit_resample(x, y)
    return train_test_split(x_resampled, y_resampled, test_size=0.2, random_state=42)

def log_metrics_and_params(model, xtrain, ytrain, xtest, ytest, report):
    """
    calculating the metrics and logging the metrics and parameters for the model
    """
    metrics = {
        'Accuracy': report['accuracy'],
        'F1_Score_Macro': report['macro avg']['f1-score'],
        'Train_score': model.score(xtrain, ytrain),
        'Test_score': model.score(xtest, ytest),
        'Training_f1_score': f1_score(ytrain, model.predict(xtrain), average='macro'),
        'Training_precision_score': precision_score(ytrain, model.predict(xtrain), average='macro'),
        'Training_recall_score': recall_score(ytrain, model.predict(xtrain), average='macro'),
    }
    for class_label in report.keys():
        if class_label not in ['accuracy', 'macro avg', 'weighted avg']:
            metrics[f'Recall_Class_{class_label}'] = report[class_label]['recall']
            metrics[f'Precision_Class_{class_label}'] = report[class_label]['precision']
    if hasattr(model, "predict_proba"):
        metrics['Training_log_loss'] = log_loss(ytrain, model.predict_proba(xtrain))
    mlflow.log_metrics(metrics)
    mlflow.log_params(model.get_params())

def plot_confusion_matrix(ytest, ypred, model_name):
    """
    Plotting the confusion matrix for the model
    """
    cm = confusion_matrix(ytest, ypred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    plt.close()

def register_model(model_name, register):
    '''
    Deleting old models version automatically which are registered with the same name then,
    Registering the new model with the same name
    '''
    client = MlflowClient()
    if register:
        try:
            client.delete_registered_model(name=model_name)
            print(f"Deleted old model {model_name}")
        except Exception as e:
            print(f"Failed to delete model {model_name}: {e}")
        model_uri = f'runs:/{mlflow.active_run().info.run_id}/{model_name}'
        mlflow.register_model(model_uri, model_name)

def train_and_log_models(xtrain, xtest, ytrain, ytest, file_path, register=False):
    """
    Training multiple models and logging the metrics and parameters for each model.
    defining the model name, logging the model, metrics, parameters, confusion matrix, and evaluation results.
    """
    models = [
        LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier(),
        KNeighborsClassifier(), GaussianNB(), SGDClassifier(),
        GradientBoostingClassifier(), AdaBoostClassifier(), ExtraTreesClassifier(),
        RidgeClassifier(), PassiveAggressiveClassifier(), Perceptron(), SVC(), NuSVC(), LinearSVC(),
        VotingClassifier(estimators=[('lr', LogisticRegression()), ('rf', RandomForestClassifier())]),
        StackingClassifier(estimators=[('dt', DecisionTreeClassifier()), ('gb', GradientBoostingClassifier())]),
        MLPClassifier(activation='identity'), LinearDiscriminantAnalysis(),  QuadraticDiscriminantAnalysis(), HistGradientBoostingClassifier()
    ]
    model_dict = {}
    for model in models:
        model_name = type(model).__name__.replace('Classifier', ' Classifier').replace('Regression', ' Regression')
        with mlflow.start_run(run_name=model_name, description=f'A {model_name.lower()} model for air quality classification using metrics like accuracy and recall for various classes.') as run:
            model.fit(xtrain, ytrain)
            ypred = model.predict(xtest)
            report = classification_report(ytest, ypred, output_dict=True)
            mlflow.sklearn.log_model(model, model_name, input_example=xtrain.iloc[0:1])
            log_metrics_and_params(model, xtrain, ytrain, xtest, ytest, report)
            mlflow.log_params({'Model': model_name})
            mlflow.log_artifact(file_path)
            model_dict.update({model_name:report['accuracy']})
            mlflow.log_input(mlflow.data.from_pandas(xtrain), context="Train")
            mlflow.log_input(mlflow.data.from_pandas(xtest), context="Eval")
            evaluation_results = pd.DataFrame(report).transpose().reset_index().rename(columns={'index':'Metrics'})
            accuracy_row = evaluation_results[evaluation_results['Metrics'] == 'accuracy']
            evaluation_results = pd.concat([accuracy_row, evaluation_results[evaluation_results['Metrics'] != 'accuracy']], ignore_index=True)
            mlflow.log_table(data=evaluation_results, artifact_file="evaluation_results.json")
            mlflow.set_tag("Model", model_name)
            plot_confusion_matrix(ytest, ypred, model_name)
            mlflow.log_artifact('confusion_matrix.png')
            register_model(model_name, register)
        mlflow.end_run()
    Model_ = pd.DataFrame.from_dict(model_dict, orient='index', columns=['Accuracy']).reset_index().rename(columns={'index':'Model'}).sort_values(by='Accuracy', ascending=False)
    Model_.to_csv(r'C:\Users\GANAPA\Downloads\MLFlow (MLOps)\model_results.csv', index=False)
    return Model_

def run_logs(experiment_name, delete_runs=True):
    """
    Deleting the old runs in the experiment if delete_runs is True
    """
    if delete_runs: 
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            for run in mlflow.search_runs(experiment_ids=[experiment.experiment_id]).itertuples():
                mlflow.delete_run(run.run_id)
                print(f"Deleted run {run.run_id} from {experiment_name}")

                # Deleting the folder of the run from root mlruns
                run_folder = f"./mlruns/{experiment.experiment_id}/{run.run_id}"
                shutil.rmtree(run_folder, ignore_errors=True)
                print(f"Deleted folder {run_folder}")

                # Deleting the datasets folder of the run from root mlruns
                dataset_folder = f"./mlruns/{experiment.experiment_id}/datasets"
                shutil.rmtree(dataset_folder, ignore_errors=True)
                print(f"Deleted folder {dataset_folder}")
                
                # Deleting the old runs artifacts from root artifacts folder
                artifact_folder = f"./artifacts/{run.run_id}"
                shutil.rmtree(artifact_folder, ignore_errors=True)
                print(f"Deleted folder {artifact_folder}")
                
        else:
            print(f"Experiment '{experiment_name}' not found.")

def define_experiment(uri, experiment_name):
    """
    Defining the experiment and setting the experiment
    """
    mlflow.set_tracking_uri(uri)
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(experiment_name, artifact_location='file:///C:/Users/GANAPA/Downloads/MLFlow (MLOps)/artifacts')
        mlflow.set_experiment(experiment_name)
        print(f"Experiment '{experiment_name}' created successfully with ID '{mlflow.get_experiment_by_name(experiment_name).experiment_id}'")
    else:
        mlflow.set_experiment(experiment_name)
        print(f"Experiment '{experiment_name}' already exists with ID '{mlflow.get_experiment_by_name(experiment_name).experiment_id}'")

def predict(model_results, data):
    """
    Predicting the air quality using the best model
    """
    name, model_version = model_results.iloc[:1].Model.values[0], '1'
    print(f'{name} is the best model with an accuracy of {model_results.iloc[:1].Accuracy.values[0]*100}')
    model_uri = f'models:/{name}/{model_version}'
    load_model = mlflow.sklearn.load_model(model_uri)
    pred = load_model.predict(data.drop('Air Quality', axis=1))
    data['Predicted_Air_Quality'] = pred
    data['Evaluate'] = data['Air Quality'] == data['Predicted_Air_Quality']
    data.to_csv(r'C:\Users\GANAPA\Downloads\MLFlow (MLOps)\predicted_air_quality.csv', index=False)
    print('Predicted air quality saved to predicted_air_quality.csv')

if __name__ == "__main__":
    """
    Main function to load the data, define the experiment, run the logs, train and log the models, and print the model results
    """

    file_path = r'C:\Users\GANAPA\Downloads\MLFlow (MLOps)\pollution_dataset.csv'
    data = pd.read_csv(file_path)
    uri, experiment_name = 'http://localhost:5000', 'Air Quality Pred'
    define_experiment(uri, experiment_name), run_logs(experiment_name)
    xtrain, xtest, ytrain, ytest = load_data(file_path)
    model_results = train_and_log_models(xtrain, xtest, ytrain, ytest, file_path, True)
    print(model_results)
    predict(model_results, data)


# Hint:
# ./artifacts or ./mlruns are under the root directory where the code is executed (all under same project folder)

# Steps to run the code:
# Run the MLflow.py file to train the models and predict the air quality using the best model
# The confusion matrix for each model is saved as confusion_matrix.png
# The evaluation results for each model are saved in evaluation_results.json
# The models are registered with the same name and the old models are deleted automatically
# The runs, datasets, and artifacts are deleted automatically
# The experiment is created if it does not exist and set if it exists
# The metrics and parameters are logged for each model
# The best model is selected based on the accuracy and used to predict the air quality
# The model results are saved in model_results.csv and the predicted air quality is saved in predicted_air_quality.csv
