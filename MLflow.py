import pandas as pd, numpy as np, warnings, mlflow, seaborn as sns, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, log_loss, confusion_matrix
from imblearn.over_sampling import SMOTE
warnings.simplefilter("ignore")

def load_data(file_path):
    df = pd.read_csv(file_path)
    x = df.drop('Air Quality', axis=1)
    y = df['Air Quality']
    smote = SMOTE()
    x_resampled, y_resampled = smote.fit_resample(x, y)
    return train_test_split(x_resampled, y_resampled, test_size=0.2, random_state=42)

def log_metrics_and_params(model, xtrain, ytrain, xtest, ytest, report):
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
    cm = confusion_matrix(ytest, ypred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    plt.close()

def train_and_log_models(xtrain, xtest, ytrain, ytest, file_path):
    models = [
        LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier(),
        KNeighborsClassifier(), GaussianNB(), SGDClassifier(),
        GradientBoostingClassifier(), AdaBoostClassifier(), ExtraTreesClassifier()
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
            mlflow.log_artifact(f'confusion_matrix.png')
            model_uri = f'runs:/{mlflow.active_run().info.run_id}/{model_name}'
            mlflow.register_model(model_uri, model_name)
            
        mlflow.end_run()
    return pd.DataFrame.from_dict(model_dict, orient='index', columns=['Accuracy'])\
        .reset_index().rename(columns={'index':'Model'}).sort_values(by='Accuracy', ascending=False)

if __name__ == "__main__":
    file_path = r'C:\Users\GANAPA\Downloads\MLFlow (MLOps)\pollution_dataset.csv'
    xtrain, xtest, ytrain, ytest = load_data(file_path)
    mlflow.set_tracking_uri('http://localhost:5000')
    experiment_name = 'Air Quality Pred'
    mlflow.set_experiment(experiment_name)
    model_results = train_and_log_models(xtrain, xtest, ytrain, ytest, file_path)
    print(model_results)
