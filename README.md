# Air-Quality-Prediction-with-MLflow (MLOps)

## Overview
This project demonstrates the use of MLflow for managing the machine learning lifecycle, including experimentation, reproducibility, and deployment. The project focuses on predicting air quality using various machine learning models.

## Features
- Experiment tracking
- Model versioning
- Model packaging
- Model deployment
- Confusion matrix visualization
- Metrics logging

## Installation
To install the necessary dependencies, run:
```bash
pip install -r requirements.txt
```

## Project Description
This project involves predicting air quality using various machine learning models. The dataset used for training and evaluation is `pollution_dataset.csv`. The project includes the following steps:

1. **Data Loading and Preprocessing**: The data is loaded from a CSV file, and SMOTE is applied to handle class imbalance.
2. **Model Training**: Multiple classifiers are trained, including Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors, Naive Bayes, SGD, Gradient Boosting, AdaBoost, and Extra Trees.
3. **Metrics Logging**: Metrics such as accuracy, F1 score, precision, recall, and log loss are logged using MLflow.
4. **Confusion Matrix Visualization**: Confusion matrices are plotted and saved for each model.
5. **Model Logging and Versioning**: Models are logged and versioned using MLflow, along with their parameters and metrics.
6. **Experiment Tracking**: MLflow is used to track experiments, log artifacts, and manage the machine learning lifecycle.

To start the MLflow server and run the project, follow the steps in the "Getting Started" section.
To install the necessary dependencies, run:
```bash
pip install -r requirements.txt
```

## Usage
1. **Tracking Experiments**: Use MLflow to log parameters, metrics, and artifacts.
2. **Model Versioning**: Save and version your models.
3. **Model Packaging**: Package your models with conda or docker.
4. **Model Deployment**: Deploy your models using MLflow's deployment tools.

## Getting Started
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/MLFlow-MLOps.git
    ```
2. Navigate to the project directory:
    ```bash
    cd MLFlow-MLOps
    ```
3. Run the MLflow server:
    ```bash
    mlflow ui
    ```
4. Execute the main script to start training and logging models:
    ```bash
    python MLflow.py
    ```

## Project Structure
- `MLflow.py`: Main script for loading data, training models, and logging metrics.
- `requirements.txt`: List of dependencies required to run the project.
- `pollution_dataset.csv`: Dataset used for training and evaluation.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss what you would like to change.

## License
This project is licensed under the MIT License.

