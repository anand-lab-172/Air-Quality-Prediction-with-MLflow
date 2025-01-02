REM filepath: /c:/Users/GANAPA/Downloads/MLFlow (MLOps)/schedule_mlflow.bat
@echo off
set PYTHON_PATH=C:\Users\GANAPA\AppData\Local\Microsoft\WindowsApps\python3.12.exe
set SCRIPT_PATH=C:\Users\GANAPA\Downloads\MLFlow_(MLOps)\MLflow.py

REM Schedule the MLflow server to start 20 minutes before the script runs
schtasks /create /tn "Start MLflow Server" /tr "cmd /c cd /d C:\Users\GANAPA\Downloads\MLFlow_(MLOps) && start /b mlflow ui" /sc daily /st 11:00

REM Schedule the MLflow script to run
schtasks /create /tn "Run MLflow Script" /tr "cmd /c cd /d C:\Users\GANAPA\Downloads\MLFlow_(MLOps) && %PYTHON_PATH% %SCRIPT_PATH% && taskkill /f /im mlflow.exe" /sc daily /st 11:01

REM Schedule the MLflow server to stop 20 minutes after the script runs
REM schtasks /create /tn "Stop MLflow Server" /tr "cmd /c taskkill /f /im mlflow.exe" /sc daily /st 18:40