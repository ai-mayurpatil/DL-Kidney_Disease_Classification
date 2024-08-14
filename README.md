# Kidney Disease Classification using MLflow & DVC
Chronic Kidney Disease (CKD) is a significant global health issue, often progressing unnoticed until it reaches an advanced stage. Early detection of CKD is crucial for timely intervention and treatment, potentially slowing disease progression and improving patient outcomes. This paper focuses on utilizing deep learning techniques to identify factors that may lead to CKD at an early stage, thereby aiding in early diagnosis.

The research utilized a publicly available dataset containing medical information collected in India. This dataset includes various attributes that are potential indicators of CKD. The initial step involved preprocessing the data to handle missing values and outliers, ensuring the dataset's integrity and suitability for analysis. Data preprocessing is a critical phase in machine learning, as it directly impacts the accuracy and reliability of the model's predictions.

Dataset Source Link : https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone

# Project Structure
## 1. Create Virtual Environment using Anaconda
```bash
- conda create -p venv python=3.11
- conda activate "Path"
```
## 2. Template on auto created all project  str
```python
import os
from pathlib import Path
import logging

# Configure logging to display information level logs with timestamp
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

# Define the project name
project_name = 'cnnClassifier'

# List of files and directories to be created
list_of_files = [
    ".github/workflows/.gitkeep",  # GitHub workflow directory with a placeholder file
    f"src/{project_name}/__init__.py",  # Init file for the main project directory
    f"src/{project_name}/components/__init__.py",  # Init file for the components subdirectory
    f"src/{project_name}/utils/__init__.py",  # Init file for the utils subdirectory
    f"src/{project_name}/config/__init__.py",  # Init file for the config subdirectory
    f"src/{project_name}/config/configuration.py",  # Configuration script
    f"src/{project_name}/pipeline/__init__.py",  # Init file for the pipeline subdirectory
    f"src/{project_name}/entity/__init__.py",  # Init file for the entity subdirectory
    f"src/{project_name}/constants/__init__.py",  # Init file for the constants subdirectory
    "config/config.yaml",  # YAML configuration file
    "dvc.yaml",  # DVC pipeline file
    "params.yaml",  # YAML file for parameters
    "requirements.txt",  # File to list dependencies
    "setup.py",  # Setup script for the project
    "research/trials.ipynb",  # Jupyter notebook for research and trials
    "templates/index.html"  # HTML template file
]

# Loop through each file path in the list
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)  # Split the path into directory and filename

    # Check if the directory exists, if not, create it
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    # Create an empty file if it doesn't exist or if it's empty
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating empty file: {filepath}")

    else:
        logging.info(f"{filename} already exists")

```
