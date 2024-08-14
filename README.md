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
## 3. Create requirements for project
### `requirements.txt`
```txt
tensorflow==2.12.0
pandas 
dvc
mlflow==2.2.2
notebook
numpy
matplotlib
seaborn
python-box==6.0.2
pyYAML
tqdm
ensure==1.0.2
joblib
types-PyYAML
scipy
Flask
Flask-Cors
gdown
-e .
```
### `setup.py`
```python
import setuptools

# Read the long description from the README.md file
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Package version
__version__ = "0.0.0"

# Metadata for the package
REPO_NAME = "DL-Kidney_Disease_Classification"  
AUTHOR_USER_NAME = "ai-mayurpatil"  
SRC_REPO = "cnnClassifier"  
AUTHOR_EMAIL = "ai.mayurpatil@gmail.com"  

# Setup configuration for the package
setuptools.setup(
    name=SRC_REPO,  
    version=__version__,  
    author=AUTHOR_USER_NAME,  
    author_email=AUTHOR_EMAIL,  
    description="A small python package for CNN app",  
    long_description=long_description,  
    long_description_content="text/markdown",  
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",  
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",  
    },
    package_dir={"": "src"},  
    packages=setuptools.find_packages(where="src"),  
)

```
Run Command:
```bash
pip install -r requirements.txt
```
## 4. Create logger functionality
### `src\cnnClassifier\__init__.py`
```python
import os
import sys
import logging

# Define the logging format string, including timestamp, log level, module, and message
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

# Directory where the log file will be stored
log_dir = "logs"

# Path to the log file
log_filepath = os.path.join(log_dir, "running_logs.log")

# Create the log directory if it doesn't exist
os.makedirs(log_dir, exist_ok=True)

# Configure the logging settings
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format=logging_str,  # Use the defined format string
    handlers=[
        logging.FileHandler(log_filepath),  # Write logs to the specified log file
        logging.StreamHandler(sys.stdout)  # Output logs to the console (stdout)
    ]
)

# Create a logger instance with a custom name
logger = logging.getLogger("cnnClassifierLogger")
```
### `main.py`
```python
from src.cnnClassifier import logger

logger.info("Welcome to our custom log")
```
```bash
python main.py
```
## 5. Create exception functionality
### `src\cnnClassifier\utils\common.py`
```python
import os
from box.exceptions import BoxValueError
import yaml
from cnnClassifier import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64



@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")




@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"


def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()


def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())
```
