instaffo-matching
==============================

ML Case Study


## 📁 Project Organization

The <- points to the important files 

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md         
    ├── data
    │   ├── external       
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               
    │
    ├── models             <- serialized model
    │
    ├── notebooks         
    │
    ├── references        
    │
    ├── reports            
    │   └── figures         
    │
    ├── requirements.txt   <- dependecies
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data            
    │   │   └── make_dataset.py    <- we convert the json to tabular data
    │   │
    │   ├── features    
    |   │   └── build_features.py  <- here we create features for modeling   
    │   │   └── feature_util.py    <- helper functions and global variables
    │   ├── models         
    │   │   │                 
    │   │   └── train_model.py     <- Trains the model
    |   |
    │   ├── util         
    │   │   │                 
    │   │   └── logger.py  
    |   │   └── paths.py
    │   │
    │   └── app  
    │       └── search.py          <- key matching functions
    |       └── ui.py              <- try out the matching functions 
    │
    └── tox.ini            


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

## 🚀 Getting Started

### 1. Clone the repository

First, clone the repository to your local machine:

```sh
git clone https://github.com/yourusername/instaffo-matching.git
cd instaffo-matching
```

### 2. Set up the environment
Create and activate a Conda or virtual environment:

Using Conda
```sh
make environment
```
This will create a conda environment on your machine if Anaconda is detected.

For local development purposes you can activate the environment:
Using Conda
```sh
conda activate instaffo-matching
```

### 3. Install dependencies
Install the required Python packages:

```sh
make requirements
```

### Add .env file

All you need to add to the file: 

```sh
WORKING_DIR="<local_path>/instaffo-matching"
```

## 🔄 Running the Pipeline

The pipeline consists of running the following scripts:


```sh
    # converts json to tabular dataset
	python src/data/make_dataset.py
    # creates training data 
	python src/features/build_features.py
    # trains the model
	python src/models/train_model.py
```

## ▶️ Run the Entire Pipeline

```sh
make pipeline
```

## 🕹️ Runing each step of the pipeline
You can run different parts of the pipeline using the Makefile commands:

### Make Dataset
```sh
make data
```

### Build Features
```sh
make features
```
### Train Model
```sh
make model
```


## 🌐 Starting Streamlit

### To start the Streamlit UI

```sh
make run
```

This will activate the appropriate environment and launch the Streamlit app. You can access it in your web browser at http://localhost:8501.

## 📽️ Demo

## Match

![Alt Text](/media/match.gif)


## Bulk Match 

![Alt Text](/media/bulk_match.gif)


## 🧹 Cleaning Up
To clean up compiled Python files:

```sh
make clean
```

## 📝 Linting
To check the code style using flake8:

```sh
make lint
```
