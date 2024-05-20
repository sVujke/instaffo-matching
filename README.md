instaffo-matching
==============================

ML Case Study


## 📁 Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

## 🚀 Getting Started

### 1. Clone the repository

First, clone the repository to your local machine:

```sh
git clone https://github.com/yourusername/instaffo-matching.git
cd instaffo-matching
```

2. Set up the environment
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

3. Install dependencies
Install the required Python packages:

```sh
make requirements
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
