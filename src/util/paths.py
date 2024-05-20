import os
from dotenv import load_dotenv, find_dotenv

# read dotenv
load_dotenv(find_dotenv())


# open Working director
class Paths:
    working_dir = os.getenv("WORKING_DIR")

    data_processed_dir = f"{working_dir}/data/processed"
    data_raw_dir = f"{working_dir}/data/raw"
    data_interim_dir = f"{working_dir}/data/interim"

    models_dir = f"{working_dir}/models"

    match_model_path = f"{models_dir}/my_model.joblib"
    raw_dataset_path = f"{data_raw_dir}/data.json"
    interim_dataset_path = f"{data_interim_dir}/data.pkl"
    processed_dataset_path = f"{data_processed_dir}/data.csv"
