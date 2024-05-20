import pandas as pd
import numpy as np
import itertools
import joblib
from src.features.feature_utils import FEATURES
from src.features.build_features import process_data_pipeline
from src.data.make_dataset import normalize_and_merge
from src.util.paths import Paths


class Search:
    def __init__(self, model_path: str) -> None:
        """
        Initializes the Search class by loading the trained model.
        :param model_path: path to the saved model file.
        """
        self.model = joblib.load(model_path)

    def predict(self, df_processed: pd.DataFrame, is_bulk=False) -> tuple[int, float]:
        """
        Make predictions and adjust the score.
        :param df_processed: Processed DataFrame for prediction.
        :return: Tuple containing the predicted label and adjusted score.
        """
        # Predict using the model
        if not is_bulk:
            label = self.model.predict(df_processed)[0]
            score = self.model.predict_proba(df_processed)[0][1]  # Positive class score
            score = round(score, 3)
        else:
            label = self.model.predict(df_processed)
            score = self.model.predict_proba(df_processed)[:, 1]
            score = np.round(score, 3)
        return label, score

    def match(self, talent: dict, job: dict) -> dict:
        """
        Predicts the match between a single talent and a single job.

        :param talent: Dictionary containing talent features.
        :param job: Dictionary containing job features.
        :return: Dictionary containing talent, job, predicted label, and score.
        """

        # for bul we can use normalize and merge, but for single case we do it like this
        # TODO create wraper to make the code cleaner
        df_talent = pd.DataFrame([talent]).add_prefix("talent_")
        df_job = pd.DataFrame([job]).add_prefix("job_")
        df = pd.concat([df_talent, df_job], axis=1)

        df_processed = process_data_pipeline(
            df, FEATURES, label=[], ignore_label=True  # there is no label at this stage
        )

        label, score = self.predict(df_processed)

        return {"talent": talent, "job": job, "label": label, "score": score}

    def match_bulk(self, talents: list[dict], jobs: list[dict]) -> list[dict]:
        """
        Predicts the matches between multiple talents and jobs.

        :param talents: List of dictionaries, each containing talent features.
        :param jobs: List of dictionaries, each containing job features.
        :return: List of dictionaries containing talents, jobs, predicted labels, and scores.
        """

        # combinations
        talents, jobs = combine_and_separate(talents, jobs)

        df = normalize_and_merge(talents, jobs)

        df_processed = process_data_pipeline(
            df, FEATURES, label=[], ignore_label=True  # there is no label at this stage
        )

        df_output = pd.DataFrame({"talent": talents, "job": jobs})

        label, score = self.predict(df_processed, is_bulk=True)
        df_output["label"] = label
        df_output["score"] = score

        df_output.sort_values(by="score", ascending=False, inplace=True)
        return df_output.to_dict(orient="records")


def combine_and_separate(list1: list, list2: list) -> tuple[list, list]:
    """
    Creates all possible pairs of dictionaries from two lists (talent and job) and then separates them back into two lists.
    :param list1: First list of elements.
    :param list2: Second list of elements.
    :return: Two lists containing the paired elements from the input lists.
    """
    paired_list = list(itertools.product(list1, list2))
    list1_back, list2_back = zip(*paired_list)

    list1_back = list(list1_back)
    list2_back = list(list2_back)

    return list1_back, list2_back


def sample_dicts_from_df(df: pd.DataFrame, n: int) -> tuple[list[dict], list[dict]]:
    """
    Samples n records from a dataframe and returns them as a list of dictionaries for talents and jobs respectively.

    :param df: DataFrame containing 'talent' and 'job' columns.
    :param n: Number of records to sample.
    :return: Tuple of two lists - sampled talents and jobs.
    """
    talents = df.talent.sample(n).tolist()
    jobs = df.job.sample(n).tolist()
    return talents, jobs


if __name__ == "__main__":

    # Initialize search system
    search_system = Search(Paths.match_model_path)

    # Load and sample data
    df = pd.read_json(Paths.raw_dataset_path)
    talents, jobs = sample_dicts_from_df(df, 5)

    # Perform bulk match
    result = search_system.match_bulk(talents, jobs)
    print(result)
