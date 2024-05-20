import pandas as pd
from src.util.paths import Paths
from src.util.logger import logger
from src.features.feature_utils import (
    append_match_ratio,
    append_hit,
    append_skill_diff,
    scale_degrees,
    append_degree_level_match,
    append_degree_level_matched,
    append_salary_expectation_features,
    append_matched_languages_ratio,
    append_required_languages,
    append_of_rating_matches,
    append_skill_match_ratio,
    FEATURES,
    LABEL,
)  # if this was writen as a class the import would be more elegant for sure


def process_data_pipeline(df, features, label, ignore_label=False):
    if ignore_label:
        columns_to_keep = features
    else:
        columns_to_keep = features + [label]
    df = append_match_ratio(df, "talent_job_roles", "job_job_roles")
    df = append_hit(df, "talent_seniority", "job_seniorities")
    df = append_skill_diff(df, "talent_job_roles", "job_job_roles")
    df = scale_degrees(df, "talent_degree")
    df = scale_degrees(df, "job_min_degree")
    df = append_skill_match_ratio(df, "talent_job_roles", "job_job_roles")
    df = append_degree_level_match(df, "talent_degree_scaled", "job_min_degree_scaled")
    df = append_degree_level_matched(
        df, "talent_degree_scaled", "job_min_degree_scaled"
    )
    df = append_salary_expectation_features(
        df, "talent_salary_expectation", "job_max_salary"
    )
    df = append_matched_languages_ratio(df, "talent_languages", "job_languages")
    df = append_required_languages(df, "job_languages")
    df = append_of_rating_matches(df, "talent_languages", "job_languages")

    # Select only the required features and label for the final output
    df = df[columns_to_keep]

    # Convert the label to an 0 and 1
    df[label] = df[label].astype(int)

    return df


if __name__ == "__main__":

    logger.info("Loading interim dataset...")
    df = pd.read_pickle(Paths.interim_dataset_path)
    logger.info(f"data shape: {df.shape}")
    df = process_data_pipeline(df, FEATURES, LABEL)
    logger.info(f"data shape: {df.shape}")

    logger.info("Saving processed dataset to disk.")
    df.to_csv(Paths.processed_dataset_path, index=False)

    logger.info(f"the dataset contains the following features: {df.columns.tolist()}")

    logger.info("done!")
