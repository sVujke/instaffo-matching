import pandas as pd
from src.util.logger import logger
from src.util.paths import Paths


def normalize_and_merge(df_talent: pd.DataFrame, df_job: pd.DataFrame) -> pd.DataFrame:
    """
    Merges the talent and job dataframes into a single dataframe.
    """
    df_talent = pd.json_normalize(df_talent).add_prefix("talent_")
    df_job = pd.json_normalize(df_job).add_prefix("job_")
    df_merged = df_talent.merge(df_job, left_index=True, right_index=True)
    return df_merged


if __name__ == "__main__":

    logger.info(
        "Normalizing json dictionaries and merging them into a single dataframe."
    )

    df = pd.read_json(Paths.raw_dataset_path)

    logger.info(f"data shape: {df.shape}")

    df_merged = normalize_and_merge(df.talent, df.job)
    print(df_merged.head(5).T)
    df = df.merge(df_merged, left_index=True, right_index=True)

    df.to_pickle(Paths.interim_dataset_path)

    logger.info(f"data shape: {df.shape}")
    logger.info(f"data shape: {df.info()}")

    logger.info("dataset created!")
