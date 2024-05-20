import streamlit as st
import pandas as pd
import joblib
from src.app.search import Search
from src.util.paths import Paths


def sample_dicts_from_df(df: pd.DataFrame, n: int) -> tuple[list[dict], list[dict]]:
    talents = df.talent.sample(n).tolist()
    jobs = df.job.sample(n).tolist()
    return talents, jobs


@st.cache_data
def load_data(data_path: str) -> pd.DataFrame:
    return pd.read_json(data_path)


def display_results(results: list[dict]) -> None:
    """
    Display the results in a 3-column layout.

    :param results: List of dictionaries containing the match results.
    """
    for i, result in enumerate(results):
        st.subheader(f"Match Example #{i + 1}")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.json(result["talent"])
        with col2:
            st.json(result["job"])
        with col3:
            st.write({"label": result["label"], "score": result["score"]})


def main():
    st.title("Talent-Job Matching System")

    # Load model and data
    search_system = Search(Paths.match_model_path)
    df = load_data(Paths.raw_dataset_path)

    # Mode selection
    mode = st.selectbox("Select Mode", ["Single Match", "Bulk Match"])

    if mode == "Single Match":
        st.subheader("Single Match")
        sample_size = st.number_input("Sample Size", min_value=1, value=1)
        talents, jobs = sample_dicts_from_df(df, sample_size)

        if st.button("Run Match"):
            results = []
            for talent, job in zip(talents, jobs):
                result = search_system.match(talent, job)
                results.append(result)
            display_results(results)
    else:
        st.subheader("Bulk Match")
        sample_size = st.number_input("Sample Size", min_value=1, value=5)
        talents, jobs = sample_dicts_from_df(df, sample_size)

        if st.button("Run Bulk Match"):
            results = search_system.match_bulk(talents, jobs)
            display_results(results)


if __name__ == "__main__":
    main()
