import pandas as pd


def calculate_match_ratio(x, talent_col, job_col):
    if not x[job_col]:
        return 0
    return len(set(x[talent_col]).intersection(set(x[job_col]))) / len(set(x[job_col]))


def calculate_hit(x, talent_col, job_col):
    return 1 if x[talent_col] in x[job_col] else 0


def calculate_skill_diff(x, talent_skills_col, job_skills_col):
    talent_set = set(x[talent_skills_col])
    job_set = set(x[job_skills_col])
    return len(talent_set - job_set), len(job_set - talent_set)


def degree_level_comparison(x, talent_degree_scaled_col, job_degree_scaled_col):
    return x[talent_degree_scaled_col] - x[job_degree_scaled_col]


def degree_level_match(x, talent_degree_scaled_col, job_degree_scaled_col):
    return 1 if x[talent_degree_scaled_col] >= x[job_degree_scaled_col] else 0


def calculate_matched_languages_ratio(x, talent_lang_col, job_lang_col):
    talent_set = set(lang["title"] for lang in x[talent_lang_col])
    job_set = set(lang["title"] for lang in x[job_lang_col])
    if not job_set:
        return 0
    return len(talent_set & job_set) / len(job_set)


def calculate_required_languages(x, job_lang_col):
    return len(x[job_lang_col])


def calculate_rating_matches_ratio(x, talent_lang_col, job_lang_col):
    talent_lang_dict = {lang["title"]: lang["rating"] for lang in x[talent_lang_col]}
    proficiency_to_numeric = lambda rating: {
        "A1": 1,
        "A2": 2,
        "B1": 3,
        "B2": 4,
        "C1": 5,
        "C2": 6,
    }.get(rating, 0)
    matched_count = sum(
        1
        for lang in x[job_lang_col]
        if proficiency_to_numeric(talent_lang_dict.get(lang["title"], "A1"))
        >= proficiency_to_numeric(lang["rating"])
    )
    return matched_count / len(x[job_lang_col]) if x[job_lang_col] else 0


def scale_degrees(df, degree_col):
    # Map degrees to numeric values with 'apprenticeship' added
    degree_scale = {
        "none": 0,
        "apprenticeship": 1,
        "associate": 2,
        "bachelor": 3,
        "master": 4,
        "doctorate": 5,
    }
    df[degree_col + "_scaled"] = df[degree_col].apply(
        lambda x: degree_scale.get(x, -1)
    )  # Use -1 for unknown degrees
    return df


def append_match_ratio(df, talent_col, job_col):
    df["match_ratio"] = df.apply(
        lambda x: calculate_match_ratio(x, talent_col, job_col), axis=1
    )
    return df


def append_hit(df, talent_col, job_col):
    df["seniority_match"] = df.apply(
        lambda x: calculate_hit(x, talent_col, job_col), axis=1
    )
    return df


def append_skill_diff(df, talent_skills_col, job_skills_col):
    df["skill_diff_talent"], df["skill_diff_job"] = zip(
        *df.apply(
            lambda x: calculate_skill_diff(x, talent_skills_col, job_skills_col), axis=1
        )
    )
    return df


def append_degree_level_match(df, talent_degree_scaled_col, job_degree_scaled_col):
    df["degree_level_diff"] = df.apply(
        lambda x: degree_level_comparison(
            x, talent_degree_scaled_col, job_degree_scaled_col
        ),
        axis=1,
    )
    return df


def calculate_salary_expectation(x, salary_col_candidate, salary_col_job):
    delta = (x[salary_col_job] - x[salary_col_candidate]) / x[salary_col_job]
    return delta, 1 if delta < 0 else 0


def append_salary_expectation_features(df, salary_col_candidate, salary_col_job):
    df["salary_expectation_delta"], df["salary_expectation_over_budget"] = zip(
        *df.apply(
            lambda x: calculate_salary_expectation(
                x, salary_col_candidate, salary_col_job
            ),
            axis=1,
        )
    )
    return df


def append_matched_languages_ratio(df, talent_lang_col, job_lang_col):
    df["language_match_ratio"] = df.apply(
        lambda x: calculate_matched_languages_ratio(x, talent_lang_col, job_lang_col),
        axis=1,
    )
    return df


def append_required_languages(df, job_lang_col):
    df["required_languages"] = df.apply(
        lambda x: calculate_required_languages(x, job_lang_col), axis=1
    )
    return df


def append_of_rating_matches(df, talent_lang_col, job_lang_col):
    df["Language_rating_match_ratio"] = df.apply(
        lambda x: calculate_rating_matches_ratio(x, talent_lang_col, job_lang_col),
        axis=1,
    )
    return df


def append_skill_match_ratio(df, talent_skills_col, job_skills_col):
    df["skill_match_ratio"] = df.apply(
        lambda x: calculate_match_ratio(x, talent_skills_col, job_skills_col),
        axis=1,
    )
    return df


def append_degree_level_matched(df, talent_degree_scaled_col, job_degree_scaled_col):
    df["degree_level_matched"] = df.apply(
        lambda x: degree_level_match(
            x, talent_degree_scaled_col, job_degree_scaled_col
        ),
        axis=1,
    )
    return df


FEATURES = [
    "skill_match_ratio",
    "seniority_match",
    "skill_diff_talent",
    "skill_diff_job",
    "salary_expectation_delta",
    "salary_expectation_over_budget",
    "degree_level_matched",
    "degree_level_diff",
    "language_match_ratio",
    "required_languages",
    "Language_rating_match_ratio",
]

LABEL = "label"
