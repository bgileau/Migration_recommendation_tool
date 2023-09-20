from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from cluster import get_cluster
from features import get_features

USE_QUANTILES = False
MAX_SCORE = 1


@st.cache_data(show_spinner=False)
def get_fips_ranking(
    data: pd.DataFrame,
    user_fips: str,
    user_income: int,
    selected_features: list[str],
    target_values: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generates ranking and scoring of counties based on the user's input.

    Args:
        data (pd.DataFrame): The feature data in a pandas DataFrame.
        user_fips (str): The county FIPS code for where the user currently lives.
        user_income (int): The user's annual income.
        selected_features (list[str]): An ordered list of the features the user selected.
        target_values (dict): A dictionary with target values for "user" direction features.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Ranking df and scoring df.
    """

    # Import combined dataframe and scorecard inputs

    score_param = get_features()

    ##############################################Implementing scoring algorithm
    ##Calculate weights
    weights = []

    no_features = len(selected_features)

    for i in range(no_features):
        weight = ((0.5 + 2 * (no_features - i)) + (0.5 * 2 ** (no_features - i))) / 2

        weights.append(weight)

    weights = np.array(weights) / sum(weights)

    weights = list(weights)

    ##########Convert scorecard parameters to dictionaries
    # band dict
    score_band = dict(zip(score_param["feature"], score_param["bands"]))

    # Direction dict
    score_dir = dict(zip(score_param["feature"], score_param["direction"]))
    # User defined dict
    user_def = score_param[["feature", "options"]][pd.notna(score_param["options"])]

    user_def["options"] = [{j: i for i, j in enumerate(k)} for k in user_def["options"]]

    score_user_def = dict(zip(user_def["feature"], user_def["options"]))
    # Assign values for negative and positiv
    dir_value = {"negative": -1, "user": 1, "positive": 1}

    max_score = score_param["bands"].max()

    ##Create dict with variables and weights
    features_weights = dict(zip(selected_features, weights))

    # Extract selected features from original df
    df_calc = data[selected_features + ["fips"]].copy()

    # Remove any row with missing values
    df_calc = df_calc.dropna()
    #####Putting the features in quantiles based on scorecard input setup
    ##The quantile for variables that are not user defined is the score.
    df_calc["score"] = 0  # initialisation
    if USE_QUANTILES:
        for col in selected_features:
            ticks = score_param[score_param["feature"] == col].iloc[0]["ticks"]
            if isinstance(ticks, list):
                df_calc[col + "_q"] = (
                    pd.cut(
                        df_calc[col] * dir_value[score_dir[col]],
                        bins=([-np.inf] + ticks + [np.inf]),
                        labels=False,
                    )
                    + 1
                )
            else:
                df_calc[col + "_q"] = (
                    pd.qcut(
                        df_calc[col] * dir_value[score_dir[col]],
                        q=score_band[col],
                        labels=False,
                    )
                    + 1
                )

            # However, for user defined, there is the need for adjustments
            if col in score_user_def:  # converting quantile to score for user defined
                df_calc[col + "_q"] = max_score - abs(
                    score_user_def[col][target_values[col]] - df_calc[col + "_q"] + 1
                ) * (max_score / score_band[col])

            df_calc["score"] += df_calc[col + "_q"] * features_weights[col]

        # Scale score
        df_calc["score"] = df_calc["score"] * (MAX_SCORE / max_score)

    else:
        for col in selected_features:
            scaler = MinMaxScaler(feature_range=(0, MAX_SCORE))
            if col in target_values:
                feature_info = score_param[score_param["feature"] == col].to_dict(
                    "records"
                )[0]
                if feature_info["type"] == "categorical":
                    opts = score_param[score_param["feature"] == col]["options"].values[
                        0
                    ]
                    le = LabelEncoder()
                    opts_labels = le.fit_transform(opts)
                    opts_scaled = scaler.fit_transform(opts_labels.reshape(-1, 1))
                    target_value = dict(zip(opts, opts_scaled.ravel()))[
                        target_values[col]
                    ]
                    df_calc[col + "_s"] = MAX_SCORE - abs(
                        target_value
                        - scaler.fit_transform(df_calc[col].values.reshape(-1, 1))
                    )
                else:
                    if feature_info["scale"] == "log":
                        col_data = np.log10(df_calc[col])
                        target_value = np.log10(target_values[col])
                    else:
                        col_data = df_calc[col]
                        target_value = target_values[col]
                    scaled_data = scaler.fit_transform(col_data.values.reshape(-1, 1))
                    df_calc[col + "_s"] = MAX_SCORE - abs(
                        scaler.transform([[target_value]]) - scaled_data
                    )
            else:
                if dir_value[score_dir[col]] == 1:
                    df_calc[col + "_s"] = scaler.fit_transform(
                        df_calc[col].values.reshape(-1, 1)
                    )
                else:
                    df_calc[col + "_s"] = MAX_SCORE - scaler.fit_transform(
                        df_calc[col].values.reshape(-1, 1)
                    )
            df_calc["score"] += df_calc[col + "_s"] * features_weights[col]

    ###Present DataFrame for visualisation
    output_cols = ["fips", "score"]
    score_df = df_calc[output_cols].copy()
    cluster = get_cluster(user_income, user_fips)

    ###Present ordered clustering result for recommendation
    # add score to clustering results
    clustering_rec = pd.merge(
        cluster,
        score_df[output_cols],
        on="fips",
        how="left",
    )

    ranking_df = clustering_rec.sort_values(by="score", ascending=False)

    return ranking_df, score_df
