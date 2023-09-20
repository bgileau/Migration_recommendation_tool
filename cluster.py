import numpy as np
import pandas as pd
import streamlit as st
from census import Census
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

CENSUS_DATA_FILE = "./data/census_data.csv"
COLS_ORDER = [
    "fips",
    "PER CAPITA INCOME",
    "MEDIAN HOUSEHOLD INCOME",
    "INDIVIDUAL INCOME diff",
    "MEDIAN INCOME diff",
]


def save_census_data():
    api_key = ""

    c = Census(api_key)
    data_list = c.acs5.get(
        [
            "B07010_001E",  # Current INDIVIDUAL INCOME
            "B07410_001E",  # Prior INDIVIDUAL INCOME
            "B07011_001E",  # Current MEDIAN INCOME
            "B07411_001E",  # Prior MEDIAN INCOME
            "B19013_001E",  # MEDIAN HOUSEHOLD INCOME
            "B19301_001E",  # PER CAPITA INCOME
        ],
        {"for": "county:*", "in": "state:*"},
    )
    df_economic = pd.DataFrame(data_list)
    df_economic["fips"] = df_economic["state"] + df_economic["county"]

    df_economic["INDIVIDUAL INCOME diff"] = (
        df_economic["B07010_001E"] - df_economic["B07410_001E"]
    )  # current - prior
    df_economic["MEDIAN INCOME diff"] = (
        df_economic["B07011_001E"] - df_economic["B07411_001E"]
    )  # current - prior

    df_economic = df_economic.drop(
        ["B07010_001E", "B07410_001E", "B07011_001E", "B07411_001E", "state", "county"],
        axis=1,
    )
    df_economic = df_economic.rename(
        columns={
            "B19013_001E": "MEDIAN HOUSEHOLD INCOME",
            "B19301_001E": "PER CAPITA INCOME",
        }
    )

    df_economic = df_economic[COLS_ORDER]

    df_economic = df_economic.dropna()

    df_economic.to_csv(CENSUS_DATA_FILE, index=False)


@st.cache_data(show_spinner=False)
def get_cluster(user_annual_income: int, user_prior_fips: str):
    user_financial_data = np.array([user_annual_income] * 2 + [0] * 2).reshape(1, -1)

    df_economic = pd.read_csv(
        CENSUS_DATA_FILE,
        dtype={"fips": str},
    )

    prior_metrics = (
        df_economic[df_economic["fips"] == user_prior_fips].iloc[0, 1:].to_list()
    )

    df_diff = df_economic[df_economic["fips"] != user_prior_fips].copy()
    for i, prior_metric in enumerate(prior_metrics):
        df_diff.iloc[:, i + 1] = df_diff.iloc[:, i + 1] - prior_metric

    #######################################################################################
    # Handle outliers (check if fully needed)
    def remove_outliers(df, quartile=0.02, multiplier=1):
        clean_df = df.copy()
        for col in df_diff.iloc[:, 1:].columns:
            Q1 = df[col].quantile(quartile)
            Q3 = df[col].quantile(1 - quartile)
            IQR = Q3 - Q1

            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR

            clean_df = clean_df[
                (clean_df[col] >= lower_bound) & (clean_df[col] <= upper_bound)
            ]

        return clean_df

    # Apply the remove_outliers function to your dataset with a 2.5 multiplier
    df_diff = remove_outliers(df_diff)

    # Scale the financial metrics
    scaler = StandardScaler()
    scaled_df = pd.DataFrame(
        scaler.fit_transform(df_diff.iloc[:, 1:].values), columns=COLS_ORDER[1:]
    )

    goal_explained_variance = 0.9
    pca = PCA(
        n_components=goal_explained_variance, svd_solver="full"
    )  # looking to select # that can explain 90% of variance
    x_pca = pca.fit_transform(scaled_df.values)

    # Determine the optimal number of clusters using the elbow method (optional)
    # This step can help you decide how many clusters to use in K-means
    distortions = []
    K = range(1, 10)
    for k in K:
        kmeans = KMeans(n_clusters=k, n_init="auto", random_state=0)
        kmeans.fit(x_pca)
        distortions.append(kmeans.inertia_)

    # Set your threshold value
    threshold = 20  # 20% change threshold

    # Calculate percentage changes in distortions
    percent_changes = np.abs(np.diff(distortions) / distortions[:-1] * 100)

    # Find the index where the percentage change falls below the threshold
    k_optimal_idx = np.where(percent_changes < threshold)[0][0] + 1

    # Add 1 to the index because cluster numbering starts from 1
    k_optimal_initial = K[k_optimal_idx]

    possible_cluster_values = [2, 3, 4, 5, 6, 7]
    k_optimal_initial = min(k_optimal_initial, 7)

    def evaluate_metrics(x_pca, labels):
        sil_score = metrics.silhouette_score(x_pca, labels)
        ch_score = metrics.calinski_harabasz_score(x_pca, labels)
        db_score = metrics.davies_bouldin_score(x_pca, labels)

        return sil_score, ch_score, db_score

    def normalize_score(score, min_score, max_score):
        return (score - min_score) / (max_score - min_score)

    def decision_function(
        sil_score,
        ch_score,
        db_score,
        min_sil_score,
        max_sil_score,
        min_ch_score,
        max_ch_score,
        min_db_score,
        max_db_score,
        sil_weight=0.4,
        ch_weight=0.4,
        db_weight=0.2,
    ):
        norm_sil_score = normalize_score(sil_score, min_sil_score, max_sil_score)
        norm_ch_score = normalize_score(ch_score, min_ch_score, max_ch_score)
        norm_db_score = normalize_score(db_score, min_db_score, max_db_score)
        decision_val = (
            sil_weight * norm_sil_score
            + ch_weight * norm_ch_score
            - db_weight * norm_db_score
        )

        # print(norm_sil_score, norm_ch_score, norm_db_score, decision_val)

        return decision_val

    best_decision_value = -float("inf")
    best_k_decision = -1

    # Store the scores in dictionaries for easy access
    sil_scores = {}
    ch_scores = {}
    db_scores = {}

    for k in possible_cluster_values:
        kmeans = KMeans(n_clusters=k, n_init="auto", random_state=0)
        df_diff_cluster = df_diff.copy()
        df_diff_cluster["cluster"] = kmeans.fit_predict(x_pca)

        labels = df_diff_cluster["cluster"]
        sil_score, ch_score, db_score = evaluate_metrics(x_pca, labels)

        sil_scores[k] = sil_score
        ch_scores[k] = ch_score
        db_scores[k] = db_score

    min_sil_score, max_sil_score = min(sil_scores.values()), max(sil_scores.values())
    min_ch_score, max_ch_score = min(ch_scores.values()), max(ch_scores.values())
    min_db_score, max_db_score = min(db_scores.values()), max(db_scores.values())

    decision_value_dict = {}

    for k in possible_cluster_values:
        decision_value = decision_function(
            sil_scores[k],
            ch_scores[k],
            db_scores[k],
            min_sil_score=min_sil_score,
            max_sil_score=max_sil_score,
            min_ch_score=min_ch_score,
            max_ch_score=max_ch_score,
            min_db_score=min_db_score,
            max_db_score=max_db_score,
        )
        decision_value_dict[k] = decision_value

        if decision_value > best_decision_value:
            best_decision_value = decision_value
            best_k_decision = k

    kmeans = KMeans(n_clusters=best_k_decision, n_init="auto", random_state=0).fit(
        x_pca
    )

    # Now, given a user's current location and financial information, find the best cluster

    # Scale user's financial data using the same scaler
    scaled_user_financial_data = scaler.transform(user_financial_data)
    scaled_user_financial_data_pca = pca.transform(scaled_user_financial_data)

    # Find the best matching cluster
    user_cluster_pca = kmeans.predict(scaled_user_financial_data_pca)
    # Get recommended counties from the best matching cluster
    cluster = df_diff_cluster[df_diff_cluster["cluster"] == user_cluster_pca[0]]["fips"]

    return cluster


if __name__ == "__main__":
    # print(get_cluster(150000, "06037"))
    # save_census_data()
    df_economic = pd.read_csv(
        CENSUS_DATA_FILE,
        dtype={"fips": str},
    )

    print(df_economic.head())
    print(df_economic.dtypes)
