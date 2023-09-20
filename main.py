from math import ceil

import pandas as pd
import streamlit as st

from features import get_features
from model import USE_QUANTILES, get_fips_ranking
from viz import get_distribution, get_map
from zip_to_county import get_county

st.set_page_config(
    layout="wide",
    page_title="Migration Recommendation Tool - Team 147",
    page_icon=":us:",
)


cur_color = "black"
top_color = "magenta"


@st.cache_data(show_spinner=False)
def load_data():
    data = pd.read_csv("./data/combined_features.csv", dtype={"fips": str})
    return data


data = load_data()

features = get_features()

if "explore" not in st.session_state:
    st.session_state["explore"] = False


def toggle_explore():
    st.session_state["explore"] = not st.session_state["explore"]


if "results" not in st.session_state:
    st.session_state["results"] = False


def toggle_results():
    st.session_state["results"] = not st.session_state["results"]


if "k" not in st.session_state:
    st.session_state["k"] = 5


def more_results():
    st.session_state["k"] += 5


def fewer_results():
    st.session_state["k"] -= 5


st.sidebar.title("Migration Recommendation Tool")
st.sidebar.image(
    "media/logo.png",
    use_column_width=True,
)
st.sidebar.write("---")

st.title("Migration Recommendation Tool")

# Side Bar
county_input = st.sidebar.text_input(
    "Where are you moving from?", placeholder="Enter Zip Code"
)

county = None
if len(county_input) == 5:
    counties = get_county(county_input)

    if len(counties) == 0:
        st.sidebar.error("Seems like we can't find this zip code", icon="🚨")

    if len(counties) == 1:
        county = counties.to_dict("records")[0]

    if len(counties) > 1:
        county_name = st.sidebar.selectbox(
            "Choose a county:", options=counties["county"]
        )
        county = counties[counties["county"] == county_name].to_dict("records")[0]

user_income = st.sidebar.number_input(
    "What is your annual income?",
    min_value=0,
    max_value=1000000,
    value=80000,
    step=10000,
)

selected_features = st.sidebar.multiselect(
    "What are the most important factors for you? (choose by order of importance)",
    features["name"].to_list()[:-1],
)

st.sidebar.write("---")

chosen_features = []
target_values = {}
target_subheader = False
for name in selected_features:
    feature_info = features[features["name"] == name].to_dict("records")[0]
    chosen_features.append(feature_info["feature"])
    if feature_info["direction"] == "user":
        if not target_subheader:
            st.sidebar.subheader("What's your ideal fit?")
            target_subheader = True
        if feature_info["type"] == "categorical" or USE_QUANTILES:
            value = st.sidebar.select_slider(
                feature_info["label"],
                options=feature_info["options"],
                value=feature_info["options"][2],
            )
        else:
            step_size = 10000
            max_value = (
                ceil(data[feature_info["feature"]].quantile(0.99) / step_size)
                * step_size
            )
            median_value = (
                round(data[feature_info["feature"]].median().item() / step_size)
                * step_size
            )
            value = st.sidebar.slider(
                feature_info["label"],
                min_value=0,
                max_value=max_value,
                value=median_value,
                step=step_size,
            )
        target_values[feature_info["feature"]] = value


# Main Page

with st.expander("**Want to know more about this tool?** here's how it works:"):
    st.write(
        """
             Our Recommendation tool is the first to use the history of migrations along
             with your personal preferences to pick the right place for you.
             Analyzing millions of datapoints of Americans migrating within the US we've
             found the hidden patterns and coupled them with the factors behind them.
             Our tool coupled these patterns with big data from a variety of public resources,
             using advanced algorithms to help you choose the right place for you.
             """
    )

tag_style = "<span style='color: white; background-color: #004d95; border-radius: 5px; padding: 5px; margin: 1px; line-height: 2.2em; white-space: nowrap;'>"
with st.container():
    if isinstance(county, dict):
        st.write(f"Looks like you're from: **{county['county']}**")
    if selected_features:
        st.write("And you care about the following when looking for your new home:")
        st.write(
            f"{tag_style}{('</span> ' + tag_style).join(selected_features)}</span>",
            unsafe_allow_html=True,
        )

if st.session_state["results"] is True and len(chosen_features) > 0:
    with st.container():
        ranking_df, score_df = get_fips_ranking(
            data, county["fips"], user_income, chosen_features, target_values
        )

        st.plotly_chart(
            get_map(
                score_df,
                feature_info=features.iloc[-1],
                user_fips=county["fips"],
                ranking_df=ranking_df[: st.session_state["k"]],
                height=600,
            ),
            use_container_width=True,
        )

        st.write(
            f"<div style='text-align: center'><span style='color: {cur_color}; font-size: 28px; vertical-align: -5px; text-align: center;'>&#x25A0;</span>&nbsp;&nbsp;&nbsp;&nbsp;Current Location&nbsp;&nbsp;&nbsp;&nbsp;<span style='color: {top_color}; font-size: 28px; vertical-align: -5px; text-align: center;'>&#x25A0;</span>&nbsp;&nbsp;&nbsp;&nbsp;Top Locations</div>",
            unsafe_allow_html=True,
        )

        if st.session_state["k"] > 0:
            st.subheader(f"The top {st.session_state['k']} places for you are:")
            ranking_df = ranking_df.merge(data, on="fips").iloc[: st.session_state["k"]]
            for idx in ranking_df.index:
                with st.expander(f"**{idx+1}\. {ranking_df['name'][idx]}**"):
                    result_df = pd.concat(
                        [
                            features.iloc[:-1]["label"].reset_index(drop=True),
                            ranking_df[features.iloc[:-1]["feature"]]
                            .iloc[idx]
                            .reset_index(drop=True),
                        ],
                        axis=1,
                    )
                    result_df.columns = ["Metric", "Value"]
                    for i, row in result_df.iterrows():
                        if features["type"][i] == "percent" or features["type"][i] == "categorical":
                            result_df.loc[i, "Value"] = f"{row['Value']:.1%}"
                        elif features["type"][i] == "currency":
                            result_df.loc[i, "Value"] = "${:,.0f}".format(row["Value"])
                        else:
                            result_df.loc[i, "Value"] = "{:,.0f}".format(row["Value"])
                    st.write(result_df)
                    # for f_idx in features.iloc[:-1].index:
                    #     st.write(f"- **{features['label'][f_idx]}**: {ranking_df[features['feature'][f_idx]][idx]}")
        col1, col2, col3 = st.columns(3)
        if st.session_state["explore"] is False:
            col1.button(
                "Show Supporting Data", on_click=toggle_explore, use_container_width=True
            )
        else:
            col1.button(
                "Hide Supporting Data", on_click=toggle_explore, use_container_width=True
            )
        col2.button(
            "Show More Results", on_click=more_results, use_container_width=True
        )
        if st.session_state["k"] > 5:
            col3.button(
                "Show Fewer Results", on_click=fewer_results, use_container_width=True
            )
        elif st.session_state["k"] > 0:
            col3.button(
                "Hide Results", on_click=fewer_results, use_container_width=True
            )

        if st.session_state["explore"] is True:
            with st.container():
                explore_feature = st.selectbox(
                    "Whice metric would you like to explore?",
                    features.iloc[:-1]["name"],
                )
                feature_idx = features[features["name"] == explore_feature].index[0]
                data_idx = data.columns.get_loc(
                    features[features["name"] == explore_feature]["feature"].iloc[0]
                )
                # st.write('You selected:', explore_feature)
                st.plotly_chart(
                    get_map(
                        data,
                        feature_info=features.iloc[feature_idx],
                        feature_index=data_idx,
                        user_fips=county["fips"],
                        ranking_df=ranking_df[: st.session_state["k"]],
                        height=600,
                    ),
                    use_container_width=True,
                )

                st.plotly_chart(
                    get_distribution(
                        data,
                        feature_info=features.iloc[feature_idx],
                        feature_index=data_idx,
                        user_fips=county["fips"],
                        height=60,
                    ),
                    use_container_width=True,
                )

                st.write(
                    f"<div style='text-align: center'><span style='color: {cur_color}; font-size: 28px; vertical-align: -3px; text-align: center;'>|</span>&nbsp;&nbsp;&nbsp;&nbsp;Current Location&nbsp;&nbsp;&nbsp;&nbsp;<span style='color: {cur_color}; font-size: 28px; vertical-align: -5px; text-align: center;'>&#x25A0;</span>&nbsp;&nbsp;&nbsp;&nbsp;Current Location&nbsp;&nbsp;&nbsp;&nbsp;<span style='color: {top_color}; font-size: 28px; vertical-align: -5px; text-align: center;'>&#x25A0;</span>&nbsp;&nbsp;&nbsp;&nbsp;Top Locations</div>",
                    unsafe_allow_html=True,
                )

st.write("---")

# st.write("")
# st.write("")

if not st.session_state["results"]:
    st.subheader(":point_left: Start Here!")

st.write("")


def submit_button():
    if not isinstance(county, dict) or len(chosen_features) < 1:
        st.error("Please Complete the questions on the left side", icon="🚨")
    else:
        st.session_state["results"] = True
        # st.experimental_rerun()


st.sidebar.button(
    "Tell me where to move!", use_container_width=True, on_click=submit_button
)
