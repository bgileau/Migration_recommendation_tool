import csv
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import get_colorscale, sample_colorscale
from plotly.subplots import make_subplots


def get_map(
    data: pd.DataFrame,
    feature_info: pd.Series,
    key_index: int = 0,
    feature_index: int = 1,
    user_fips: str = None,
    ranking_df: pd.DataFrame = None,
    width: int = None,
    height: int = None,
) -> go.Figure:
    """
    Generates a choropleth map with hover data of county names and the selected feature. Input
    dataframe should have the FIPS code in column 0 and the feature values in column 1 unless
    key_index and feature_index are specified.

    Args:
        data (pandas DataFrame): Pandas dataframe containing data for the choropleth map.
        feature_info (pandas Series): Information about the feature to be mapped.
        key_index (int, optional): Index of the key column in the dataframe containing the county FIPS codes. Defaults to 0.
        feature_index (int, optional): Index of the feature column in the dataframe to display on the map. Defaults to 1.
        user_fips: The county FIPS the user currently resides in. Defaults to None (auto).
        width (int, optional): Width of the plot in pixels. Defaults to None (auto).
        height (int, optional): Height of the plot in pixels. Defaults to None (auto).

    Returns:
        go.Figure: A choropleth map Plotly figure.
    """

    # Load county names
    with open("./data/county_name.csv", encoding="utf-8") as file:
        county_name = {row[0]: row[1] for row in csv.reader(file)}

    # Add county names to data
    data["county"] = data.iloc[:, key_index].map(county_name)

    # Load county and state shape data
    @st.cache_data(show_spinner=False)
    def load_shape_data():
        with open("./data/counties.json", encoding="utf-8") as file:
            counties = json.load(file)
        with open("./data/states.json", encoding="utf-8") as file:
            states = json.load(file)
        return counties, states

    counties, states = load_shape_data()

    # Create list of states
    state_list = [state["properties"]["name"] for state in states["features"]]

    # Calculate upper and lower fence
    q1 = data.iloc[:, feature_index].quantile(0.1)
    q3 = data.iloc[:, feature_index].quantile(0.9)
    feature_min = max(data.iloc[:, feature_index].min(), 2.5 * q1 - 1.5 * q3)
    feature_max = min(data.iloc[:, feature_index].max(), 2.5 * q3 - 1.5 * q1)

    if feature_info["type"] == "percent" or feature_info["type"] == "categorical":
        tickformat = ".0%"
        hovertemplate = "%{z:.0%}<extra>%{customdata}</extra>"
    elif feature_info["type"] == "currency":
        tickformat = "$~s"
        hovertemplate = "%{z:$.3s}<extra>%{customdata}</extra>"
    else:
        tickformat = None
        hovertemplate = "%{z:.3s}<extra>%{customdata}</extra>"

    cur_color = "black"
    top_color = "magenta"
    # Create figure
    fig = make_subplots(rows=1, cols=1)

    # Add counties trace
    fig.add_trace(
        go.Choropleth(
            geojson=counties,
            locations=data.iloc[:, key_index],
            z=data.iloc[:, feature_index],
            zmin=feature_min,
            zmax=feature_max,
            colorscale=feature_info["color_scale"],
            reversescale=bool(feature_info["color_scale_rev"]),
            marker={"line": {"color": "rgba(0,0,0,0)", "width": 0}},
            colorbar={
                "title": feature_info["label"],
                "titleside": "right",
                "titlefont": {"size": 16},
                "tickfont": {"size": 14},
                "tickformat": tickformat,
            },
            name=feature_info["label"],
            customdata=data["county"],
            hovertemplate=hovertemplate,
            hoverlabel={"font_size": 14},
        )
    )

    # Outline states
    fig.add_trace(
        go.Choropleth(
            geojson=states,
            locations=state_list,
            featureidkey="properties.name",
            z=[1] * len(state_list),
            colorscale=["rgba(255,0,0,0)", "rgba(255,0,0,0)"],
            marker={"line": {"color": "white", "width": 1}},
            hoverinfo="skip",
            showscale=False,
            showlegend=False,
        )
    )

    # Outline ranked counties
    if ranking_df is not None:
        # Add counties trace
        fig.add_trace(
            go.Choropleth(
                geojson=counties,
                locations=ranking_df.iloc[:, key_index],
                z=[1] * len(ranking_df),
                colorscale=[top_color, top_color],
                marker={"line": {"color": top_color, "width": 1}},
                hoverinfo="skip",
                showscale=False,
                showlegend=False,
            )
        )

    # Outline current county
    if user_fips is not None:
        fig.add_trace(
            go.Choropleth(
                geojson=counties,
                locations=[user_fips],
                z=[1],
                colorscale=[cur_color, cur_color],
                marker={"line": {"color": cur_color, "width": 1}},
                hoverinfo="skip",
                showscale=False,
                showlegend=False,
            )
        )

    # Update layout
    fig.update_layout(
        geo={"scope": "usa", "subunitcolor": "rgba(255,0,0,0)"},
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        width=width,
        height=height,
    )

    return fig


def get_distribution(
    data: pd.DataFrame,
    feature_info: pd.Series,
    feature_index: str = 1,
    user_fips: str = None,
    width: int = None,
    height: int = None,
) -> go.Figure:
    """
    Generates a histogram of a feature.

    Args:
        data (pandas DataFrame): Value data for the choropleth map.
        feature_info (pandas Series): Information about the feature to be mapped.
        feature_index (str): Index of the feature column in data to show the distribution for.
        user_fips: The county FIPS the user currently resides in. Defaults to None (auto).
        width (int, optional): Width of the plot in pixels. Defaults to None (auto).
        height (int, optional): Height of the plot in pixels. Defaults to None (auto).

    Returns:
        go.Figure: A histogram Plotly figure.
    """

    def format_tick(t):
        if feature_info["type"] == "percent":
            return f"{t:.1f}%"
        elif feature_info["type"] == "currency":
            return "${:,.0f}".format(t)
        else:
            return "{:,.0f}".format(t)

    if feature_info["type"] == "percent" or feature_info["type"] == "categorical":
        tickformat = ".0%"
        hovertemplate = "%{x:.0%}<extra></extra>"
    elif feature_info["type"] == "currency":
        tickformat = "$~s"
        hovertemplate = "%{x:$.3s}<extra></extra>"
    else:
        tickformat = None
        hovertemplate = "%{x:.3s}<extra></extra>"

    tickvals = None
    ticktext = None
    col_data = data.iloc[:, feature_index]
    min_val = col_data.min()
    max_val = col_data.max()
    if min_val >= 0 and max_val <= 100:
        start = 0
        end = 100
        x = data[data["fips"] == user_fips].iloc[0, feature_index]
    elif np.log10(max_val) - np.log10(min_val) < 2:
        start = min_val
        end = max_val
        x = data[data["fips"] == user_fips].iloc[0, feature_index]
    else:
        start = np.log10(min_val)
        end = np.log10(max_val)
        col_data = np.log10(col_data)
        x = np.log10(data[data["fips"] == user_fips].iloc[0, feature_index])
        if feature_info["ticks"] is not None:
            tickvals = np.log10(feature_info["ticks"])
            ticktext = [format_tick(t) for t in feature_info["ticks"]]

    color = sample_colorscale(get_colorscale(feature_info["color_scale"]), 0.5)[0]
    cur_color = "black"

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=col_data,
            histfunc="count",
            histnorm="percent",
            marker_color=color,
            xbins={"start": start, "end": end},
            hovertemplate=hovertemplate,
            hoverlabel={"font_size": 14},
        )
    )

    if user_fips is not None:
        fig.add_vline(x=x, line_color=cur_color)

    fig.update_layout(
        xaxis=dict(
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            tickformat=tickformat,
        ),
        yaxis={
            "showticklabels": False,
            "showgrid": False,
            "zeroline": False,
            "showline": False,
        },
        plot_bgcolor="white",
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        width=width,
        height=height,
    )

    return fig
