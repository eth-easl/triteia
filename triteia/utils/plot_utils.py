import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from glob import glob
from typing import List
from pathlib import Path


def set_matplotlib_style():
    pd.set_option("display.max_columns", 500)

    sns.set_style("ticks")
    font = {
        # "font.family": "Roboto",
        "font.size": 12,
    }
    sns.set_style(font)
    paper_rc = {
        "lines.linewidth": 3,
        "lines.markersize": 10,
    }
    sns.set_context("paper", font_scale=2, rc=paper_rc)
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42


def autolabel(rects, ax, prec=1, offset=0, fontsize=9):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            f"{height:.{prec}f}",
            xy=(rect.get_x() + rect.get_width() / 2, height+offset),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            size=fontsize,
        )


def read_csv_with_concat(path="./csv", file_name=None):
    file = Path(path, f"{file_name}.csv")

    if file.exists():
        # If original file exists, read it directly
        df = pd.read_csv(file)
        print(f"Reading {file_name}")
    else:
        # If original file does not exist, read all the split files
        split_files = sorted(glob(f"{path}/{file_name}-2023-*.csv"))
        print(f"Reading splitted files: {split_files}")
        df = pd.concat([pd.read_csv(split_file) for split_file in split_files])
        df.reset_index(drop=True, inplace=True)
    return df


def calculate_sum_cdf_axis100(df, dot_num=1000):
    """
    Calculate quantity percentile CDF, y-axis: 0-100%,
    """
    print("Parsing")
    data = df.melt(id_vars="Time", var_name="Server")
    data.dropna(subset=["value"], inplace=True)
    # data.sort_values('value', ascending=True, inplace=True)
    # data.reset_index(drop=True, inplace=True)

    y = np.linspace(0, 1, num=dot_num)
    x = data["value"].quantile(y).values
    y = y * 100
    return x, y


def calculate_num_cdf_customized_xaxis(df: pd.DataFrame, x_axis: List, key: str):
    """
    Calculate quantity percentile CDF with customized threshold of x-axis, y-axis: 0-100%,
    """
    # print("Parsing")
    data = df[[key]].copy()
    data.dropna(inplace=True)

    y = [len(data[data[key] <= x]) / len(data) * 100 for x in x_axis]

    return y


def calculate_sum_cdf_customized_xaxis(
    df: pd.DataFrame, x_axis: List, key: str, key_to_time=None
):
    """
    Calculate sum CDF with customized threshold of x-axis, y-axis: 0-100%,
    """
    # print("Parsing")
    if key_to_time is not None:
        data = df[[key, key_to_time]].copy()
        data["new"] = data[key] * data[key_to_time]
    else:
        data = df[[key]].copy()
        data["new"] = data[key]
    data.dropna(inplace=True)
    sum = data["new"].sum()
    y = [data[data[key] <= x]["new"].sum() / sum * 100 for x in x_axis]
    return y
