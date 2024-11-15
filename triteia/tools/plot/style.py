import os
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from glob import glob
from typing import List
from pathlib import Path


def set_matplotlib_style():
    from matplotlib import font_manager

    # get project dir
    project_dir = Path(__file__).resolve().parents[3]
    font_dirs = [os.path.join(project_dir, ".local", "fonts")]
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)

    pd.set_option("display.max_columns", 500)
    sns.set_style("ticks")
    font = {
        "font.family": "Roboto",
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
    sns.set_theme(style="whitegrid")
    sns.set_context("paper")
    sns.despine()


def autolabel(rects, ax, prec=1):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            f"{height:.{prec}f}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            size=9,
        )
