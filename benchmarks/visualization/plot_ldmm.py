import json

import matplotlib.pyplot as plt
import os
from typing import List
import numpy as np
import utils

colors = utils.colors_multiple

# Config:
INTERMEDIATE_PLOTS = False
filenames = ('ldmm_bench_16_new.json', 'ldmm_bench_32_new.json', 'ldmm_bench_64_new.json')
y_label = ("latency (ms)", "MFU (%)") # "MFU" or "latency"
x_label = "Total number of Adapters"

# number of models
nm = [1, 2, 4, 8, 16, 32, 64]
nm = [x * 2 for x in nm]

label_map = {"MFU (%)": "mfu", "latency (ms)": "elapsed"}

title = (f"{y_label[0]} for a 50/50 split of Adapters between LoRA and Delta",
            f"{y_label[1]} for a 50/50 split of Adapters between LoRA and Delta")
         
out_dir = "./"

data_names = []
# contains the results of the benchmark, the keys being the matrix sizes
data_metrics: dict[str, list[list]] = {}
measurements: dict[str, list[list]] = {}

for filename in filenames:
    with open(filename, "r") as data:
        data = json.load(data)
        # initialize data_metrics for the two matrix sizes
        for i in range(2):
            res = data["results"][i]
            data_name = f"{res["config"]["rank"]}, {res["config"]["n"]}x{res["config"]["m"]}"
            data_names.append(data_name)
            data_metrics[data_name] = {}
            measurements[data_name] = {}
            for key in res.keys():
                if key == "config":
                    continue
                data_metrics[data_name][key] = []
                measurements[data_name][key] = []
        
        for i, res in enumerate(data["results"]):
            if i >= len(data["results"]) - 2:
                continue # skip the first result, which is used to warm up the GPU
            for key in res.keys():
                data_name = f"{res["config"]["rank"]}, {res["config"]["n"]}x{res["config"]["m"]}"
                if key == "config":
                    continue
                if key == "measurements":
                    for metric in res[key].keys():
                        measurements[data_name][metric].append(res[key][metric])
                    continue
                data_metrics[data_name][key].append(res[key])


# create plot for each metric
plot_titles_and_labels = [
    (
        title[0],
        y_label[0],
        x_label,
    ),
    (
        title [1],
        y_label[1],
        x_label,
    )
]


def create_plot(
    title: str,
    x_label: str,
    y_label: str,
    names: list[str],
    metrics: dict[str, dict[str, list]],
):
    global colors

    # plot configurations
    plt.figure(figsize=(10, 6))
    plt.grid(True, which="major", axis="y", color="white", linestyle="-", linewidth=1.5)
    plt.xticks(nm, fontsize=12)
    plt.yticks(fontsize=12)
    plt.gca().set_facecolor("#dbdbdb")

    select = label_map[y_label]

    # plot data
    for j, name in enumerate(names):
        color = colors[(j//2)]
        y_data = metrics[name][select]

        # Compute confidence intervals
        ci_values = []
        for measurement_list in measurements[name][select]:
            ci = utils.compute_confidence_interval(measurement_list)
            ci_values.append(ci)
        ci_values = np.array(ci_values).T

        # Plot line
        if j % 2 == 0:
            plt.plot(nm, y_data, "o-", color=color, label=name)
        else: # plot dashed line
            plt.plot(nm, y_data, "o--", color=color, label=name)

        # Plot error bars at measurement points
        plt.errorbar(
            nm,
            y_data,
            yerr=ci_values,
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.5,
            capsize=4,
        )


    # add legend
    plt.legend(names, fontsize=13)

    # remove border
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    # set title and labels
    plt.title(title, fontsize=14, loc="left", pad=24, fontweight="bold")
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, rotation=0, labelpad=40, loc="bottom", fontsize=14)
    plt.gca().yaxis.set_label_coords(0, 1)
    plt.gca().get_yaxis().get_offset_text().set_x(-0.04)

    plt.savefig(
        f"{out_dir}/{filename.replace(".json","")}_{y_label}.svg", format="svg", dpi=300, bbox_inches="tight"
    )
    plt.show()
    #plt.close()

for title, y_label, x_label in plot_titles_and_labels:
    create_plot(title, x_label, y_label, data_names, data_metrics)
