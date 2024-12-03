import json

import matplotlib.pyplot as plt
import os
import numpy as np
import utils

colors = utils.colors

# Config:
INTERMEDIATE_PLOTS = False
filename = 'ldmm_bench.json'
y_label = "latency" # "MFU" or "latency"
x_label = "Total number of models"

# number of models
nm = [1, 2, 4, 8, 16, 32, 64, 100]
nm = [x * 2 for x in nm]

label_map = {"MFU": "mfu", "latency": "elapsed"}

title = f"{y_label} for a 50/50 split of models between lora and sbmm"
out_dir = "./"

data_names = []
# contains the results of the benchmark, the keys being the matrix sizes
data_metrics: dict[str, list[list]] = {}
measurements: dict[str, list[list]] = {}


with open(filename, "r") as data:
    data = json.load(data)
    # initialize data_metrics for the two matrix sizes
    for i in range(2):
        res = data["results"][i]
        data_name = f"{res["config"]["m"]}x{res["config"]["n"]}"
        data_names.append(data_name)
        data_metrics[data_name] = {}
        measurements[data_name] = {}
        for key in res.keys():
            if key == "config":
                continue
            data_metrics[data_name][key] = []
            measurements[data_name][key] = []
    
    for i, res in enumerate(data["results"]):
        if i == 0 or i == 1:
            continue # skip the first result, which is used to warm up the GPU
        for key in res.keys():
            data_name = f"{res["config"]["m"]}x{res["config"]["n"]}"
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
        title,
        y_label,
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
    plt.figure(figsize=(7, 4))
    plt.grid(True, which="major", axis="y", color="white", linestyle="-", linewidth=1.5)
    plt.xticks(nm, fontsize=12)
    plt.yticks(fontsize=12)
    plt.gca().set_facecolor("#dbdbdb")

    select = label_map[y_label]

    # plot data
    for j, name in enumerate(names):
        color = colors[j]
        y_data = metrics[name][select]

        # Compute confidence intervals
        ci_values = []
        for measurement_list in measurements[name][select]:
            ci = utils.compute_confidence_interval(measurement_list)
            ci_values.append(ci)
        ci_values = np.array(ci_values).T

        # Plot line
        plt.plot(nm, y_data, "o-", color=color, label=name)

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
    plt.legend(names, fontsize=12)

    # remove border
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    # set title and labels
    plt.title(title, loc="left", pad=24, fontweight="bold")
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, rotation=0, labelpad=40, loc="bottom", fontsize=12)
    plt.gca().yaxis.set_label_coords(0, 1)
    plt.gca().get_yaxis().get_offset_text().set_x(-0.04)

    plt.savefig(
        f"{out_dir}/{str(title).replace(' ', '_').replace('/', '_').replace('\n', ' ')}_{len(names)}.pdf", format="pdf", bbox_inches="tight"
    )
    plt.show()
    #plt.close()

for title, y_label, x_label in plot_titles_and_labels:
    create_plot(title, x_label, y_label, data_names, data_metrics)
