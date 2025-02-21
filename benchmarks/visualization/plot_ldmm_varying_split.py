import json

import matplotlib.pyplot as plt
import os
from typing import List
import numpy as np
import utils

colors = utils.colors

# Config:
version = "full" # "0_10" or "full" or "10_0"
y_label = ("latency (ms)", "MFU (%)") # "MFU" or "latency"
x_label = "Number of Delta queries"

if version == "full":
    nm_sbmm = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    nm_lora = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0]
    filenames = ('ldmm_bench_varying_split_16_full_new.json', 'ldmm_bench_varying_split_32_full_new.json', 'ldmm_bench_varying_split_64_full_new.json')
elif version == "0_10":
    nm_sbmm = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    nm_lora = [100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90]
    filenames = ('ldmm_bench_varying_split_16_0-10_new.json', 'ldmm_bench_varying_split_32_0-10_new.json', 'ldmm_bench_varying_split_64_0-10_new.json')
elif version == "10_0":
    nm_sbmm = [100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90]
    nm_lora = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    filenames = ('ldmm_bench_varying_split_16_100-90_new.json', 'ldmm_bench_varying_split_32_100-90_new.json', 'ldmm_bench_varying_split_64_100-90_new.json')
else:
    nm_sbmm = []
    nm_lora = []

nm = nm_sbmm

label_map = {"MFU (%)": "mfu", "latency (ms)": "elapsed"}

title = (f"{y_label[0]} for 100 queries split between LoRA and Delta",
            f"{y_label[1]} for 100 queries split between LoRA and Delta")
out_dir = "./"

data_names = []
# contains the results of the benchmark, the keys being the matrix sizes
data_metrics: dict[str, dict[str, list]] = {}
measurements: dict[str, dict[str, list]] = {}



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
            # if i == 0:
            #     continue # skip the first result, which is used to warm up the GPU
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
        title[1],
        y_label[1],
        x_label,
    )
]

def create_plot(
    title: str,
    x_label: str,
    y_label: str,
    names: List[str],
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
        #print size of nm and y_data
        print(len(nm), len(y_data))
        
        # Plot line
        if j % 2 == 0:
            plt.plot(nm, y_data, "o-", color=color, label=name)
        else: # plot dotted line
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
    plt.legend(names, fontsize=12)

    # remove border
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    # set title and labels
    plt.title(title, fontsize = 14, loc="left", pad=24, fontweight="bold")
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, rotation=0, labelpad=40, loc="bottom", fontsize=14)
    plt.gca().yaxis.set_label_coords(0, 1)
    plt.gca().get_yaxis().get_offset_text().set_x(-0.04)

    plt.savefig(
        f"{out_dir}/{filename.replace(".json","")}_{y_label}_{version}.svg", format="svg", dpi=300, bbox_inches="tight"
    )
    plt.show()
    #plt.close()

for title, y_label, x_label in plot_titles_and_labels:
    create_plot(title, x_label, y_label, data_names, data_metrics)
