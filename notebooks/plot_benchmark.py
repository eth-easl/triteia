import json
import pandas as pd
import numpy as np
from triteia.utils.plot_utils import set_matplotlib_style, autolabel
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

set_matplotlib_style()
cmp = sns.color_palette("tab10")
sns.set_style("ticks")
trials = 1
matrix_sizes = ["2k", "4k"]
num_models = [16, 64]
distribution = "uniform"
df = []
for id in range(0, trials):
    for ms in matrix_sizes:
        for num_model in num_models:
            with open(f".local/run_3/{ms}_native_{num_model}_{id}.json", "r") as fp:
                ibmm_data = json.load(fp)
            with open(f".local/run_3/{ms}_forloop_{num_model}_{id}.json", "r") as fp:
                naive_data = json.load(fp)
            with open(f".local/run_3/{ms}_forloop_{num_model}_{id}.json", "r") as fp:
                naive_data = json.load(fp)
            naive_total_time = sum([x["TAvg (ns)"] * x["Count"] for x in naive_data])
            ibmm_total_time = sum([x["TAvg (ns)"] * x["Count"] for x in ibmm_data])
            
            naive_kernel_calls = [
                x for x in naive_data if "marlin::Marlin_2_4" in x["Kernel Name"]
            ]
            ibmm_kernel_calls = [
                x for x in ibmm_data if "marlin::IBMM_2_4" in x["Kernel Name"]
            ]
            naive_kernel_time = sum(
                [x["KAvg (ns)"] * x["Count"] for x in naive_kernel_calls]
            )
            ibmm_kernel_time = sum(
                [x["KAvg (ns)"] * x["Count"] for x in ibmm_kernel_calls]
            )
            df.append(
                {
                    "size": ms,
                    "num_model": num_model,
                    "approach": "Naive",
                    "total_time": naive_total_time / 1e6,
                    "kernel_time": naive_kernel_time / 1e6,
                }
            )
            df.append(
                {
                    "size": ms,
                    "num_model": num_model,
                    "approach": "SBMM",
                    "total_time": ibmm_total_time / 1e6,
                    "kernel_time": ibmm_kernel_time / 1e6,
                }
            )
df = pd.DataFrame(df)
fig, (ax1, ax2) = plt.subplots(
    ncols=2, nrows=1, constrained_layout=True, figsize=(9, 3.75), sharey=True
)
width = 0.30
positions = np.arange(2)
positions_naive = [x - width / 2 for x in positions]
positions_sbmm = [x + width / 2 for x in positions]
small_df = df[df["size"] == "2k"]
large_df = df[df["size"] == "4k"]
small_groups = small_df.groupby("num_model")
large_groups = large_df.groupby("num_model")

for pos_naive, pos_sbmm, (num_model, group) in zip(
    positions_naive, positions_sbmm, small_groups
):
    naive_group = group[group["approach"] == "Naive"]
    sbmm_group = group[group["approach"] == "SBMM"]
    # Plot for 'Naive'
    p1 = ax1.bar(
        pos_naive,
        naive_group["kernel_time"].mean(),
        width,
        label="Kernel Time (Naive)",
        color=cmp[0],
        yerr=naive_group["kernel_time"].std(),
        linewidth=1,
        edgecolor="k",
    )

    p2 = ax1.bar(
        pos_naive,
        (naive_group["total_time"] - naive_group["kernel_time"]).mean(),
        width,
        bottom=naive_group["kernel_time"].mean(),
        color=cmp[0],
        alpha=0.5,
        label="Total Time (Naive)",
        yerr=(naive_group["total_time"] - naive_group["kernel_time"]).std(),
        linewidth=1,
        edgecolor="k",
    )

    # Plot for 'SBMM'
    p3 = ax1.bar(
        pos_sbmm,
        sbmm_group["kernel_time"].mean(),
        width,
        label="Kernel Time (SBMM)",
        color=cmp[1],
        yerr=sbmm_group["kernel_time"].std(),
        linewidth=1,
        edgecolor="k",
    )
    p4=ax1.bar(
        pos_sbmm,
        (sbmm_group["total_time"] - sbmm_group["kernel_time"]).mean(),
        width,
        bottom=sbmm_group["kernel_time"].mean(),
        color=cmp[1],
        alpha=0.5,
        label="Total Time (SBMM)",
        yerr=(sbmm_group["total_time"] - sbmm_group["kernel_time"]).std(),
        linewidth=1,
        edgecolor="k",
    )

for pos_naive, pos_sbmm, (num_model, group) in zip(
    positions_naive, positions_sbmm, large_groups
):
    naive_group = group[group["approach"] == "Naive"]
    sbmm_group = group[group["approach"] == "SBMM"]
    # Plot for 'Naive'
    # error bar, use the standard deviation
    p5=ax2.bar(
        pos_naive,
        naive_group["kernel_time"].mean(),
        width,
        label="Kernel Time (Naive)",
        color=cmp[0],
        yerr=naive_group["kernel_time"].std(),
        linewidth=1,
        edgecolor="k",
    )
    p6=ax2.bar(
        pos_naive,
        (naive_group["total_time"] - naive_group["kernel_time"]).mean(),
        width,
        bottom=naive_group["kernel_time"].mean(),
        color=cmp[0],
        alpha=0.5,
        label="Total Time (Naive)",
        yerr=(naive_group["total_time"] - naive_group["kernel_time"]).std(),
        linewidth=1,
        edgecolor="k",
    )

    # Plot for 'SBMM'
    p7=ax2.bar(
        pos_sbmm,
        sbmm_group["kernel_time"].mean(),
        width,
        label="Kernel Time (SBMM)",
        color=cmp[1],
        yerr=sbmm_group["kernel_time"].std(),
        linewidth=1,
        edgecolor="k",
    )
    p8=ax2.bar(
        pos_sbmm,
        (sbmm_group["total_time"] - sbmm_group["kernel_time"]).mean(),
        width,
        bottom=sbmm_group["kernel_time"].mean(),
        color=cmp[1],
        alpha=0.5,
        label="Total Time (SBMM)",
        yerr=(sbmm_group["total_time"] - sbmm_group["kernel_time"]).std(),
        linewidth=1,
        edgecolor="k",
    )

# autolabel(p1, ax1)
# autolabel(p2, ax1)
# autolabel(p3, ax1)
# autolabel(p4, ax1)

# autolabel(p5, ax2)
# autolabel(p6, ax2)
# autolabel(p7, ax2)
# autolabel(p8, ax2)

group_center_positions = [(n + s) / 2 for n, s in zip(positions_naive, positions_sbmm)]
ax1.set_xticks(group_center_positions)
ax1.set_xticklabels([f"{num_model} Models" for num_model in small_groups.groups.keys()])

group_center_positions = [(n + s) / 2 for n, s in zip(positions_naive, positions_sbmm)]
ax2.set_xticks(group_center_positions)
ax2.set_xticklabels([f"{num_model} Models" for num_model in small_groups.groups.keys()])

# # Add some text for labels, title and custom x-axis tick labels, etc.
ax1.set_xlabel("2048x2048")
ax1.set_ylabel("Time (ms)")
ax1.grid(axis="y", linestyle=":")

ax2.set_xlabel("4096x4096")
ax2.grid(axis="y", linestyle=":")

# Only add legends for the first instance of each type
handles, labels = ax1.get_legend_handles_labels()
# Remove the first instance of each type
# put the legend at the top
handles, labels = handles[:4], labels[:4]

fig.legend(
    handles=handles,
    labels=labels,
    ncols=2,
    bbox_to_anchor=(0.1, 1.3),
    loc=2,
    columnspacing=1.5,
    handletextpad=0.5,
)
sns.despine()
plt.savefig(".local/kernel_time_breakdown.pdf", bbox_inches="tight", dpi=300)
