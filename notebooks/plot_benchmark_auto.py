import os
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

local_path = ".local/run_4"
df = []

files = [x for x in os.listdir(local_path) if x.endswith(".json")]
for file in files:
    with open(f"{local_path}/{file}", "r") as fp:
        data = json.load(fp)
        file_id = file.split(".json")[0]
        if "ibmm_fp16_for" in file:
            approach = "FP16"
            kernel_names = ['ampere_fp16', 'cutlass::Kernel']
        elif "for-loop" in file:
            approach = "For Loop"
            kernel_names = ['marlin::Marlin_2_4']
        elif "ibmm_native" in file:
            approach = "Native"
            kernel_names = ['marlin::IBMM_2_4']
        elif "ibmm_fp16_bmm" in file:
            approach = "FP16 BMM"
            kernel_names = ['cutlass::Kernel']
        else:
            raise ValueError(f"Unknown approach: {file}")
        size = file_id.split("_")[0]
        trial = file_id.split("_")[-2]
        num_models = file_id.split("_")[-1]
        total_time = sum([x["TAvg (ns)"] * x["Count"] for x in data])
        kernel_time = sum(
            [x["KAvg (ns)"] * x["Count"] for x in data if any([y in x["Kernel Name"] for y in kernel_names])]
        )
        df.append(
                {
                    "size": size,
                    "num_model": num_models,
                    "approach": approach,
                    "total_time": total_time / 1e6,
                    "kernel_time": kernel_time / 1e6,
                }
            )
df = pd.DataFrame(df)
df.to_csv(".local/benchmark_marlin_nsys.csv", index=False)
print(df)
fig, (ax1, ax2) = plt.subplots(
    ncols=2, nrows=1, constrained_layout=True, figsize=(9, 3.75), sharey=True
)
width = 0.15
positions = np.arange(2)

positions_fp16_bmm = [x - 2*width for x in positions]
positions_fp16_forloop = [x-width for x in positions]
positions_naive = [x for x in positions]
positions_sbmm = [x+width for x in positions]

small_df = df[df["size"] == "2k"]
large_df = df[df["size"] == "4k"]
small_groups = small_df.groupby("num_model")
large_groups = large_df.groupby("num_model")
print(large_df)
for pos_fp16, pos_fp16bmm, pos_naive, pos_sbmm, (num_model, group) in zip(
    positions_fp16_forloop, positions_fp16_bmm, positions_naive, positions_sbmm, small_groups
):
    naive_group = group[group["approach"] == "For Loop"]
    sbmm_group = group[group["approach"] == "Native"]
    fp16_for_group = group[group["approach"] == "FP16"]
    fp16_bmm_group = group[group["approach"] == "FP16 BMM"]
    # Plot for 'Naive'
    p1 = ax1.bar(
        pos_fp16,
        fp16_for_group["kernel_time"].mean(),
        width,
        label="FP16 for-loop",
        color=cmp[0],
        yerr=fp16_for_group["kernel_time"].std(),
        linewidth=1,
        edgecolor="k",
    )
    p2 = ax1.bar(
        pos_fp16,
        (fp16_for_group["total_time"] - fp16_for_group["kernel_time"]).mean(),
        width,
        bottom=fp16_for_group["kernel_time"].mean(),
        color=cmp[0],
        alpha=0.5,
        label="FP16 for-loop",
        yerr=(fp16_for_group["total_time"] - fp16_for_group["kernel_time"]).std(),
        linewidth=1,
        edgecolor="k",
    )
    p3 = ax1.bar(
        pos_fp16bmm,
        fp16_bmm_group["kernel_time"].mean(),
        width,
        label="FP16 bmm",
        color=cmp[1],
        yerr=fp16_bmm_group["kernel_time"].std(),
        linewidth=1,
        edgecolor="k",
    )
    p4 = ax1.bar(
        pos_fp16bmm,
        (fp16_bmm_group["total_time"] - fp16_bmm_group["kernel_time"]).mean(),
        width,
        bottom=fp16_bmm_group["kernel_time"].mean(),
        color=cmp[1],
        alpha=0.5,
        label="FP16 bmm",
        yerr=(fp16_bmm_group["total_time"] - fp16_bmm_group["kernel_time"]).std(),
        linewidth=1,
        edgecolor="k",
    )
    p5 = ax1.bar(
        pos_naive,
        naive_group["kernel_time"].mean(),
        width,
        label="Naive for-loop",
        color=cmp[2],
        yerr=naive_group["kernel_time"].std(),
        linewidth=1,
        edgecolor="k",
    )
    p6 = ax1.bar(
        pos_naive,
        (naive_group["total_time"] - naive_group["kernel_time"]).mean(),
        width,
        bottom=naive_group["kernel_time"].mean(),
        color=cmp[2],
        alpha=0.5,
        label="Naive for-loop",
        yerr=(naive_group["total_time"] - naive_group["kernel_time"]).std(),
        linewidth=1,
        edgecolor="k",
    )
    # Plot for 'SBMM'
    p7 = ax1.bar(
        pos_sbmm,
        sbmm_group["kernel_time"].mean(),
        width,
        label="SBMM",
        color=cmp[4],
        yerr=sbmm_group["kernel_time"].std(),
        linewidth=1,
        edgecolor="k",
    )
    p8= ax1.bar(
        pos_sbmm,
        (sbmm_group["total_time"] - sbmm_group["kernel_time"]).mean(),
        width,
        bottom=sbmm_group["kernel_time"].mean(),
        color=cmp[4],
        alpha=0.5,
        label="SBMM",
        yerr=(sbmm_group["total_time"] - sbmm_group["kernel_time"]).std(),
        linewidth=1,
        edgecolor="k",
    )
for pos_fp16bmm, pos_fp16, pos_naive, pos_sbmm, (num_model, group) in zip(
    positions_fp16_bmm, positions_fp16_forloop, positions_naive, positions_sbmm, large_groups
):
    naive_group = group[group["approach"] == "For Loop"]
    sbmm_group = group[group["approach"] == "Native"]
    fp16_for_group = group[group["approach"] == "FP16"]
    fp16_bmm_group = group[group["approach"] == "FP16 BMM"]
    p1 = ax2.bar(
        pos_fp16,
        fp16_for_group["kernel_time"].mean(),
        width,
        label="Kernel Time (Naive)",
        color=cmp[0],
        yerr=fp16_for_group["kernel_time"].std(),
        linewidth=1,
        edgecolor="k",
    )
    p2 = ax2.bar(
        pos_fp16,
        (fp16_for_group["total_time"] - fp16_for_group["kernel_time"]).mean(),
        width,
        bottom=fp16_for_group["kernel_time"].mean(),
        color=cmp[0],
        alpha=0.5,
        label="Total Time (Naive)",
        yerr=(fp16_for_group["total_time"] - fp16_for_group["kernel_time"]).std(),
        linewidth=1,
        edgecolor="k",
    )
    p3 = ax2.bar(
        pos_fp16bmm,
        fp16_bmm_group["kernel_time"].mean(),
        width,
        label="Kernel Time (Naive)",
        color=cmp[1],
        yerr=fp16_bmm_group["kernel_time"].std(),
        linewidth=1,
        edgecolor="k",
    )
    p4 = ax2.bar(
        pos_fp16bmm,
        (fp16_bmm_group["total_time"] - fp16_bmm_group["kernel_time"]).mean(),
        width,
        bottom=fp16_bmm_group["kernel_time"].mean(),
        color=cmp[1],
        alpha=0.5,
        label="Total Time (Naive)",
        yerr=(fp16_bmm_group["total_time"] - fp16_bmm_group["kernel_time"]).std(),
        linewidth=1,
        edgecolor="k",
    )
    # Plot for 'Naive'
    # error bar, use the standard deviation
    p5=ax2.bar(
        pos_naive,
        naive_group["kernel_time"].mean(),
        width,
        label="Kernel Time (Naive)",
        color=cmp[2],
        yerr=naive_group["kernel_time"].std(),
        linewidth=1,
        edgecolor="k",
    )
    p6=ax2.bar(
        pos_naive,
        (naive_group["total_time"] - naive_group["kernel_time"]).mean(),
        width,
        bottom=naive_group["kernel_time"].mean(),
        color=cmp[2],
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
        color=cmp[4],
        yerr=sbmm_group["kernel_time"].std(),
        linewidth=1,
        edgecolor="k",
    )
    p8=ax2.bar(
        pos_sbmm,
        (sbmm_group["total_time"] - sbmm_group["kernel_time"]).mean(),
        width,
        bottom=sbmm_group["kernel_time"].mean(),
        color=cmp[4],
        alpha=0.5,
        label="Total Time (SBMM)",
        yerr=(sbmm_group["total_time"] - sbmm_group["kernel_time"]).std(),
        linewidth=1,
        edgecolor="k",
    )

group_center_positions = [(n + s) / 2-width for n, s in zip(positions_naive, positions_sbmm)]
print(group_center_positions)
ax1.set_xticks(group_center_positions)
ax1.set_xticklabels([f"{num_model} Models" for num_model in small_groups.groups.keys()])
# set ylimit
ax1.set_ylim(0, 15)
ax2.set_ylim(0, 15)
group_center_positions = [(n + s) / 2-width for n, s in zip(positions_naive, positions_sbmm)]
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
print(handles)
handles, labels = handles[::2][:4], labels[::2][:4]

fig.legend(
    handles=handles,
    labels=labels,
    ncols=4,
    bbox_to_anchor=(0, 1.145),
    loc=2,
    columnspacing=1.5,
    handletextpad=0.5,
)
sns.despine()
plt.savefig(".local/kernel_time_breakdown_full.pdf", bbox_inches="tight", dpi=300)
