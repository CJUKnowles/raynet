#!/usr/bin/env python

# Generates a single csv file for given experiment name
# generateSingleCsvFile experimentName protocolName runNumber
# 

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path
import os
import subprocess
import re
import time as termTime
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import re
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator

protocol_colors = {
    "Cubic": "#ff7f0e",
    "CleanSlate": "#2ca02c",
    "Orca": "#1f77b4",
    "Astrea": "#8736a4",
}

protocol_markers = {
    "Cubic": "o",
    "CleanSlate": "s",
    "Orca": "P",
    "Astrea": "^",
}


def parse_numeric(value, as_int=False):
    match = re.search(r"[-+]?\d*\.?\d+", str(value))
    if as_int:
        return int(match.group()) if match else None
    else:
        return float(match.group()) if match else None

def is_ax_empty(ax):
    return not (ax.has_data() or ax.patches or ax.lines or ax.collections)

def create_csv_dict(results_dir:str=None):
    """
    Returns a dataframe of csv files, in the format [experiment, protocol, module, metric, csv_path].
    This dataframe provides easy navigation of all metric CSVs, which is useful for create combined plots.
    """
    if not results_dir:
        results_dir = os.getenv('HOME') + "/raynet/results"
    
    csvs = []
    for root, dirs, files in os.walk(results_dir):
        for filename in files:
            if filename.endswith(".csv"):
                metric = os.path.splitext(filename)[0]
                
                dir_names = root.split(os.sep)          # Split by "/" or "\\" depending on platform
                module = dir_names[-1].split('.', 1)[1] if '.' in dir_names[-1] else dir_names[-1]   # Module name, excluding the network name prefix
                module_type = "client" if "client" in module else "server" if "server" in module else "queue" if "queue" in module else "scenario" if "scenario" in module else "other"
                # csvs folder is'[-2]
                protocol = dir_names[-3]
                run = parse_numeric(dir_names[-4], as_int=True)
                params = dir_names[-5]
                experiment = dir_names[-6]
                
                csv_path = os.path.join(root, filename)
                csv_info = {
                    "experiment" : experiment,
                    "params": params,
                    "protocol": protocol,
                    "run": run,
                    "module": module,
                    "module_type": module_type,
                    "metric": metric,
                    "csv_path": csv_path
                            }

                
                # Add extra columns for each param (note to self - will likely be different per experiment, so these columns wil be sparsely populated)
                for param_str in params.split("_")[1:]:
                    param = param_str.split("-")
                    if len(param) < 2:
                        # Param string is not formatted correctly, skip
                        break
                    param_value = param[0]
                    param_name = param[1]
                    csv_info[param_name] = param_value
                
                csvs.append(csv_info)
    return pd.DataFrame(csvs)

def plot_pacerate_timeseries(csv_df, ax=None, show_competition=True):
    """
    Overlays results from several experiments to create a single pacerate plot for comparison.
    - Expects a single experiment's dataframe as input
    - Only plots primary flows (module number is 0, module type is server)
    """

    csv_df = csv_df[csv_df["module_type"].str.contains("client")]
    csv_df = csv_df[csv_df["module"].str.contains("conn")]
    csv_df = csv_df[csv_df["metric"].str.contains("paceRate")]

    if csv_df.empty:
        print("plot_paceRate_timeseries(): CSV dataframe is empty. Returning.")
        return None

    print("Plotting paceRate timeseries for:")
    print(csv_df)

    all_y_values = []
    for _, row in csv_df.iterrows():
        data = pd.read_csv(row["csv_path"])
        
    window = 50
    all_y_values = []
    for _, row in csv_df.iterrows():
        data = pd.read_csv(row["csv_path"])
        is_primary_flow = "0" in row["module"]
        if(show_competition or is_primary_flow):
            x = data["time"]
            y = data["paceRate"]
            
            # Rolling stats
            rolling_mean = y.rolling(window, center=True).mean()
            rolling_std = y.rolling(window, center=True).std()

            # Raw data
            ax.plot(x, y, alpha=0.2, linewidth=0.8, color=protocol_colors[row["protocol"]])

            # Smoothed mean
            ax.plot(x, rolling_mean, linewidth=2, label=row["protocol"], color=protocol_colors[row["protocol"]], linestyle='-' if is_primary_flow else '--')
            all_y_values.extend(data["paceRate"].values)

    
    ax.set_xlabel("Time")
    ax.set_ylabel("paceRate (Mbps)")
    ax.set_title("paceRate over time")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0, top=np.percentile(all_y_values, 99)*1.1)
    ax.set_yscale("linear")
    ax.ticklabel_format(style='plain', axis='y')
    ax.grid(True)

    return ax


def plot_qsize_timeseries(csv_df, ax=None):
    """
    Overlays results from several experiments to create a single qsize plot for comparison.
    - Expects a single experiment's dataframe as input
    - Only plots primary flows (module number is 0, module type is server)
    """

    csv_df = csv_df[csv_df["module"].str.contains("router1")]
    csv_df = csv_df[csv_df["module"].str.contains("0")]
    csv_df = csv_df[csv_df["metric"].str.contains("queueBitLength")]

    if csv_df.empty:
        print("plot_qsize_timeseries(): CSV dataframe is empty. Returning.")
        return None

    print("Plotting qsize timeseries for:")
    print(csv_df)

    window = 100
    all_y_values = []
    for _, row in csv_df.iterrows():
        data = pd.read_csv(row["csv_path"])
        x = data["time"]    
        y = data["queueBitLength"]
        
        # Rolling stats
        rolling_mean = y.rolling(window, center=True).mean()
        rolling_std = y.rolling(window, center=True).std()

        # Raw data
        line, = ax.plot(x, y, alpha=0.2, linewidth=0.8, color=protocol_colors[row["protocol"]])

        # Smoothed mean
        ax.plot(x, rolling_mean, linewidth=2, label=row["protocol"], color=protocol_colors[row["protocol"]])

    ax.set_xlabel("Time")
    ax.set_ylabel("Queue Size (bits)")
    ax.set_title("Queue Size Over Time")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    ax.set_yscale("linear")
    ax.ticklabel_format(style='plain', axis='y')
    ax.grid(True)

    return ax
        
def plot_cwnd_timeseries(csv_df, ax=None, show_competition=True):
    """
    Overlays results from several experiments to create a single cwnd plot for comparison.
    - Expects a single experiment's dataframe as input
    - Only plots primary flows (module number is 0, module type is server)
    """

    csv_df = csv_df[csv_df["module_type"].str.contains("client")]
    csv_df = csv_df[csv_df["module"].str.contains("conn")]
    csv_df = csv_df[csv_df["metric"].str.contains("cwnd")]

    if csv_df.empty:
        print("plot_cwnd_timeseries(): CSV dataframe is empty. Returning.")
        return None

    print("Plotting cwnd timeseries for:")
    print(csv_df)

    all_y_values = []
    for _, row in csv_df.iterrows():
        data = pd.read_csv(row["csv_path"])
    
    window = 50
    all_y_values = []
    for _, row in csv_df.iterrows():
        data = pd.read_csv(row["csv_path"])
        is_primary_flow = "0" in row["module"]
        if(show_competition or is_primary_flow):
            x = data["time"]
            y = data["cwnd"]
            
            # Rolling stats
            rolling_mean = y.rolling(window, center=True).mean()
            rolling_std = y.rolling(window, center=True).std()

            # Raw data
            line, = ax.plot(x, y, alpha=0.2, linewidth=0.8, color=protocol_colors[row["protocol"]])

            # Smoothed mean
            ax.plot(x, rolling_mean, linewidth=2, label=row["protocol"] if is_primary_flow else None, color=protocol_colors[row["protocol"]], linestyle='-' if is_primary_flow else '--')

    ax.set_xlabel("Time")
    ax.set_ylabel("cwnd (bytes)")
    ax.set_title("cwnd over time")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    ax.set_yscale("linear")
    ax.ticklabel_format(style='plain', axis='y')
    ax.grid(True)

    return ax

def plot_srtt_timeseries(csv_df, ax=None, show_competition=True):
    """
    Overlays results from several experiments to create a single srtt plot for comparison.
    - Expects a single experiment's dataframe as input
    - Only plots primary flows (module number is 0, module type is server)
    """
    scenario_df = csv_df[csv_df["module_type"].str.contains("scenario")]
    csv_df = csv_df[csv_df["module_type"].str.contains("client")]
    csv_df = csv_df[csv_df["module"].str.contains("conn")]
    csv_df = csv_df[csv_df["metric"].str.contains("srtt")]

    if csv_df.empty:
        print("plot_srtt_timeseries(): CSV dataframe is empty. Returning.")
        return None

    print("Plotting srtt timeseries for:")
    print(csv_df)

    all_y_values = []
    for _, row in csv_df.iterrows():
        data = pd.read_csv(row["csv_path"])
        is_primary_flow = "0" in row["module"]
        if(show_competition or is_primary_flow):
            ax.plot(
                data["time"],
                data["srtt"],
                label=row["protocol"], 
                color=protocol_colors[row["protocol"]],
                linestyle='-' if is_primary_flow else '--'
            )
            all_y_values.extend(data["srtt"].values)

    # Plot the base RTT as optimal if there is a scenario file that alters it
    scenario_delay_df = scenario_df[scenario_df["metric"] == "delay"]
    if not scenario_delay_df.empty:
        scenario_delay_df = scenario_delay_df.iloc[0]
        scenario_delay_data = pd.read_csv(scenario_delay_df["csv_path"])
        scenario_delay_data["delay"] *= 2 * .001 # one-way to two-way delay (RTT)
        print("-")
        print(scenario_delay_data)
        ax.plot(
                scenario_delay_data["time"],
                scenario_delay_data["delay"],
                linestyle='--',
                linewidth=1,
                color='black',
                alpha=1,
                label="Optimal",
                drawstyle="steps-post",
                zorder=10
            )
        
    ax.set_xlabel("Time")
    ax.set_ylabel("sRTT (ms)")
    ax.set_title("sRTT over time")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    ax.set_yscale("linear")
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda y, _: f"{y*1000:g}") # Display y-axis in ms
    )
    ax.grid(True)

    return ax

def plot_throughput_timeseries(csv_df, ax=None, show_competition=True):
    """
    Overlays results from several experiments to create a single throughput plot for comparison.
    - Expects a single experiment's dataframe as input
    - Only plots primary flows (module number is 0, module type is server)
    """
    scenario_df = csv_df[csv_df["module_type"].str.contains("scenario")]
    csv_df = csv_df[csv_df["module_type"].str.contains("server")]
    csv_df = csv_df[csv_df["module"].str.contains("conn")]
    csv_df = csv_df[csv_df["metric"].str.contains("throughput")]

    if csv_df.empty:
        print("plot_throughput_timeseries(): CSV dataframe is empty. Returning.")
        return None

    print("Plotting throughput timeseries for:")
    print(csv_df)
    
    colors = {}
    all_y_values = []
    for _, row in csv_df.iterrows():
        data = pd.read_csv(row["csv_path"])
        is_primary_flow = "0" in row["module"]
        if(show_competition or is_primary_flow):
            line = ax.plot(
                data["time"],
                data["throughput"],
                label=row["protocol"],
                color=protocol_colors[row["protocol"]],
                linestyle='-' if is_primary_flow else '--'
            )
            all_y_values.extend(data["throughput"].values)
            
    # Plot optimal throughputs only if there is a scenario file containing that info (mostly for responsiveness experiments)
    scenario_throughput_df = scenario_df[scenario_df["metric"] == "datarate"]
    if not scenario_throughput_df.empty:
        scenario_throughput_df = scenario_throughput_df.iloc[0]
        scenario_throughput_data = pd.read_csv(scenario_throughput_df["csv_path"])
        scenario_throughput_data["datarate"] *= 125000 * 8 #mbps
        print("-")
        print(scenario_throughput_data)
        ax.plot(
                scenario_throughput_data["time"],
                scenario_throughput_data["datarate"],
                linestyle='--',
                linewidth=1,
                color='black',
                alpha=1,
                label="Optimal",
                drawstyle="steps-post",
                zorder=10
            )

    ax.set_xlabel("Time")
    ax.set_ylabel("Throughput (Mbps)")
    ax.set_title("Throughput over time")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0, top=np.percentile(all_y_values, 99)*1.1)
    ax.set_yscale("linear")
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda y, _: f"{y/1e6:g}") # makes the y-axis Mbps insead of Bps
    )
    ax.grid(True)

    return ax

def plot_tcp_friendliness(csv_df, ax=None, show_competition=False, startup_time=30):
    csv_df = csv_df[csv_df["module_type"].str.contains("server")]
    csv_df = csv_df[csv_df["module"].str.contains("conn")]
    csv_df = csv_df[csv_df["metric"].str.contains("throughput")]

    if csv_df.empty:
        print("plot_goodput_ratio_aggregate(): CSV dataframe is empty. Returning.")
        return None

    qsizes = sorted(csv_df["QSIZE"].unique())
    print("qsizes")
    print(qsizes)
    if ax is None:
        fig, axes = plt.subplots(1, len(qsizes), figsize=(6 * len(qsizes), 5), layout="constrained", sharex=True)
        if len(qsizes) == 1:
            axes = [axes]
    else:
        axes = ax if isinstance(ax, (list, np.ndarray)) else [ax]

    empty_axes = [a for a in axes if is_ax_empty(a)]
    axes_to_use = empty_axes[:len(qsizes)]

    if len(axes_to_use) < len(qsizes):
        raise ValueError("Not enough empty subplots to draw all queue sizes")

    for ax_i, qsize in zip(axes_to_use, qsizes):
        df_q = csv_df[csv_df["QSIZE"] == qsize]

        delays = sorted(df_q["DELAY"].unique(), key=parse_numeric)
        protocols = df_q["protocol"].unique()

        bar_width = 0.8 / len(protocols)

        all_y_values = []

        for i, protocol in enumerate(protocols):
            proto_df = df_q[df_q["protocol"] == protocol]
            y_vals = []
            y_err_lower = []
            y_err_upper = []

            for delay in delays:
                row = proto_df[proto_df["DELAY"] == delay]
                if row.empty:
                    y_vals.append(0)
                    y_err_lower.append(0)
                    y_err_upper.append(0)
                    continue

                ratios = []
                for run in row["run"].unique():
                    main_flow = row[row["module"].str.contains("0")]
                    main_flow = main_flow[main_flow["run"] == run].iloc[0]
                    main_flow_data = pd.read_csv(main_flow["csv_path"])
                    main_flow_data = main_flow_data[main_flow_data["time"] > startup_time]

                    competing_flow = row[row["module"].str.contains("1")]
                    competing_flow = competing_flow[competing_flow["run"] == run].iloc[0]
                    competing_flow_data = pd.read_csv(competing_flow["csv_path"])
                    competing_flow_data = competing_flow_data[competing_flow_data["time"] > startup_time]
                    
                    ratio = main_flow_data["throughput"].mean() / competing_flow_data["throughput"].mean()
                    ratios.append(ratio)

                mean = np.mean(ratios)
                min_val = np.min(ratios)
                max_val = np.max(ratios)

                y_vals.append(mean)
                y_err_lower.append(mean - min_val)
                y_err_upper.append(max_val - mean)

            ax_i.errorbar(
                delays,
                y_vals,
                yerr=[y_err_lower, y_err_upper],
                label=protocol,
                color=protocol_colors[protocol],
                marker=protocol_markers[protocol],
                linestyle='-',
                linewidth=1.5,
                markersize=8,
                capsize=4,
            )
        
        ax_i.axhline(
            y=1.0,
            linestyle='--',
            linewidth=1,
            color='black',
            alpha=.5,
            label="Optimal"
        )
        
        ax_i.set_xlabel("Base RTT (ms)")
        # ax_i.set_xticks(x_positions + bar_width * (len(protocols) - 1) / 2)
        ax_i.tick_params(direction="inout")
        # ax_i.set_xticklabels(map(lambda value: parse_numeric(value, as_int=True), delays))
        ax_i.set_title(f"Buffer Size: {parse_numeric(qsize)}x BDP")

        if ax_i is axes_to_use[0]:
            ax_i.set_ylabel("Goodput Ratio")
        ax_i.set_yscale("log")
        ax_i.set_ylim(0.01, 100)
        ax_i.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
        ax_i.set_yticks([.01, .1, 1, 10, 100])
            
    if ax is None:
        fig.suptitle("TCP Friendliness; Goodput ratio against competing cubic flow", fontsize=20, y=0.02, verticalalignment="bottom")
        legend_handles, legend_labels = axes_to_use[0].get_legend_handles_labels()
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=len(legend_labels),
            frameon=False, 
            bbox_to_anchor=(.5, 1),
            fontsize=12,
        )
        fig.set_constrained_layout_pads(
            w_pad=0.2,
            h_pad=0.6,
        )
        return (fig, axes_to_use)
    else:
        axes_to_use[0].legend(fontsize=8)
        return axes_to_use
    return axes_to_use


def plot_goodput_ratio_aggregate(csv_df, ax=None, show_competition=False):
    csv_df = csv_df[csv_df["module_type"].str.contains("server")]
    csv_df = csv_df[csv_df["module"].str.contains("conn")]
    csv_df = csv_df[csv_df["module"].str.contains("0")]
    csv_df = csv_df[csv_df["metric"].str.contains("throughput")]

    if csv_df.empty:
        print("plot_goodput_ratio_aggregate(): CSV dataframe is empty. Returning.")
        return None

    qsizes = sorted(csv_df["QSIZE"].unique())

    if ax is None:
        fig, axes = plt.subplots(1, len(qsizes), figsize=(6 * len(qsizes), 5), sharey=True)
        if len(qsizes) == 1:
            axes = [axes]
    else:
        axes = ax if isinstance(ax, (list, np.ndarray)) else [ax]

    empty_axes = [a for a in axes if is_ax_empty(a)]
    axes_to_use = empty_axes[:len(qsizes)]

    if len(axes_to_use) < len(qsizes):
        raise ValueError("Not enough empty subplots to draw all queue sizes")

    for ax_i, qsize in zip(axes_to_use, qsizes):
        df_q = csv_df[csv_df["QSIZE"] == qsize]

        delays = sorted(df_q["DELAY"].unique())
        protocols = df_q["protocol"].unique()

        bar_width = 0.8 / len(protocols)
        x_positions = np.arange(len(delays))

        all_y_values = []

        for i, protocol in enumerate(protocols):
            proto_df = df_q[df_q["protocol"] == protocol]

            y_vals = []
            for delay in delays:
                row = proto_df[proto_df["DELAY"] == delay]
                if row.empty:
                    y_vals.append(0)
                    continue
                row = row.iloc[0]
                data = pd.read_csv(row["csv_path"])

                ratio = data["throughput"].mean() / parse_numeric(row["BANDWIDTH"])
                y_vals.append(ratio)
                all_y_values.append(ratio)

            ax_i.bar(
                x_positions + i * bar_width,
                y_vals,
                width=bar_width,
                label=protocol,
                color=protocol_colors[protocol],
            )

        ax_i.set_xticks(x_positions + bar_width * (len(protocols) - 1) / 2)
        ax_i.set_xticklabels(delays)

        ax_i.set_title(f"QSIZE = {qsize}")
        ax_i.set_xlabel("Delay")
        ax_i.grid(True)

        if ax_i is axes_to_use[0]:
            ax_i.set_ylabel("Goodput Ratio")

    axes_to_use[0].legend(fontsize=8)
    
    return axes_to_use

if __name__ == "__main__":
    # experimentNames = ["double-flow-dumbbell", "single-flow", "responsiveness"]
    # protocols = ["Cubic", "Orca"]
    # for experimentName in experimentNames:
    #     for protocol in protocols:
    #         # Generate output.csv and individual vector csvs for all tracked vectors of agiven experiment
    #         exp_results_dir = os.getenv('HOME') + f"/raynet/_experiments/{experimentName}/ini_variants/results"
    #         generate_exp_csvs(exp_results_dir, protocol, do_dumb_plots=True)
    
    
    metric_csvs = create_csv_dict()        # dataframe containing [experiment, params, protocol, module, metric, csv_path] for easy access
    experiments = ["competing-flows"]
    for exp in experiments:
        exp_df = metric_csvs[metric_csvs["experiment"] == exp]
        
        # Special aggregate plots unique to each experiment
        if exp == "competing-flows":
            print("Plotting completing flows aggregate plots")
            fig, axes = plot_tcp_friendliness(exp_df)
            fig.savefig(os.getenv('HOME') + f"/raynet/results/{exp}/aggregate_plots.pdf")
            plt.close(fig)
        elif exp == "responsiveness":
            # TODO: Find a way to plot overall performance of responsiveness experiments. Compare each datapoint against its theoretical max? cant just take a mean.
            print("responot implemented")
        else:
            print("Plotting single flow aggregate plots")
            fig, axs = plt.subplots(20, 1, figsize=(15, 100))
            plot_goodput_ratio_aggregate(exp_df, axs)
            fig.tight_layout()
            fig.savefig(os.getenv('HOME') + f"/raynet/results/{exp}/aggregate_plots.pdf")
            plt.close(fig)
        

        # Summary Timeseries plots for all experiments
        for params in exp_df["params"].unique():
            params_df = exp_df[exp_df["params"] == params]
            for run in params_df["run"].unique():
                run_df = params_df[params_df["run"] == run]
                fig, axs = plt.subplots(20, 1, figsize=(15, 100))
                
                
                plot_throughput_timeseries(run_df, axs[0])
                plot_srtt_timeseries(run_df, axs[1])
                plot_cwnd_timeseries(run_df, axs[2])
                plot_pacerate_timeseries(run_df, axs[3])
                plot_qsize_timeseries(run_df, axs[4])
                
                fig.tight_layout()
                fig.savefig(os.getenv('HOME') + f"/raynet/results/{exp}/{params}/run{int(run)}/summary.pdf")
                plt.close(fig)
                    
            
    # metric_csvs = metric_csvs[metric_csvs["module"].str.contains("0")]          # Only grab data from primary flows
    # #metric_csvs = metric_csvs[(metric_csvs["module_type"] == "client")]       # Only grab data from clients
    
    # # Make Plots
    # experiments = ["single-flow", "double-flow-dumbbell", "responsiveness"]
    # metrics = ["throughput", "srtt", "pacerate","paceRate", "intervalDuration", "cwnd", "action", "incomingDataRate", "outgoingDataRate", "queueBitLength"]
    # module_types = ["client", "server", "queue"]
    # for experiment in experiments:
    #     exp_df = metric_csvs[metric_csvs["experiment"] == experiment]
    #     for module_type in module_types:
    #         module_type_df = exp_df[exp_df["module_type"] == module_type]
    #         for metric in metrics:
    #             metric_df = module_type_df[module_type_df["metric"] == metric]
    #             if metric_df.empty:
    #                 print("Dataframe is emtpy, continuing")
    #                 continue
    #             print(f"Plotting {metric} using: ")
    #             print(tabulate(metric_df, headers='keys'))
                
    #             plt.figure(figsize=(15,6))
    #             for _, row in metric_df.iterrows():
    #                 csv_data = pd.read_csv(row["csv_path"])
    #                 plt.plot(csv_data["time"], csv_data[metric], label=f'{row["protocol"]}: {row["module"]}')
                
    #             plt.xlabel("Time")
    #             plt.ylabel(metric)
    #             plt.title(f"{metric} over time")
    #             plt.legend(fontsize=8)  # smaller font if many lines
    #             plt.ylim(bottom=0)
    #             plt.yscale("linear")
    #             plt.ticklabel_format(style='plain', axis='y')
    #             plt.grid(True)
    #             plt.tight_layout()
    #             plt.savefig(os.getenv('HOME') + f"/raynet/results/{experiment}/summary/{metric}-{module_type}.pdf")
    #             plt.close()