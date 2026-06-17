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
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

protocol_colors = {
    "Cubic": "#ff7f0e",
    "CleanSlate": "#2ca02c",
    "Orca": "#1f77b4",
    "Astraea": "#8736a4",
}

protocol_markers = {
    "Cubic": "o",
    "CleanSlate": "s",
    "Orca": "P",
    "Astraea": "^",
}

FLOW_COLOR_MODE_SHARED = "shared-primary"
FLOW_COLOR_MODE_UNIQUE = "unique-flow"
FLOW_COLOR_MODES = {FLOW_COLOR_MODE_SHARED, FLOW_COLOR_MODE_UNIQUE}
FLOW_COLOR_CYCLE = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors) + list(plt.get_cmap("tab20c").colors)

def get_flow_id(module):
    matches = re.findall(r"\[(\d+)\]", str(module))
    if matches:
        return matches[0]
    match = re.search(r"(\d+)", str(module))
    return match.group(1) if match else "0"

def is_primary_flow(module):
    return get_flow_id(module) == "0"

def get_flow_key(row):
    return f"{row['protocol']}:{get_flow_id(row['module'])}"

def build_flow_color_map(csv_df):
    flow_df = csv_df[csv_df["module_type"].str.contains("client|server")]
    flow_df = flow_df[flow_df["module"].str.contains("conn|app")]
    flow_keys = sorted({get_flow_key(row) for _, row in flow_df.iterrows()})
    return {
        flow_key: FLOW_COLOR_CYCLE[i % len(FLOW_COLOR_CYCLE)]
        for i, flow_key in enumerate(flow_keys)
    }

def get_timeseries_style(row, flow_color_mode=FLOW_COLOR_MODE_SHARED, flow_color_map=None):
    if flow_color_mode not in FLOW_COLOR_MODES:
        raise ValueError(f"Unknown flow_color_mode '{flow_color_mode}'. Expected one of {sorted(FLOW_COLOR_MODES)}")

    primary = is_primary_flow(row["module"])
    if flow_color_mode == FLOW_COLOR_MODE_UNIQUE:
        flow_color_map = flow_color_map or {}
        color = flow_color_map.get(get_flow_key(row), protocol_colors[row["protocol"]])
        label = f"{row['protocol']} flow {get_flow_id(row['module'])}"
        linestyle = "-"
        alpha = 1.0
    else:
        color = protocol_colors[row["protocol"]]
        label = row["protocol"] if primary else None
        linestyle = "-" if primary else (0, (1, .5))
        alpha = 1.0 if primary else 0.7

    return {
        "color": color,
        "label": label,
        "linestyle": linestyle,
        "alpha": alpha,
        "is_primary": primary,
    }

def percentile_clip(series, lower=1, upper=99):
    """
    Clips extreme values outside the given percentiles.
    
    lower, upper: percentile thresholds (0-100)
    """
    low = np.percentile(series, lower)
    high = np.percentile(series, upper)

    return series.clip(lower=low, upper=high)

def hampel_filter(series, window_size=7, n_sigmas=3):
    """
    Hampel filter for outlier removal.
    Replaces outliers with the rolling median.
    """
    s = series.copy()

    rolling_median = s.rolling(window=window_size, center=True).median()
    mad = (s - rolling_median).abs().rolling(window=window_size, center=True).median()

    threshold = n_sigmas * 1.4826 * mad
    diff = (s - rolling_median).abs()

    outliers = diff > threshold
    s[outliers] = rolling_median[outliers]

    return s

def remove_single_point_spikes(series, threshold):
    s = series.copy()

    for i in range(1, len(s)-1):
        if (
            abs(s[i] - s[i-1]) > threshold and
            abs(s[i] - s[i+1]) > threshold and
            abs(s[i-1] - s[i+1]) < threshold
        ):
            s[i] = (s[i-1] + s[i+1]) / 2

    return pd.Series(s, index=series.index)

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
        results_dir = os.getenv('RAYNET_PATH') + "/_results"
    
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

def time_weighted_mean(df, startup_time=0, end_time=None):
    """
    Takes a dataframe with two columns (times, values) as input, at returns a time-weighted mean of the values.
    - This assumes that, at each timestep, the most recent value is canon. There isn't any interpolation.
    - This is intended for use with scenario data from responsiveness experiments.
    """
    times = df.iloc[:, 0].values
    values = df.iloc[:, 1].values
    if not end_time:
        end_time = times[-1]

    total_weighted = 0.0
    total_time = 0.0

    for i in range(len(times)):
        t_start = times[i]
        t_end = times[i+1] if i+1 < len(times) else end_time

        # Clip interval to [startup_time, end_time]
        interval_start = max(t_start, startup_time)
        interval_end = min(t_end, end_time)

        if interval_end <= interval_start:
            continue

        duration = interval_end - interval_start
        total_weighted += values[i] * duration
        total_time += duration

    if total_time == 0:
        return np.nan

    return total_weighted / total_time

def plot_pacerate_timeseries(csv_df, ax=None, show_competition=True, startup_time=0, end_time=None, flow_color_mode=FLOW_COLOR_MODE_SHARED, flow_color_map=None):
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
        
    window = 50
    all_y_values = []
    for _, row in csv_df.iterrows():
        data = pd.read_csv(row["csv_path"])
        data = data[data["time"] > startup_time]
        if end_time:
            data = data[data["time"] < end_time]
        style = get_timeseries_style(row, flow_color_mode, flow_color_map)
        if(show_competition or style["is_primary"]):
            x = data["time"]
            y = data["paceRate"]
            
            # Rolling stats
            rolling_mean = y.rolling(window, center=True).mean()
            rolling_std = y.rolling(window, center=True).std()

            # Raw data
            ax.plot(x, y, alpha=0.2, linewidth=0.8, color=style["color"])

            # Smoothed mean
            ax.plot(x, rolling_mean, linewidth=2, label=style["label"], color=style["color"], linestyle=style["linestyle"], alpha=style["alpha"])
            all_y_values.extend(data["paceRate"].values)

    ax.set_ylabel("paceRate (Mbps)")
    ax.set_ylim(bottom=0, top=np.percentile(all_y_values, 99)*1.1)
    ax.set_xlim(left=0)
    ax.set_yscale("linear")
    if end_time:
        ax.set_xlim(right=end_time)
    ax.ticklabel_format(style='plain', axis='y')

    return ax


def plot_qsize_timeseries(csv_df, ax=None, startup_time=0, end_time=None):
    """
    Overlays results from several experiments to create a single qsize plot for comparison.
    - Expects a single experiment's dataframe as input
    - Only plots primary flows (module number is 0, module type is server)
    """

    csv_df = csv_df[csv_df["module"].str.contains("router1")]
    csv_df = csv_df[csv_df["module"].str.contains("0")]
    csv_df = csv_df[csv_df["metric"].str.contains("queueLength")]

    if csv_df.empty:
        print("plot_qsize_timeseries(): CSV dataframe is empty. Returning.")
        return None
        
    print("Plotting qsize timeseries for:")
    print(csv_df)

    window = 100
    all_y_values = []
    for _, row in csv_df.iterrows():
        data = pd.read_csv(row["csv_path"])
        data = data[data["time"] > startup_time]
        if end_time:
            data = data[data["time"] < end_time]
        x = data["time"]    
        y = data["queueLength"]
        
        # Rolling stats
        rolling_mean = y.rolling(window, center=True).mean()
        rolling_std = y.rolling(window, center=True).std()

        # Raw data
        line, = ax.plot(x, y, alpha=0.2, linewidth=0.8, color=protocol_colors[row["protocol"]])

        # Smoothed mean
        ax.plot(x, rolling_mean, linewidth=2, label=row["protocol"], color=protocol_colors[row["protocol"]])

    ax.set_ylabel("Queue Size (pkts)")
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.set_yscale("linear")
    ax.ticklabel_format(style='plain', axis='y')
    if end_time:
        ax.set_xlim(right=end_time)
    return ax
        
def plot_cwnd_timeseries(csv_df, ax=None, show_competition=True, startup_time=0, end_time=None, flow_color_mode=FLOW_COLOR_MODE_SHARED, flow_color_map=None):
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
        data = data[data["time"] > startup_time]
        if end_time:
            data = data[data["time"] < end_time]
        style = get_timeseries_style(row, flow_color_mode, flow_color_map)
        if(show_competition or style["is_primary"]):
            x = data["time"]
            y = data["cwnd"]
            all_y_values.extend(data["cwnd"].values)
            # Rolling stats
            rolling_mean = y.rolling(window, center=True).mean()
            rolling_std = y.rolling(window, center=True).std()

            # Raw data
            line, = ax.plot(
                x, 
                y, 
                alpha=0.2, 
                linewidth=0.8, 
                color=style["color"],
                )

            # Smoothed mean
            ax.plot(
                x, 
                rolling_mean, 
                label=style["label"],
                color=style["color"],
                linestyle=style["linestyle"],
                lw= 1,
                alpha= style["alpha"],
                path_effects=[
                    pe.Stroke(linewidth=2, foreground='white', alpha=0.8),
                    pe.Normal()
                ]
                )
    
    ax.set_ylabel("cwnd (bytes)")
    ax.set_ylim(bottom=0, top=np.percentile(all_y_values, 99)*1.1)
    ax.set_xlim(left=0)
    ax.set_yscale("linear")
    ax.ticklabel_format(style='plain', axis='y')
    if end_time:
        ax.set_xlim(right=end_time)
    return ax

def plot_srtt_timeseries(csv_df, ax=None, show_competition=True, startup_time=0, end_time=None, flow_color_mode=FLOW_COLOR_MODE_SHARED, flow_color_map=None):
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
        data = data[data["time"] > startup_time]
        if end_time:
            data = data[data["time"] < end_time]
        style = get_timeseries_style(row, flow_color_mode, flow_color_map)
        if(show_competition or style["is_primary"]):
            ax.plot(
                data["time"],
                data["srtt"],
                label=style["label"],
                color=style["color"],
                linestyle=style["linestyle"],
                lw= 1,
                alpha= style["alpha"],
                path_effects=[
                    pe.Stroke(linewidth=2, foreground='white', alpha=0.8),
                    pe.Normal()
                ]
            )
            all_y_values.extend(data["srtt"].values)

    # Plot optimal delays only if there is a scenario file containing that info (mostly for responsiveness experiments)
    scenario_delay_df = scenario_df[scenario_df["metric"] == "delay"]

    if not scenario_delay_df.empty:
        scenario_delay_df = scenario_delay_df.iloc[0]
        scenario_delay_data = pd.read_csv(scenario_delay_df["csv_path"])

        if end_time:
            scenario_delay_data = scenario_delay_data[scenario_delay_data["time"] < end_time]
            # Add the last scenario entry at end_time so the optimal line reaches the end of the plot
            last_delay = scenario_delay_data.loc[scenario_delay_data["time"].idxmax(), "delay"]
            extra_delay_entry = pd.DataFrame([{"time": end_time, "datarate": last_delay}])
            scenario_delay_data = pd.concat([scenario_delay_data, extra_delay_entry], ignore_index=True)

        scenario_delay_data["delay"] *= 2 * .001 # one-way to two-way delay (RTT)
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
        
    ax.set_ylabel("sRTT (ms)")
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.set_yscale("linear")
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda y, _: f"{y*1000:g}") # Display y-axis in ms
    )
    if end_time:
        ax.set_xlim(right=end_time)
    return ax

def plot_throughput_timeseries(csv_df, ax=None, show_competition=True, startup_time=0, end_time=None, flow_color_mode=FLOW_COLOR_MODE_SHARED, flow_color_map=None):
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
    
    all_y_values = []
    for _, row in csv_df.iterrows():
        data = pd.read_csv(row["csv_path"])
        data = data[data["time"] > startup_time]
        if end_time:
            data = data[data["time"] < end_time]
        style = get_timeseries_style(row, flow_color_mode, flow_color_map)
        if(show_competition or style["is_primary"]):
            line = ax.plot(
                data["time"],
                data["throughput"],
                label=style["label"],
                color=style["color"],
                linestyle=style["linestyle"],
                lw= 1,
                alpha= style["alpha"],
                path_effects=[
                    pe.Stroke(linewidth=2, foreground='white', alpha=0.8),
                    pe.Normal()
                ]
            )
            all_y_values.extend(data["throughput"].values)
            
    # Plot optimal throughputs only if there is a scenario file containing that info (mostly for responsiveness experiments)
    scenario_throughput_df = scenario_df[scenario_df["metric"] == "datarate"]

    if not scenario_throughput_df.empty:
        scenario_throughput_df = scenario_throughput_df.iloc[0]
        scenario_throughput_data = pd.read_csv(scenario_throughput_df["csv_path"])

        if end_time:
            scenario_throughput_data = scenario_throughput_data[scenario_throughput_data["time"] < end_time]
            # Add the last scenario entry at end_time so the optimal line reaches the end of the plot
            last_bw = scenario_throughput_data.loc[scenario_throughput_data["time"].idxmax(), "datarate"]
            extra_bw_entry = pd.DataFrame([{"time": end_time, "datarate": last_bw}])
            scenario_throughput_data = pd.concat([scenario_throughput_data, extra_bw_entry], ignore_index=True)

        scenario_throughput_data["datarate"] *= 125000 * 8  # mbps
        
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

    ax.set_ylabel("Throughput (Mbps)")
    if not scenario_throughput_df.empty:
        ax.set_ylim(bottom=0, top=np.percentile(scenario_throughput_data["datarate"].values, 99)*1.1)
    else:
        ax.set_ylim(bottom=0, top=np.percentile(all_y_values, 99)*1.1)
    ax.set_xlim(left=0)
    ax.set_yscale("linear")
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda y, _: f"{y/1e6:g}") # makes the y-axis Mbps insead of Bps
    )
    if end_time:
        ax.set_xlim(right=end_time)
    return ax

def plot_goodput_timeseries(csv_df, ax=None, show_competition=True, startup_time=0, end_time=None, filter_spikes=True, flow_color_mode=FLOW_COLOR_MODE_SHARED, flow_color_map=None):
    """
    Overlays results from several experiments to create a single goodput plot for comparison.
    - Expects a single experiment's dataframe as input
    - Only plots primary flows (module number is 0, module type is server)
    """
    scenario_df = csv_df[csv_df["module_type"].str.contains("scenario")]
    csv_df = csv_df[csv_df["module_type"].str.contains("server")]
    csv_df = csv_df[csv_df["module"].str.contains("app")]
    csv_df = csv_df[csv_df["metric"].str.contains("goodput")]

    if csv_df.empty:
        print("plot_goodput_timeseries(): CSV dataframe is empty. Returning.")
        return None

    print("Plotting goodput timeseries for:")
    print(csv_df)
    
    all_y_values = []
    for _, row in csv_df.iterrows():
        data = pd.read_csv(row["csv_path"])
        data = data[data["time"] > startup_time]
        if end_time:
            data = data[data["time"] < end_time]
        if filter_spikes:
            data["goodput"] = hampel_filter(data["goodput"], window_size=10, n_sigmas=3) # remove spikes for clarity
        
        style = get_timeseries_style(row, flow_color_mode, flow_color_map)
        if(show_competition or style["is_primary"]):
            line = ax.plot(
                data["time"],
                data["goodput"],
                label=style["label"],
                color=style["color"],
                linestyle=style["linestyle"],
                lw= 1,
                alpha= style["alpha"],
                path_effects=[
                    pe.Stroke(linewidth=2, foreground='white', alpha=0.65),
                    pe.Normal()
                ]
            )
            all_y_values.extend(data["goodput"].values)

    # Plot optimal goodput only if there is a scenario file containing that info (mostly for responsiveness experiments)
    scenario_goodput_df = scenario_df[scenario_df["metric"] == "datarate"]

    if not scenario_goodput_df.empty:
        scenario_goodput_df = scenario_goodput_df.iloc[0]
        scenario_goodput_data = pd.read_csv(scenario_goodput_df["csv_path"])

        if end_time:
            scenario_goodput_data = scenario_goodput_data[scenario_goodput_data["time"] < end_time]
            # Add the last scenario entry at end_time so the optimal line reaches the end of the plot
            last_bw = scenario_goodput_data.loc[scenario_goodput_data["time"].idxmax(), "datarate"]
            extra_bw_entry = pd.DataFrame([{"time": end_time, "datarate": last_bw}])
            scenario_goodput_data = pd.concat([scenario_goodput_data, extra_bw_entry], ignore_index=True)

        scenario_goodput_data["datarate"] *= 125000 * 8  # mbps
        ax.plot(
                scenario_goodput_data["time"],
                scenario_goodput_data["datarate"],
                linestyle='--',
                linewidth=1,
                color='black',
                alpha=1,
                label="Optimal",
                drawstyle="steps-post",
                zorder=10
            )

    ax.set_ylabel("Goodput (Mbps)")
    if not scenario_goodput_df.empty:
        ax.set_ylim(bottom=0, top=np.percentile(scenario_goodput_data["datarate"].values, 99)*1.1)
    else:
        ax.set_ylim(bottom=0, top=np.percentile(all_y_values, 99)*1.1)
    ax.set_xlim(left=0)
    ax.set_yscale("linear")
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda y, _: f"{y/1e6:g}") # makes the y-axis Mbps insead of Bps
    )
    if end_time:
        ax.set_xlim(right=end_time)
    return ax

def plot_simsec_timeseries(csv_df, ax=None, startup_time=0, end_time=None):
    """
    Plots OMNeT++ simulation throughput reported by Broker as simsec/sec.
    """
    csv_df = csv_df[csv_df["module"].str.contains("broker")]
    csv_df = csv_df[csv_df["metric"] == "simsecPerSec"]

    if csv_df.empty:
        print("plot_simsec_timeseries(): CSV dataframe is empty. Returning.")
        return None

    print("Plotting simsecPerSec timeseries for:")
    print(csv_df)

    all_y_values = []
    for _, row in csv_df.iterrows():
        data = pd.read_csv(row["csv_path"])
        data = data[data["time"] > startup_time]
        if end_time:
            data = data[data["time"] < end_time]
        if data.empty:
            continue

        y = data["simsecPerSec"]
        rolling_window = min(10, max(1, len(y)))
        rolling_mean = y.rolling(rolling_window, center=True, min_periods=1).mean()

        ax.plot(
            data["time"],
            y,
            alpha=0.2,
            linewidth=0.8,
            color=protocol_colors[row["protocol"]],
        )
        ax.plot(
            data["time"],
            rolling_mean,
            label=row["protocol"],
            color=protocol_colors[row["protocol"]],
            linewidth=1.5,
        )
        all_y_values.extend(y.values)

    ax.set_ylabel("Sim Throughput\n(simsec/sec)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    if all_y_values:
        ax.set_ylim(top=np.percentile(all_y_values, 99) * 1.1)
    ax.set_yscale("linear")
    ax.ticklabel_format(style='plain', axis='y')
    if end_time:
        ax.set_xlim(right=end_time)
    return ax

def plot_timeseries(exp_df, startup_time=0, end_time=60, all=True, show_competing=False, size=.5, flow_color_mode=FLOW_COLOR_MODE_SHARED):
    if all:
        count = 7
    else:
        count = 4
    fig, axs = plt.subplots(count, 1, figsize=(15 * size, count * 5 * size))
    flow_color_map = build_flow_color_map(exp_df) if flow_color_mode == FLOW_COLOR_MODE_UNIQUE else None
    plot_goodput_timeseries(exp_df, axs[0], startup_time=startup_time,  end_time=end_time, show_competition=show_competing, flow_color_mode=flow_color_mode, flow_color_map=flow_color_map)
    plot_throughput_timeseries(exp_df, axs[1], startup_time=startup_time,  end_time=end_time, show_competition=show_competing, flow_color_mode=flow_color_mode, flow_color_map=flow_color_map)
    plot_srtt_timeseries(exp_df, axs[2], startup_time=startup_time, end_time=end_time, show_competition=show_competing, flow_color_mode=flow_color_mode, flow_color_map=flow_color_map)
    plot_cwnd_timeseries(exp_df, axs[3], startup_time=startup_time,  end_time=end_time, show_competition=show_competing, flow_color_mode=flow_color_mode, flow_color_map=flow_color_map)
    if all: 
        plot_pacerate_timeseries(exp_df, axs[4], startup_time=startup_time,  end_time=end_time, show_competition=show_competing, flow_color_mode=flow_color_mode, flow_color_map=flow_color_map)
        plot_qsize_timeseries(exp_df, axs[5], startup_time=startup_time,  end_time=end_time)
        plot_simsec_timeseries(exp_df, axs[6], startup_time=startup_time, end_time=end_time)
    fig.subplots_adjust(top=0.95, bottom=.07, left=.1, right=.97)
    axs[count-1].set_xlabel("Time (seconds)")
    # for ax_i in axs[0]:
    #     ax_i.set_xlabel("") 
    # for ax_i in axs[1]:
    #     ax_i.set_title("") 

    legend_handles, legend_labels = axs[0].get_legend_handles_labels()
    if show_competing and flow_color_mode == FLOW_COLOR_MODE_SHARED:
        print("CREATING CUSTOM HANDLE FOR COMPETING FLOW")
        custom_handle = Line2D(
            [0], [0],
            linestyle=(0, (1, .5)),
            color="grey",
            linewidth=2,
            label="competing cubic flow"
        )
        legend_handles.append(custom_handle)
        legend_labels.append("Competing Cubic flow")
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=len(legend_labels),
        frameon=False, 
        bbox_to_anchor=(.5, 1),
        fontsize=12,
    )
    
    return (fig, axs)

def plot_tcp_friendliness(csv_df, ax=None, show_competition=False, startup_time=30, end_time=None, size = .6):
    csv_df = csv_df[csv_df["module_type"].str.contains("server")]
    csv_df = csv_df[csv_df["module"].str.contains("conn")]
    csv_df = csv_df[csv_df["metric"].str.contains("throughput")]

    if csv_df.empty:
        print("plot_throughput_ratio_aggregate(): CSV dataframe is empty. Returning.")
        return None

    qsizes = sorted(csv_df["QSIZE"].unique())
    if ax is None:
        fig, axes = plt.subplots(1, len(qsizes), figsize=(6 * len(qsizes) * size, 6 * size), layout="constrained", sharex=True)
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
                    if end_time:
                        main_flow_data = main_flow_data[main_flow_data["time"] < end_time]
                    competing_flow = row[row["module"].str.contains("1")]
                    competing_flow = competing_flow[competing_flow["run"] == run].iloc[0]
                    competing_flow_data = pd.read_csv(competing_flow["csv_path"])
                    competing_flow_data = competing_flow_data[competing_flow_data["time"] > startup_time]
                    if end_time:
                        competing_flow_data = competing_flow_data[competing_flow_data["time"] < end_time]
                    
                    ratio = main_flow_data["throughput"].mean() / competing_flow_data["throughput"].mean()
                    ratios.append(ratio)

                mean = np.mean(ratios)
                min_val = np.min(ratios)
                max_val = np.max(ratios)

                y_vals.append(mean)
                y_err_lower.append(mean - min_val)
                y_err_upper.append(max_val - mean)

            ax_i.errorbar(
                list(map(parse_numeric, delays)),
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
        ax_i.tick_params(direction="inout")
        # ax_i.set_xticklabels(map(lambda value: parse_numeric(value, as_int=True), delays))
        ax_i.set_title(f"Buffer Size: {parse_numeric(qsize)}x BDP")

        if ax_i is axes_to_use[0]:
            ax_i.set_ylabel("Throughput Ratio")
        ax_i.set_yscale("log")
        ax_i.set_ylim(0.01, 100)
        ax_i.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
        ax_i.set_yticks([.01, .1, 1, 10, 100])
            
    if ax is None:
        # fig.suptitle("TCP Friendliness; Throughput ratio against competing cubic flow", fontsize=20, y=0.02, verticalalignment="bottom")
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
            w_pad=0.1,
            h_pad=0.6,
        )
        
        return (fig, axes_to_use)
    else:
        axes_to_use[0].legend(fontsize=8)
        return axes_to_use
    return axes_to_use


def plot_delay_aggregate(csv_df, ax=None, show_competition=False, startup_time=30, end_time=None, size = .6):
    csv_df = csv_df[csv_df["module_type"].str.contains("client")]
    csv_df = csv_df[csv_df["module"].str.contains("conn")]
    csv_df = csv_df[csv_df["metric"].str.contains("srtt")]

    if csv_df.empty:
        print("plot_throughput_ratio_aggregate(): CSV dataframe is empty. Returning.")
        return None

    qsizes = sorted(csv_df["QSIZE"].unique())
    if ax is None:
        fig, axes = plt.subplots(1, len(qsizes), figsize=(6 * len(qsizes) * size, 6 * size), layout="constrained", sharex=True)
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

                throughputs = []
                for run in row["run"].unique():
                    main_flow = row[row["module"].str.contains("0")]
                    main_flow = main_flow[main_flow["run"] == run].iloc[0]
                    main_flow_data = pd.read_csv(main_flow["csv_path"])
                    main_flow_data = main_flow_data[main_flow_data["time"] > startup_time]
                    if end_time:
                        main_flow_data = main_flow_data[main_flow_data["time"] < end_time]
                    
                    throughputs.append(main_flow_data["srtt"].mean()*1000/parse_numeric(delay)) # Convert to ms and normalize

                mean = np.mean(throughputs)
                min_val = np.min(throughputs)
                max_val = np.max(throughputs)

                y_vals.append(mean)
                y_err_lower.append(mean - min_val)
                y_err_upper.append(max_val - mean)

            ax_i.errorbar(
                list(map(parse_numeric, delays)),
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
        ax_i.tick_params(direction="inout")
        ax_i.set_title(f"Buffer Size: {parse_numeric(qsize)}x BDP")

        if ax_i is axes_to_use[0]:
            ax_i.set_ylabel("Normalized Delay")
        ax_i.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}x"))
        ax_i.set_ylim(bottom=0, top=5)
    
    
    
    if ax is None:
        # fig.suptitle("TCP Friendliness; Throughput ratio against competing cubic flow", fontsize=20, y=0.02, verticalalignment="bottom")
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
            w_pad=0.1,
            h_pad=0.6,
        )
        
        return (fig, axes_to_use)
    else:
        return axes_to_use
    return axes_to_use

def plot_throughput_aggregate(csv_df, ax=None, show_competition=False, startup_time=30, end_time=None, size = .6):
    csv_df = csv_df[csv_df["module_type"].str.contains("server")]
    csv_df = csv_df[csv_df["module"].str.contains("conn")]
    csv_df = csv_df[csv_df["metric"].str.contains("throughput")]

    if csv_df.empty:
        print("plot_throughput_ratio_aggregate(): CSV dataframe is empty. Returning.")
        return None

    qsizes = sorted(csv_df["QSIZE"].unique())
    if ax is None:
        fig, axes = plt.subplots(1, len(qsizes), figsize=(6 * len(qsizes) * size, 6 * size), layout="constrained", sharex=True, sharey=True)
        if len(qsizes) == 1:
            axes = [axes]
    else:
        axes = ax if isinstance(ax, (list, np.ndarray)) else [ax]
        print(axes)

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

                throughputs = []
                for run in row["run"].unique():
                    main_flow = row[row["module"].str.contains("0")]
                    main_flow = main_flow[main_flow["run"] == run].iloc[0]
                    main_flow_data = pd.read_csv(main_flow["csv_path"])
                    main_flow_data = main_flow_data[main_flow_data["time"] > startup_time]
                    if end_time:
                        main_flow_data = main_flow_data[main_flow_data["time"] < end_time]
                    
                    throughputs.append(main_flow_data["throughput"].mean()*.000001) # Convert to mbps

                mean = np.mean(throughputs)
                min_val = np.min(throughputs)
                max_val = np.max(throughputs)

                y_vals.append(mean)
                y_err_lower.append(mean - min_val)
                y_err_upper.append(max_val - mean)

            ax_i.errorbar(
                list(map(parse_numeric, delays)),
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
        
        ax_i.set_xlabel("Base RTT (ms)")
        ax_i.tick_params(direction="inout")
        # ax_i.set_xticklabels(map(lambda value: parse_numeric(value, as_int=True), delays))
        ax_i.set_title(f"Buffer Size: {parse_numeric(qsize)}x BDP")

        if ax_i is axes_to_use[0]:
            ax_i.set_ylabel("Throughput (Mbps)")
        ax_i.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
        ax_i.set_ylim(bottom=0)
    if ax is None:
        # fig.suptitle("TCP Friendliness; Throughput ratio against competing cubic flow", fontsize=20, y=0.02, verticalalignment="bottom")
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
            w_pad=0.1,
            h_pad=0.6,
        )
        
        return (fig, axes_to_use)
    else:
        return axes_to_use
    return axes_to_use

def plot_aggregate_metrics(exp_df, end_time=60, size=.7):
    fig, axs = plt.subplots(2, 3, figsize=(6 * 3 * size, 6 * size))
    fig.subplots_adjust(top=0.85, left=.05, right=.95)
    plot_throughput_aggregate(exp_df, axs[0], end_time=60)
    for ax_i in axs[0]:
        ax_i.set_xlabel("") 
    plot_delay_aggregate(exp_df, axs[1], end_time=60)
    for ax_i in axs[1]:
        ax_i.set_title("") 

    legend_handles, legend_labels = axs[0][0].get_legend_handles_labels()
    
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=len(legend_labels),
        frameon=False, 
        bbox_to_anchor=(.5, 1),
        fontsize=12,
    )
    fig.savefig(f"{os.getenv('RAYNET_PATH')}/_plots/{exp}_aggregate-metrics.pdf")
    plt.close(fig)

def plot_throughput_cdf(csv_df, ax=None, startup_time=15):
    scenario_df = csv_df[csv_df["module_type"].str.contains("scenario")]
    df = csv_df[csv_df["module_type"].str.contains("server")]
    df = df[df["module"].str.contains("conn")]
    df = df[df["metric"].str.contains("throughput")]
    max_throughput = 0 # Maintains maximum throughput observed, used for shading
    exp_end_time = 0 # Maintains the largest time value observed for throughput. Used to infer experiment duration, useful for getting time-weighted-mean from scenario values
    if df.empty:
        print("No data.")
        return

    protocols = sorted(df["protocol"].unique())

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = None

    for protocol in protocols:
        proto_df = df[df["protocol"] == protocol]
        
        # Collect the mean throughput of all runs
        throughputs = []
        for run in proto_df["run"].unique():
            row = proto_df[proto_df["run"] == run].iloc[0]
            data = pd.read_csv(row["csv_path"])
            exp_end_time = max(exp_end_time, data["time"].max())
            data = data[data["time"] > startup_time]
            mean_tp = data["throughput"].mean()
            throughputs.append(mean_tp)
        if len(throughputs) == 0:
            continue
        
        x = np.sort(throughputs)*.000001 # Converting to Mbps
        n = len(x)
        y = (np.arange(1, n + 1) / n)
        max_throughput = max(max_throughput, x.max())

        ax.plot(
            x,
            y * 100,
            label=protocol,
            color=protocol_colors[protocol],
            linestyle='-',
            linewidth=2
        )
        
    # Plot optimal throughputs only if there is a scenario file containing that info
    optimal_throughputs = []
    scenario_throughput_df = scenario_df[scenario_df["metric"] == "datarate"]
    if not scenario_throughput_df.empty:
        for run in scenario_throughput_df["run"].unique():
            row = scenario_throughput_df[scenario_throughput_df["run"] == run].iloc[0] # Will return a copy for each protocol, I only need one
            scenario_throughput_data = pd.read_csv(row["csv_path"])
            optimal_avg_throughput = time_weighted_mean(scenario_throughput_data, startup_time=startup_time, end_time=exp_end_time)
            optimal_throughputs.append(optimal_avg_throughput)
        
        x = np.sort(optimal_throughputs)
        n = len(x)
        y = (np.arange(1, n + 1) / n) * 100
        max_throughput = max(max_throughput, x.max())
        
        ax.plot(
            x,
            y,
            linestyle='--',
            linewidth=1,
            color='black',
            alpha=1,
            label="Optimal",
            zorder=10
        )
        ax.fill_between(
            x,
            y,
            0,
            color="black",
            alpha=.1
            )
    ax.set_xlabel("Average Throughput (Mbps)")
    ax.set_ylabel("% of Trials")
    ax.set_xlim(right=max_throughput)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}%"))
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.4)

    if fig is not None:
        ax.legend()
        fig.tight_layout()
        return fig, ax
    return ax

def plot_delay_cdf(csv_df, ax=None, startup_time=15):
    scenario_df = csv_df[csv_df["module_type"].str.contains("scenario")]
    df = csv_df[csv_df["module_type"].str.contains("client")]
    df = df[df["module"].str.contains("conn")]
    df = df[df["metric"].str.contains("srtt")]
    lowest_rtt = 99999 # Maintain lowest RTT observed for shading
    exp_end_time = 0 # Maintains the largest time value observed for throughput. Used to infer experiment duration, useful for getting time-weighted-mean from scenario values
    if df.empty:
        print("No data.")
        return

    protocols = sorted(df["protocol"].unique())

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = None

    for protocol in protocols:
        proto_df = df[df["protocol"] == protocol]
        
        # Collect the mean srtt of all runs
        delays = []
        for run in proto_df["run"].unique():
            row = proto_df[proto_df["run"] == run].iloc[0]
            data = pd.read_csv(row["csv_path"])
            exp_end_time = max(exp_end_time, data["time"].max())
            data = data[data["time"] > startup_time]
            mean_delay = data["srtt"].mean()
            delays.append(mean_delay)
        if len(delays) == 0:
            continue
        
        x = np.sort(delays) * 1000 # s to ms
        n = len(x)
        y = (np.arange(1, n + 1) / n)
        lowest_rtt = min(lowest_rtt, x.min())

        ax.plot(
            x,
            y * 100,
            label=protocol,
            color=protocol_colors[protocol],
            linestyle='-',
            linewidth=2
        )
        
    # Plot optimal delays only if there is a scenario file containing that info
    optimal_delays = []
    scenario_delay_df = scenario_df[scenario_df["metric"] == "delay"]
    if not scenario_delay_df.empty:
        for run in scenario_delay_df["run"].unique():
            row = scenario_delay_df[scenario_delay_df["run"] == run].iloc[0] # Will return a copy for each protocol, I only need one
            scenario_delay_data = pd.read_csv(row["csv_path"])
            optimal_avg_delay = time_weighted_mean(scenario_delay_data, startup_time=startup_time, end_time=exp_end_time)
            optimal_delays.append(optimal_avg_delay)
        
        x = np.sort(optimal_delays) * 2 # one-way delay to two-way delay (RTT)
        n = len(x)
        y = (np.arange(1, n + 1) / n) * 100
        lowest_rtt = min(lowest_rtt, x.min())

        ax.plot(
            x,
            y,
            linestyle='--',
            linewidth=1,
            color='black',
            alpha=1,
            label="Optimal",
            zorder=10
        )
        
        ax.fill_between(
            x,
            y,
            100,
            color="black",
            alpha=.1
            )
    ax.set_xlabel("Average RTT (ms)")
    ax.set_ylabel("% of Trials")
    ax.set_xlim(left=lowest_rtt)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}%"))
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.4)

    

    if fig is not None:
        ax.legend()
        fig.tight_layout()
        return fig, ax
    return ax

def plot_cdfs(exp_df, size=.6): 
    fig, axs = plt.subplots(1, 2, figsize=(12 * size, 3.5 * size))
    fig.subplots_adjust(top=0.8, bottom=.25, left=.10, right=.95, wspace=.3)
    plot_throughput_cdf(exp_df, axs[0])
    plot_delay_cdf(exp_df, axs[1])

    # fig.suptitle("Responsiveness: Performance over many random trials", fontsize=20, y=0.02, verticalalignment="bottom")
    legend_handles, legend_labels = axs[0].get_legend_handles_labels()
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

    fig.savefig(f"{os.getenv('RAYNET_PATH')}/_plots/{exp}_cdf.pdf")
    plt.close(fig)


"""
Automatically generates plots for all experiments. This is mostly built-to-purpose for the dissertation plots,
but can serve as a template for other experiment/plot automation by future maintainers as well.
- Timeseries plots are for a particular run
- CDF plots are intended for the responsiveness experiment, showing aggregate performance over many random trials
- TCP friendliness plots are intended for the competing flows experiment, showing throughput ratio against the competing cubic flow
"""
if __name__ == "__main__":    
    metric_csvs = create_csv_dict()        # dataframe containing [experiment, params, protocol, module, metric, csv_path] for easy access
    experiments = ["competing-flows", "responsiveness",] # List of experiments to generate plots for. Order matters for plot layering (e.g. competing flows should be last so it appears in the foreground of timeseries plots)
    for exp in experiments:
        exp_df = metric_csvs[metric_csvs["experiment"] == exp]
        
        # Special aggregate plots unique to each experiment
        if exp == "competing-flows":
            print("Plotting completing flows aggregate plots")
            fig, axes = plot_tcp_friendliness(exp_df, startup_time=30)
            fig.savefig(f"{os.getenv('RAYNET_PATH')}/_plots/{exp}_tcp-friendliness.pdf")
            plt.close(fig)
        elif exp == "responsiveness":
            print("Plotting responsiveness aggregate plots")
            plot_cdfs(exp_df)
        else:
            print("Plotting single flow aggregate plots")
            plot_aggregate_metrics(exp_df)

        # Summary Timeseries plots for all experiments (may be slow!)
        for params in exp_df["params"].unique():
            params_df = exp_df[exp_df["params"] == params]
            for run in params_df["run"].unique():
                run_df = params_df[params_df["run"] == run]
                if exp == "single-flow":
                    end_time = 60
                else:
                    end_time = 120
                if exp == "competing-flows":
                    start_time = 0
                    end_time = 100
                else:
                    start_time = 0
                show_competing = exp == "competing-flows"
                flow_color_mode = FLOW_COLOR_MODE_UNIQUE if exp == "competing-flows" else FLOW_COLOR_MODE_SHARED
                (fig, axs) = plot_timeseries(run_df, startup_time=start_time, end_time=end_time, all=True, show_competing=show_competing, flow_color_mode=flow_color_mode)
                fig.savefig(f"{os.getenv('RAYNET_PATH')}/_results/{exp}/{params}/run{int(run)}/summary.pdf")
                print(f"Plotted timeseries: {os.getenv('RAYNET_PATH')}/_results/{exp}/{params}/run{int(run)}/summary.pdf")
                plt.close(fig)

                
                
                
                
                
                
                
                
                
                
                
                
                
                
