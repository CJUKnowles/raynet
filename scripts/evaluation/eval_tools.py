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
from tabulate import tabulate
from matplotlib.backends.backend_pdf import PdfPages

def parse_if_number(s):
    try: return float(s)
    except: return True if s=="true" else False if s=="false" else s if s else None

def parse_ndarray(s):
    return np.fromstring(s, sep=' ') if s else None
    
def getResults(file):
    resultsFile = pd.read_csv(file, converters = {
    'attrvalue': parse_if_number,
    'binedges': parse_ndarray,
    'binvalues': parse_ndarray,
    'vectime': parse_ndarray,
    'vecvalue': parse_ndarray})
    vectors = resultsFile[resultsFile.type=='vector']
    return vectors

def dumb_plot(csv_file:str, output_name:str="plotted"):
    """
    Naively plots the given csv and exports to the same directory.
    Only useful for quick debugging.
    """
    print(f"Potting {csv_file}")
    # Read the CSV
    data = pd.read_csv(csv_file)

    # Assume the CSV has two columns: 'x' and 'y'
    x = data.iloc[:, 0]  # first column
    y = data.iloc[:, 1]  # second column

    # Plot
    plt.figure(figsize=(10,4))
    plt.plot(x, y)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Plot from CSV")
    plt.ylim(bottom=0)
    plt.yscale("linear")
    plt.ticklabel_format(style='plain', axis='y')
    plt.xlim(left=0)
    plt.grid(True)
    plt.tight_layout()
    
    
    export_dir = csv_file.rsplit("/", 1)[0]
    pdf_path = export_dir + f"/{output_name}.pdf"
    print(f"Saving to {pdf_path}")
    plt.savefig(pdf_path)


def generate_exp_csvs(filepath:str, protocol, protocol_nickname=None, exp_nickname=None, do_dumb_plots=False):
    """
    Uses the .vec file in the included results directory and produces a series of metric CSVs.
    The exp and protocol parameters are used to locate the correct vec file and to organize the output directories.
    do_dumb_plots=True will automatically produce a simple plot.pdf for each metric, for quick debugging.
    """
    argNum = 0
    vectorsToExtract = ["throughput", "srtt", "pacerate", "paceRate", "intervalDuration", "cwnd", "action", "queueLength", "queueBitLength", "incomingDataRate", "outgoingDataRate"]
    extracted = False
    if not protocol_nickname:
        protocol_nickname = protocol
    if not exp_nickname:
        exp_nickname = filepath.rsplit("/", 3)[1]
    
    vec_file = filepath + f"/{protocol}-#0.vec"
    out_path = filepath + "output.csv"
    
    # Create an index file, and output a combined results CSV
    cmd = f"""
                source ~/omnetpp/setenv &&
                opp_scavetool i "{vec_file}" &&
                opp_scavetool export "{vec_file}" -F CSV-R -o "{out_path}"
            """
    subprocess.Popen(cmd, shell=True, executable="/bin/bash").communicate(timeout=40)
    
    # Create an individual output CSV for each metric in vectorsToExtract
    extracted_paths = []
    rawResults = getResults(out_path)
    for vec in vectorsToExtract:
        results = rawResults.loc[(rawResults['name'] == str(vec)+":vector") | (rawResults['name'] == str(vec)+":vector(removeRepeats)")]
        print(results)
        for mod in range(len(results.vecvalue.to_numpy())):
            if(not results.vecvalue.to_numpy()[mod] is None):
                val = results.vecvalue.to_numpy()[mod] #VALUE
                time = results.vectime.to_numpy()[mod] #TIME
                modName = results.module.to_numpy()[mod]
                if 'thread' in modName:
                    modName = re.sub(r'\.thread_\d+', '', modName)
                modName = re.sub(r'(conn)-\d+', r'\1', modName)
                
                finallist = pd.DataFrame({'time': time, str(vec): val})
                subprocess.Popen("mkdir -p " + os.getenv('HOME') + '/raynet/results/' + exp_nickname + "/summary", shell=True).communicate(timeout=40)
                subprocess.Popen("mkdir -p " + os.getenv('HOME') + '/raynet/results/' + exp_nickname + "/runs", shell=True).communicate(timeout=40)
                subprocess.Popen("mkdir -p " + os.getenv('HOME') + '/raynet/results/' + exp_nickname + "/runs/" + protocol_nickname, shell=True).communicate(timeout=40)
                subprocess.Popen("mkdir -p " + os.getenv('HOME') + '/raynet/results/' + exp_nickname + "/csvs", shell=True).communicate(timeout=40) 
                subprocess.Popen("mkdir -p " + os.getenv('HOME') + '/raynet/results/' + exp_nickname + "/csvs/" + protocol_nickname, shell=True).communicate(timeout=40)
                subprocess.Popen("mkdir -p " + os.getenv('HOME') + '/raynet/results/' + exp_nickname + "/csvs/" + protocol_nickname + "/" + str(modName), shell=True).communicate(timeout=40)
                csv_path = os.getenv('HOME') + '/raynet/results/'+ exp_nickname +'/csvs/' + protocol_nickname + "/" + str(modName) + "/" + vec + '.csv'
                finallist.to_csv(csv_path, index=False)
                extracted = True
                if (do_dumb_plots): 
                    dumb_plot(csv_path, output_name=vec)
    termTime.sleep(1)

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
                module = dir_names[-1].split('.', 1)[1]   # Module name, excluding the network name prefix
                module_type = "client" if "client" in module else "server" if "server" in module else "queue" if "queue" in module else "other"
                protocol = dir_names[-2]
                experiment = dir_names[-4]
                
                csv_path = os.path.join(root, filename)
                
                csvs.append({
                    "experiment" : experiment,
                    "protocol": protocol,
                    "module": module,
                    "module_type": module_type,
                    "metric": metric,
                    "csv_path": csv_path
                            })

    return pd.DataFrame(csvs)

def plot_summary(experiment:str, protocols:list, metrics:list=None, results_dir:str=None):
    """
    Generates a set of naive timeseries plots from a series of protocols and their associated metrics.
    Pulls data from the raynet/results directory, and uses the experiment and protocol names provided to locate relevant files.
    If no metrics are provided, this function will plot anything it can find, and combine/match by metric name.
    """    
    if not results_dir:
        results_dir = os.getenv('HOME') + "/raynet/_experiments/experiment2/results"
        
    for metric in metrics:
        print()

def plot_pacerate_timeseries(csv_df, ax=None):
    """
    Overlays results from several experiments to create a single pacerate plot for comparison.
    - Expects a single experiment's dataframe as input
    - Only plots primary flows (module number is 0, module type is server)
    """

    csv_df = csv_df[csv_df["module_type"].str.contains("client")]
    csv_df = csv_df[csv_df["module"].str.contains("0")]
    csv_df = csv_df[csv_df["module"].str.contains("conn")]
    csv_df = csv_df[csv_df["metric"].str.contains("paceRate")]

    if csv_df.empty:
        print("plot_paceRate_timeseries(): CSV dataframe is empty. Returning.")
        return None

    print("Plotting paceRate timeseries for:")
    print(tabulate(csv_df, headers='keys'))

    all_y_values = []
    for _, row in csv_df.iterrows():
        data = pd.read_csv(row["csv_path"])
        
    window = 50
    all_y_values = []
    for _, row in csv_df.iterrows():
        data = pd.read_csv(row["csv_path"])
        
        x = data["time"]
        y = data["paceRate"]
        
        # Rolling stats
        rolling_mean = y.rolling(window, center=True).mean()
        rolling_std = y.rolling(window, center=True).std()

        # Raw data
        line, = ax.plot(x, y, alpha=0.2, linewidth=0.8)
        color = line.get_color()

        # Smoothed mean
        ax.plot(x, rolling_mean, linewidth=2, label=row["protocol"], color=color)
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
    print(tabulate(csv_df, headers='keys'))

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
        line, = ax.plot(x, y, alpha=0.2, linewidth=0.8)
        color = line.get_color()

        # Smoothed mean
        ax.plot(x, rolling_mean, linewidth=2, label=row["protocol"], color=color)

    ax.set_xlabel("Time")
    ax.set_ylabel("Queue Size (bits)")
    ax.set_title("Queue Size Over Time")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    ax.set_yscale("linear")
    ax.ticklabel_format(style='plain', axis='y')
    ax.grid(True)

    return ax
        
def plot_cwnd_timeseries(csv_df, ax=None):
    """
    Overlays results from several experiments to create a single cwnd plot for comparison.
    - Expects a single experiment's dataframe as input
    - Only plots primary flows (module number is 0, module type is server)
    """

    csv_df = csv_df[csv_df["module_type"].str.contains("client")]
    csv_df = csv_df[csv_df["module"].str.contains("0")]
    csv_df = csv_df[csv_df["module"].str.contains("conn")]
    csv_df = csv_df[csv_df["metric"].str.contains("cwnd")]

    if csv_df.empty:
        print("plot_cwnd_timeseries(): CSV dataframe is empty. Returning.")
        return None

    print("Plotting cwnd timeseries for:")
    print(tabulate(csv_df, headers='keys'))

    all_y_values = []
    for _, row in csv_df.iterrows():
        data = pd.read_csv(row["csv_path"])
    
    window = 1
    all_y_values = []
    for _, row in csv_df.iterrows():
        data = pd.read_csv(row["csv_path"])
        
        x = data["time"]
        y = data["cwnd"]
        
        # Rolling stats
        rolling_mean = y.rolling(window, center=True).mean()
        rolling_std = y.rolling(window, center=True).std()

        # Raw data
        line, = ax.plot(x, y, alpha=0.2, linewidth=0.8)
        color = line.get_color()

        # Smoothed mean
        ax.plot(x, rolling_mean, linewidth=2, label=row["protocol"], color=color)

        # Variance band
        ax.fill_between(
            x,
            rolling_mean - rolling_std,
            rolling_mean + rolling_std,
            alpha=0.2,
            color=color
        )

    ax.set_xlabel("Time")
    ax.set_ylabel("cwnd (bytes)")
    ax.set_title("cwnd over time")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    ax.set_yscale("linear")
    ax.ticklabel_format(style='plain', axis='y')
    ax.grid(True)

    return ax

def plot_srtt_timeseries(csv_df, ax=None):
    """
    Overlays results from several experiments to create a single srtt plot for comparison.
    - Expects a single experiment's dataframe as input
    - Only plots primary flows (module number is 0, module type is server)
    """

    csv_df = csv_df[csv_df["module_type"].str.contains("client")]
    csv_df = csv_df[csv_df["module"].str.contains("0")]
    csv_df = csv_df[csv_df["module"].str.contains("conn")]
    csv_df = csv_df[csv_df["metric"].str.contains("srtt")]

    if csv_df.empty:
        print("plot_srtt_timeseries(): CSV dataframe is empty. Returning.")
        return None

    print("Plotting srtt timeseries for:")
    print(tabulate(csv_df, headers='keys'))

    all_y_values = []
    for _, row in csv_df.iterrows():
        data = pd.read_csv(row["csv_path"])
        ax.plot(
            data["time"],
            data["srtt"],
            label=row["protocol"]
        )
        all_y_values.extend(data["srtt"].values)

    ax.set_xlabel("Time")
    ax.set_ylabel("sRTT (ms)")
    ax.set_title("sRTT over time")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0, top=np.percentile(all_y_values, 99)*1.1)
    ax.set_yscale("linear")
    ax.ticklabel_format(style='plain', axis='y')
    ax.grid(True)

    return ax

def plot_throughput_timeseries(csv_df, ax=None):
    """
    Overlays results from several experiments to create a single throughput plot for comparison.
    - Expects a single experiment's dataframe as input
    - Only plots primary flows (module number is 0, module type is server)
    """

    csv_df = csv_df[csv_df["module_type"].str.contains("server")]
    csv_df = csv_df[csv_df["module"].str.contains("0")]
    csv_df = csv_df[csv_df["module"].str.contains("conn")]
    csv_df = csv_df[csv_df["metric"].str.contains("throughput")]

    if csv_df.empty:
        print("plot_throughput_timeseries(): CSV dataframe is empty. Returning.")
        return None

    print("Plotting throughput timeseries for:")
    print(tabulate(csv_df, headers='keys'))
    
    all_y_values = []
    for _, row in csv_df.iterrows():
        data = pd.read_csv(row["csv_path"])
        
        ax.plot(
            data["time"],
            data["throughput"],
            label=row["protocol"]
        )
        all_y_values.extend(data["throughput"].values)

    ax.set_xlabel("Time")
    ax.set_ylabel("Throughput (Mbps)")
    ax.set_title("Throughput over time")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0, top=np.percentile(all_y_values, 99)*1.1)
    ax.set_yscale("linear")
    ax.ticklabel_format(style='plain', axis='y')
    ax.grid(True)

    return ax

if __name__ == "__main__":
    # experimentNames = ["double-flow-dumbbell", "single-flow", "responsiveness"]
    # protocols = ["Cubic", "Orca"]
    # for experimentName in experimentNames:
    #     for protocol in protocols:
    #         # Generate output.csv and individual vector csvs for all tracked vectors of agiven experiment
    #         exp_results_dir = os.getenv('HOME') + f"/raynet/_experiments/{experimentName}/ini_variants/results"
    #         generate_exp_csvs(exp_results_dir, protocol, do_dumb_plots=True)
    
    
    metric_csvs = create_csv_dict()        # dataframe containing [experiment, protocol, module, metric, csv_path] for easy access
    experiments = ["single-flow", "responsiveness", "double-flow-dumbbell"]
    for exp in experiments:
        exp_df = metric_csvs[metric_csvs["experiment"] == exp]
        fig, axs = plt.subplots(20, 1, figsize=(15, 100))
        
        
        plot_throughput_timeseries(exp_df, axs[0])
        plot_cwnd_timeseries(exp_df, axs[1])
        plot_pacerate_timeseries(exp_df, axs[2])
        plot_srtt_timeseries(exp_df, axs[3])
        plot_qsize_timeseries(exp_df, axs[4])
        
        fig.tight_layout()
        fig.savefig(os.getenv('HOME') + f"/raynet/results/{exp}/summary.pdf")
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