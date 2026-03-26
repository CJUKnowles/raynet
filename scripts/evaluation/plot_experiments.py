import os
import pandas as pd
from pathlib import Path
import pprint
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import eval_tools

subflow_colors_list = ["#1f77bf", "#ff7f0e", "#2ca02c", "#d6272b", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
gradient_colors_list = ["#1f77bf","#3c6ab1","#5a5da3","#785095", "#964387","#b43679", "#d2296b", "#e63946",]
cmap = mcolors.LinearSegmentedColormap.from_list(
    "blue_red", ["#1f77bf", "#d62728"]
)

RESET = "\033[0m"
def printGreen(string):
    COLOR = "\033[32m"
    print("\r\033[K", end='', flush=True)
    print(f"{COLOR}{string}{RESET}")

def plot_all(path:Path):
    results_file = path / "all_experiments_summary.csv"
    if not results_file.exists():
        printGreen(f"{results_file} does not exist. Aborting.")
        return None
    all_results_df = pd.read_csv(results_file)
    printGreen(all_results_df)

    fig, axs = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    axs = axs.flatten()
    plot_num = 0
    # Plot average goodput against subflow count, for each controller/experiment combo
    # for experiment in all_results_df['experiment'].unique():
    #     for flow_count in all_results_df['flows'].unique():
    for controller in all_results_df['controller'].unique():
        required_values = {
            'experiment': 'manhattan_openflow_random_flooded',
            'controller': controller,
            'flows': 8
        }
        group_by = ['subflows']
        agg_columns = ['average_goodput']
        agg_function = get_mean
        ax = bar_plot(all_results_df, required_values, group_by, agg_columns, agg_function, axs[plot_num], debug=True)
        if ax != None:
            axs[plot_num] = ax
            plot_num += 1
    fig.suptitle('Average Goodput vs Number of Subflows', fontsize=14, fontweight='bold')
    #plt.tight_layout()
    plot_path = path / 'firstplot.pdf'
    plt.savefig(plot_path)
    printGreen(f"Plot saved to: {plot_path}")
    #plt.show()

def bar_plot(df:pd.DataFrame, required_values:dict, group_by:list[str], agg_columns:list[str], agg_function, ax, debug=False, debug_paths=False, ):
    """
    General-use function that can creates and returns a bar plot using an arbitrary set of parameters from the input data.
        required_values: Any df rows that do not have the correct values will be dropped before plotting
        group_by: What columns the df should be grouped by before aggregation. This is usually the parameter we are interested in showing the behaviour of.
        agg_columns: Which columns should be aggregated together to produce a single result value.
        agg_function: What function will be called with the agg_columns as parameters, will return the final result value for a given group.
    
    This setup may seem a bit complicated, but it is very flexible.
    The following example can be used to produce a plot showing how goodput increases with subflow count.
        required_values: {"experiment": "experiment_manhattan_openflow_flooded", "controller": "k-shortest"}
        group_by: ['subflows']
        agg_columns: ['average_goodput']
        agg_function: get_mean
    """

    # Copy the df and filter out any values that do not match the required_values dict
    filtered_df = df.copy()
    for col, val in required_values.items():
        filtered_df = filtered_df[filtered_df[col] == val]
    if filtered_df.empty:
        return None
    grouped = filtered_df.groupby(group_by)
    agg_results = grouped[agg_columns].agg(agg_function)

    if debug_paths:
        pd.set_option('display.max_colwidth', None)
        for group_name, group_data in grouped:
            print(str(group_data['plot_path']))
            break 
    if debug:
        printGreen(required_values)
        printGreen(group_by)
        printGreen(agg_results)
        printGreen('--------------------------')

    # Plot the dataframe
    ax = agg_results.plot(kind="bar", legend=False, ax=ax)
    ax.set_xlabel(group_by[0])
    ax.set_ylabel(agg_columns[0])
    ax.set_title(f'{agg_columns[0]} by {group_by[0]}')

    # Dynamically set bar colors 
    labels = [int(tick.get_text()) for tick in ax.get_xticklabels()]
    norm = mcolors.Normalize(vmin=min(labels), vmax=max(labels))
    for bar, label in zip(ax.patches, labels):
        color = cmap(norm(label))
        bar.set_facecolor(color)

    return ax

def get_sum(list):
    """
    proof of concept function that returns the sum of the provided values
    """
    return np.sum(list)

def get_mean(list):
    """
    proof of concept function that returns the mean of the provided values
    """
    return np.mean(list)

def get_custom_agg_thing(list):
    """
    Proof of concept function that returns some special transformation of the the provided values
    """
    added = []
    for item in list:
        added.append(item + len(added) + 7)
    return added[5] + added[6] + added[-7]

# def plot_double_flows_competition(df, axs):
#     for experiment in df['experiment'].unique():
#         required_values = {
#             'experiment': experiment
#         }
#         group_by = ['protocol']
#         agg_columns = ['throughput']
#         agg_function = get_mean
#         ax = bar_plot(df, required_values, group_by, agg_columns, agg_function, axs[plot_num], debug=True)
#         if ax != None:
#             axs[plot_num] = ax
#             plot_num += 1
#     return axs

def line_plot(df: pd.DataFrame, plot_info: dict, ax: plt.Axes):
    df_copy = df.copy()

    # Filter rows based on required values
    for column, value in plot_info.get('required_values', {}).items():
        df_copy = df_copy[df_copy[column] == value]

    if df_copy.empty:
        ax.set_title("No data")
        return ax

    # Plot each matching CSV as a separate line
    for _, row in df_copy.iterrows():
        try:
            data = pd.read_csv(row["csv_path"])

            # Expect columns: time, value
            ax.plot(
                data.iloc[:, 0],   # time
                data.iloc[:, 1],   # value
                label=f"{row['protocol']} | {row['module']}"
            )

        except Exception as e:
            print(f"Failed to load {row['csv_path']}: {e}")

    # Labels and title
    ax.set_title(plot_info.get("title", "Line Plot"))
    ax.set_xlabel("Time")
    ax.set_ylabel(plot_info.get("ylabel", "Value"))

    if len(df_copy) > 1:
        ax.legend(fontsize=8)

    return ax

if __name__ == "__main__":
    path = os.getenv('HOME') + "/raynet/results"
    metric_csvs = eval_tools.create_csv_dict()


    # competition plots:
    # Make Plots
    experiments = ["double-flow-dumbbell"]
    metrics = ["throughput", "srtt", "pacerate", "paceRate", "intervalDuration", "cwnd", "action", "incomingDataRate", "outgoingDataRate", "queueBitLength"]
    module_types = ["server", "client", "queue"]
    for experiment in experiments:
        exp_df = metric_csvs[metric_csvs["experiment"] == experiment]
        for module_type in module_types:
            module_type_df = exp_df[exp_df["module_type"] == module_type]
            for metric in metrics:
                metric_df = module_type_df[module_type_df["metric"] == metric]
                if metric_df.empty:
                    print("Dataframe is emtpy, continuing")
                    continue
                
                for protocol in metric_df['protocol'].unique():
                    protocol_df = metric_df[metric_df['protocol'] == protocol]
                    plt.figure(figsize=(15,6))
                    for _, row in protocol_df.iterrows():
                        csv_data = pd.read_csv(row["csv_path"])
                        plt.plot(csv_data["time"], csv_data[metric], label=f'{row["protocol"]}: {row["module"]}')
                    
                    plt.xlabel("Time")
                    plt.ylabel(metric)
                    plt.title(f"{metric} vs competing flow")
                    plt.legend(fontsize=8)  # smaller font if many lines
                    plt.ylim(bottom=0)
                    plt.yscale("linear")
                    plt.ticklabel_format(style='plain', axis='y')
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(os.getenv('HOME') + f"/raynet/results/{experiment}/runs/{protocol}/COMPETITION-{metric}-{module_type}.pdf")
                    plt.close()
    # fig, axs = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    # axs = axs.flatten()

    # PLOT_NUM = 0
    
    # for protocol in df['protocol'].unique():
    #     # Example plot config
    #     flow_comparison_plot = {
    #         "required_values": {
    #             "experiment": "single-flow",
    #             "module_type": "server",
    #             "metric": "throughput",
    #             "protocol": protocol
    #         },
    #         "title": "Flow Competition Throughput",
    #         "ylabel": "Throughput"
    #     }

    #     line_plot(df, flow_comparison_plot, axs[PLOT_NUM])
    #     PLOT_NUM += 1

    # # You can add more plots here using the same pattern
    # # line_plot(df, another_plot_config, axs[PLOT_NUM])
    # # PLOT_NUM += 1

    plt.savefig(path + "/test.pdf")