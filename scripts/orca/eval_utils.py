import pandas as pd
import matplotlib.pyplot as plt
import os
from pypdf import PdfWriter

def _combine_pdfs(pdf_paths, output_path):
    if not pdf_paths:
        print("No PDF files found.")
        return False # Error
    if not output_path:
        print("Output not provided.")
        return False # Error
    
    
    merger = PdfWriter()
    for path in pdf_paths:
        if not path.lower().endswith(".pdf"):
            print(f"{path} is not a .pdf file! Omitting.")
            break
        merger.append(path)
    merger.write(output_path)
    merger.close()
    
    print(f"Combined PDF saved to: {output_path}")

def _plot_by_metric(df, metric:str, output_path:str, figure_label:str=None, x_label:str=None, y_label:str=None, color:str=None, title:str=None):
    print(f"Plotting metric: {metric}")
    y_label = y_label if y_label else metric
    x_label = x_label if x_label else "Training Iteration"
    color = color if color else "orange"
    figure_label = figure_label if figure_label else metric

    plt.figure(figsize=(10,5))
    plt.plot(df['training_iteration'], df[metric], color='orange', label=figure_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title if title else f"{y_label} over {x_label}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return(output_path)




def plot_experiment_summary(df, output_dir, output_name):
    plot_paths = [
        # _plot_by_metric(df, metric="ray/tune/perf/ram_util_percent", output_path=output_dir+"Ram_util.pdf", title="RAM Utilization over time"),
        # _plot_by_metric(df, metric="ray/tune/perf/cpu_util_percent", output_path=output_dir+"CPU_util.pdf", title="CPU Utilization over time"),
        # _plot_by_metric(df, metric="ray/tune/perf/gpu_util_percent0", output_path=output_dir+"GPU_util.pdf", title="GPU Utilization over time")
    ]
    
    for metric in df.columns:
        if "return" in metric or "reward" in metric or "ram_until_percent" in metric or "cpu_util_percent" in metric or "gpu_util_percent" in metric:
            sanitized_metric_name = metric.replace("/", "-")
            plot_paths.append(
                 _plot_by_metric(df, metric=metric, output_path=output_dir+f"{sanitized_metric_name}.pdf")
            )

    _combine_pdfs(pdf_paths=plot_paths, output_path=os.path.join(output_dir, output_name))
    
if __name__ == '__main__':
    print("Running this script directly is deprected. Will likely be removed soon.")
    results_path = f"{os.getenv('HOME')}/ray_results/James_training/PPO_OmnetGymApiEnv_a3318_00000_0_2026-02-04_12-40-07/"
    

    df = pd.read_csv(results_path + "progress.csv")
    plot_experiment_summary(df, results_path, "time_series.pdf")