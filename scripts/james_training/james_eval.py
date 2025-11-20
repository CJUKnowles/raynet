import pandas as pd
import matplotlib.pyplot as plt
import os
from pypdf import PdfWriter

def combine_pdfs(folder_path, output="combined.pdf"):
    merger = PdfWriter()

    # Get all pdf files in the folder, sorted alphabetically
    pdfs = sorted(f for f in os.listdir(folder_path) if f.lower().endswith(".pdf"))

    if not pdfs:
        print("No PDF files found.")
        return

    for pdf in pdfs:
        full_path = os.path.join(folder_path, pdf)
        print(f"Adding: {full_path}")
        merger.append(full_path)

    merger.write(folder_path + "/" + output)
    merger.close()
    print(f"Combined PDF saved as: {output}")

def plot_by_metric(df, metric:str, path:str, figure_label:str=None, x_label:str=None, y_label:str=None, color:str=None):

    y_label = y_label if y_label else metric
    x_label = x_label if x_label else "Training Iteration"
    color = color if color else "orange"
    figure_label = figure_label if figure_label else metric

    plt.figure(figsize=(10,5))
    plt.plot(df['training_iteration'], df[metric], color='orange', label=figure_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title("RAM Usage During RLlib Training")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

if __name__ == '__main__':
    results_path = "/home/cjuknowles/ray_results/DQN_OmnetGymApiEnv_2025-11-20_13-42-11vckswx5v/"

    df = pd.read_csv(results_path + "progress.csv")

    plt.plot(df["training_iteration"], df["env_runners/episode_return_mean"])
    plt.fill_between(
        df["training_iteration"],
        df["env_runners/episode_return_min"],
        df["env_runners/episode_return_max"],
        alpha=0.2
    )

    plt.xlabel("Training Iteration")
    plt.ylabel("Episode Return")
    plt.title("Learning Curve")
    plt.grid(True)
    plt.show()

    plt.savefig(results_path + "learning_curve.pdf", dpi=200)
    plot_by_metric(df, "perf/ram_util_percent", results_path + "Ram_util.pdf")
    plot_by_metric(df, "perf/cpu_util_percent", results_path + "CPU_util.pdf")
    plot_by_metric(df, "perf/gpu_util_percent0", results_path + "GPU_util.pdf")

    combine_pdfs(results_path)

