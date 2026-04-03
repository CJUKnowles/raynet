import os
import subprocess
import eval_tools

experiments_dir = f"{os.getenv('HOME')}/raynet/_experiments"
experiment_paths = {
    "single-flow": f"{experiments_dir}/single-flow/single-flow.ini",
    "double-flow-dumbbell": f"{experiments_dir}/double-flow-dumbbell/double-flow-dumbbell.ini",
    "responsiveness": f"{experiments_dir}/responsiveness/responsiveness.ini",
}

runner_paths = {
    "Orca": f"{os.getenv('HOME')}/raynet/simlibs/Orca/src/OrcaEval.py",
    "Cubic": f"{os.getenv('HOME')}/raynet/simlibs/Orca/src/CubicEval.py",
    "Astrea": f"{os.getenv('HOME')}/raynet/simlibs/Astrea/src/AstreaEval.py",
}


def run_experiments(experiments_dict, create_output_csv=True):
    """
    Runs the given experiment for each protocol listed, using their respective eval runner scripts
    Input:
        A dict containing the experiments to run as keys, and the protoocls to run them with as values.
    """
    for experiment_name, protocol_names in experiments_dict.items():
        python = f"{os.getenv('HOME')}/raynet/.venv/bin/python"
        experiment_path = experiment_paths[experiment_name]
        for protocol_name in protocol_names:
            print(f"\t ---------- {protocol_name} running experiment: {experiment_name} ----------")
            protocol_runner = runner_paths[protocol_name]
            os.system(f"{python} {protocol_runner} {experiment_path}")
            
            # Generate output.csv and individual vector csvs for all tracked vectors of agiven experiment
            exp_results_dir = os.getenv('HOME') + f"/raynet/_experiments/{experiment_name}/ini_variants/results"
            eval_tools.generate_exp_csvs(exp_results_dir, protocol_name, do_dumb_plots=False)
        
if __name__ == "__main__":
    
    experiments_to_run = {
        #"single-flow": ["Astrea"],
        "double-flow-dumbbell": ["Astrea"],
        #"responsiveness": ["Astrea"],
    }
    
    run_experiments(experiments_to_run)