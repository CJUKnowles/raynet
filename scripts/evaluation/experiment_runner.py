import os
import subprocess
import eval_tools
import itertools

experiments_dir = f"{os.getenv('HOME')}/raynet/_experiments"
experiment_paths = {
    "single-flow": f"{experiments_dir}/single-flow/single-flow.ini",
    "single-flow": f"{experiments_dir}/single-flow-dynamic/single-flow-dynamic.ini",
    "double-flow-dumbbell": f"{experiments_dir}/double-flow-dumbbell/double-flow-dumbbell.ini",
    "responsiveness": f"{experiments_dir}/responsiveness/responsiveness.ini",
}

runner_paths = {
    "Orca": f"{os.getenv('HOME')}/raynet/simlibs/Orca/src/OrcaEval.py",
    "Cubic": f"{os.getenv('HOME')}/raynet/simlibs/Orca/src/CubicEval.py",
    "Astrea": f"{os.getenv('HOME')}/raynet/simlibs/Astrea/src/AstreaEval.py",
}

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

def run_experiments(experiments_dict, create_output_csv=True):
    """
    Runs the given experiment for each protocol listed, using their respective eval runner scripts
    Input:
        A dict containing the experiments to run as keys, and the protoocls to run them with as values.
    """
    python = f"{os.getenv('HOME')}/raynet/.venv/bin/python"
    
    # Perform each experiment with each protocol and parameter combination
    for experiment_name in experiments_dict.keys():
        original_ini_file = experiment_paths[experiment_name]
        ini_variants_base = f"{original_ini_file.rsplit("/", 1)[0]}/ini_variants/{original_ini_file.rsplit("/", 1)[1]}"
        
        # Create a list of all unique param combinations as dicts
        params_dict = experiments_dict[experiment_name]["params"]
        param_keys = list(params_dict.keys())
        param_values_product = list(itertools.product(*params_dict.values()))
        unique_param_combinations = [dict(zip(param_keys, param_values)) for param_values in param_values_product]
    
        # Generate a .ini file for each unique param combination
        for params in unique_param_combinations:
            with open(original_ini_file, 'r') as fin:
                ini_string = fin.read()
            params_suffix = ""
            for param, value in params.items():
                ini_string = ini_string.replace(param + "_PLACEHOLDER",  value)
                params_suffix = params_suffix + f"_{value}-{param}"
            modified_ini_file = ini_variants_base + params_suffix
            with open(modified_ini_file, 'w') as fout:
                fout.write(ini_string)
        
        # Perform all generated experiment files for each protocol
        for protocol_name in experiments_dict[experiment_name]["protocols"]:
            for params in unique_param_combinations: # Looping through a second time separately so I don't have to generate all the files multiple times
                # Run the experiment for this protocol and param combo
                for param, value in params.items():
                    params_suffix = params_suffix + f"_{value}-{param}"
                modified_ini_file = ini_variants_base + params_suffix
                protocol_runner_path = runner_paths[protocol_name]
                print(f"\t ---------- {protocol_name} running experiment: {experiment_name + params_suffix} ----------")
                os.system(f"{python} {protocol_runner_path} {modified_ini_file}") # Finally runs the exp
                
                # Generate output.csv and individual vector csvs for all tracked vectors for this exp/protocol/params combo
                exp_results_dir = os.getenv('HOME') + f"/raynet/_experiments/{experiment_name}/ini_variants/results"
                eval_tools.generate_exp_csvs(exp_results_dir, protocol_name, exp_nickname=f"{experiment_name + params_suffix}")
        
if __name__ == "__main__":
    experiments_to_run = {
        "single-flow-dynamic": {
            "protocols": ["Cubic"],
            "params": {
                "BANDWIDTH" : ["1Mbps","2Mbps","3Mbps"],
                "DELAY"     : ["1ms","10ms","100ms"],    
                "QUEUE_SIZE": [".2b","1b","2b"],
                }
            },
        # "responsiveness": ["Cubic"], # TODO: change format to add protocols and params once everything is working 
        # "double-flow-dumbbell": ["Cubic"], # TODO: change format to add protocols and params once everything is working 
    }
    
    run_experiments(experiments_to_run)