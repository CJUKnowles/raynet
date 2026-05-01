import os
import subprocess
import itertools
import pandas as pd
import numpy as np
import random
from pathlib import Path
import os
import subprocess
import re
import time as termTime
import xml.etree.ElementTree as ET

experiments_dir = f"{os.getenv('HOME')}/raynet/_experiments"
experiment_paths = {
    "single-flow": f"{experiments_dir}/single-flow/single-flow.ini",
    "competing-flows": f"{experiments_dir}/competing-flows/competing-flows.ini",
    "responsiveness": f"{experiments_dir}/responsiveness/responsiveness.ini",
}

runner_paths = {
    "Orca": f"{os.getenv('HOME')}/raynet/simlibs/Orca/src/OrcaEval.py",
    "Cubic": f"{os.getenv('HOME')}/raynet/simlibs/Orca/src/CubicEval.py",
    "Astrea": f"{os.getenv('HOME')}/raynet/simlibs/Astrea/src/AstreaEval.py",
    "CleanSlate": f"{os.getenv('HOME')}/raynet/simlibs/CleanSlate/src/CleanSlateEval.py",
}

def parse_numeric(value): 
    match = re.search(r"[-+]?\d*\.?\d+", str(value))
    return float(match.group()) if match else None

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


def generate_exp_csvs(filepath:str, protocol, protocol_nickname=None, exp_nickname=None, do_dumb_plots=False, params_str=None, short_params_str=None, run=None):
    """
    Uses the .vec file in the included results directory and produces a series of metric CSVs.
    The exp and protocol parameters are used to locate the correct vec file and to organize the output directories.
    do_dumb_plots=True will automatically produce a simple plot.pdf for each metric, for quick debugging.
    - params_str is used to find the name of the experiment
    - short_params_str is used to name the csv directory (I use this to remove param names like BANDWIDTH)
    """
    if not params_str:
        params_str = "__default__"
    if not short_params_str:
        short_params_str = params_str
    if not run:
        run = 0
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
        print(vec)
        for mod in range(len(results.vecvalue.to_numpy())):
            if(not results.vecvalue.to_numpy()[mod] is None):
                val = results.vecvalue.to_numpy()[mod] #VALUE
                time = results.vectime.to_numpy()[mod] #TIME
                modName = results.module.to_numpy()[mod]
                if 'thread' in modName:
                    modName = re.sub(r'\.thread_\d+', '', modName)
                modName = re.sub(r'(conn)-\d+', r'\1', modName)
                
                final_list = pd.DataFrame({'time': time, str(vec): val})
                subprocess.Popen("mkdir -p " + os.getenv('HOME') + '/raynet/results/' + exp_nickname + "/" + short_params_str, shell=True).communicate(timeout=40)
                subprocess.Popen("mkdir -p " + os.getenv('HOME') + '/raynet/results/' + exp_nickname + "/" + short_params_str + f"/run{run}", shell=True).communicate(timeout=40)
                subprocess.Popen("mkdir -p " + os.getenv('HOME') + '/raynet/results/' + exp_nickname + "/" + short_params_str + f"/run{run}" + "/"  + protocol_nickname, shell=True).communicate(timeout=40)
                subprocess.Popen("mkdir -p " + os.getenv('HOME') + '/raynet/results/' + exp_nickname + "/" + short_params_str + f"/run{run}" + "/"  + protocol_nickname + "/csvs/", shell=True).communicate(timeout=40)
                subprocess.Popen("mkdir -p " + os.getenv('HOME') + '/raynet/results/' + exp_nickname + "/" + short_params_str + f"/run{run}" + "/"  + protocol_nickname + "/csvs/" + str(modName), shell=True).communicate(timeout=40)
                csv_path = os.getenv('HOME') + '/raynet/results/'+ exp_nickname + "/" + short_params_str + f"/run{run}" + "/"  + protocol_nickname + "/csvs/" + str(modName) + "/" + vec + '.csv'
                final_list.to_csv(csv_path, index=False)
                extracted = True
                if (do_dumb_plots): 
                    dumb_plot(csv_path, output_name=vec)
    
    # Parse the associated scenario.xml (if applicable) and output changed values to csvs 
    scenario_file = filepath.rsplit("/", 1)[0]  # ini_variants directory
    scenario_file += f"/scenario.xml_{params_str}"

    print(scenario_file)

    if not os.path.exists(scenario_file):
        print("No scenario.xml found, continuing to next experiment")
    else:
        print(f"Parsing scenario.xml... ({scenario_file})")

        tree = ET.parse(scenario_file)
        root = tree.getroot()

        param_series = {}

        for at_block in root.findall("at"):
            time_str = at_block.get("t")
            time_val = float(time_str.replace("s", ""))

            for param in at_block.findall("set-channel-param"):
                par_name = param.get("par")    
                param_val = parse_numeric(param.get("value"))

                if par_name not in param_series:
                    param_series[par_name] = []

                param_series[par_name].append((time_val, param_val))

        # Output one CSV per parameter
        for par_name, series in param_series.items():
            if not series:
                continue

            times, values = zip(*series)
            df = pd.DataFrame({
                "time": times,
                par_name: values
            })

            # Create directory: .../scenario/
            base_dir = (
                os.getenv('HOME')
                + '/raynet/results/'
                + exp_nickname + "/"
                + short_params_str + f"/run{run}/"
                + protocol_nickname + "/csvs/scenario/"
            )

            subprocess.Popen(f"mkdir -p {base_dir}", shell=True).communicate(timeout=40)

            csv_path = base_dir + f"{par_name}.csv"
            df.to_csv(csv_path, index=False)

            print(f"Saved scenario CSV: {csv_path}")
    termTime.sleep(1)

def run_experiments(experiments_dict, create_output_csv=True):
    """
    Runs the given experiment for each protocol listed, using their respective eval runner scripts
    - This is admittedly very messy and was developed in a time crunch for the dissertation, but it gets the job done and can be cleaned up later if needed
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
    
        # Generate a .ini file for each unique param combination and run
        for params in unique_param_combinations:
            for run in range(1, experiments_dict[experiment_name]["meta"]["runs"] + 1):
                with open(original_ini_file, 'r') as fin:
                    ini_string = fin.read()
                ini_string = ini_string.replace("HOME",  os.getenv('HOME'))
                ini_string = ini_string.replace("START_TIME_PLACEHOLDER",  str(random.uniform(0, 0.2)))
                
                params_suffix = ""
                for param, value in params.items():
                    if experiment_name == "responsiveness":
                        old_value = value
                        params_suffix = params_suffix + f"_{old_value}-{param}"
                        
                        # Create scenario.xml variant
                        with open(f"{original_ini_file.rsplit("/", 1)[0]}/scenario.xml") as scenario_file:
                            scenario_string = scenario_file.read()
                        bw_min = experiments_dict[experiment_name]["meta"]["bw_range"][0]
                        bw_max = experiments_dict[experiment_name]["meta"]["bw_range"][1]
                        initial_bw = int((bw_min+bw_max)/2.0)
                        
                        scenario_string = re.sub(
                            r"INITIAL_BW_PLACEHOLDER",
                            lambda _: f"{initial_bw}Mbps",
                            scenario_string
                        )
                        
                        scenario_string = re.sub(
                            r"BW_PLACEHOLDER",
                            lambda _: f"{random.uniform(bw_min, bw_max)}Mbps",
                            scenario_string
                        )
                        
                        rtt_min = experiments_dict[experiment_name]["meta"]["rtt_range"][0]/2
                        rtt_max = experiments_dict[experiment_name]["meta"]["rtt_range"][1]/2
                        initial_delay = int((rtt_min+rtt_max)/2.0)

                        scenario_string = re.sub(
                            r"INITIAL_DELAY_PLACEHOLDER",
                            lambda _: f"{initial_delay}ms",
                            scenario_string
                        )
                        
                        delay_done = False
                        while not delay_done:
                            delay_value = f"{int(random.uniform(rtt_min, rtt_max))}ms"
                            scenario_string = scenario_string.replace("DELAY_PLACEHOLDER", delay_value, 2)
                            if "DELAY_PLACEHOLDER" not in scenario_string:
                                delay_done=True
                        # scenario_string = re.sub(
                        #     r"DELAY_PLACEHOLDER",
                        #     lambda _: f"{random.uniform(rtt_min, rtt_max)}ms",
                        #     scenario_string,
                        #     count=2
                        # )
                        modified_scenario_file = f"{ini_variants_base.rsplit("/", 1)[0]}/scenario.xml_" + params_suffix + f"_RUN{run}"
                        with open(modified_scenario_file, 'w') as fout:
                            fout.write(scenario_string)
                        
                        # Convert qmult to BDP in bytes (if applicable)
                        if param == "QSIZE" and "bdp" in value:
                            value = f"{int(initial_bw * 1000 * initial_delay * 2 * parse_numeric(value))}b"
                        print(f"using {value}")
                        ini_string = ini_string.replace("BW_PLACEHOLDER", f"{int(initial_bw)}Mbps")
                        ini_string = ini_string.replace("DELAY_PLACEHOLDER", f"{int(initial_delay)}ms")
                        ini_string = ini_string.replace(param + "_PLACEHOLDER",  value)
                        ini_string = ini_string.replace("SCENARIO_PLACEHOLDER", modified_scenario_file)
                    else:
                        # Convert qmult to BDP in bytes (if applicable)
                        old_value = value
                        if param == "QSIZE" and "bdp" in value:
                            value = f"{int(parse_numeric(params["BANDWIDTH"]) * 1000 * parse_numeric(params["DELAY"]) * parse_numeric(value))}b"
                            print(f"{old_value} changed to {value}")
                        # Convert two-way delay (RTT) to one-way delay (if applicable)
                        if param == "DELAY":
                            value = f"{parse_numeric(value)/2.0}ms"
                            print(f"{old_value} changed to {value}")
                        print(f"using {value}")
                        ini_string = ini_string.replace(param + "_PLACEHOLDER",  value)
                        params_suffix = params_suffix + f"_{old_value}-{param}"
                modified_ini_file = ini_variants_base + params_suffix + f"_RUN{run}"
                with open(modified_ini_file, 'w') as fout:
                    fout.write(ini_string)
        
        # Perform all generated experiment files for each protocol
        for protocol_name in experiments_dict[experiment_name]["protocols"]:
            build_str = "ORCA" if protocol_name == "Cubic" else protocol_name.upper()
            subprocess.Popen(f"source ~/omnetpp/setenv && cd ~/raynet && ./build.sh -f {build_str}", shell=True, executable="/bin/bash").communicate(timeout=40)
            for params in unique_param_combinations: # Looping through a second time separately so I don't have to generate all the files multiple times
                for run in range(1, experiments_dict[experiment_name]["meta"]["runs"] + 1):
                    # Run the experiment for this protocol and param combo
                    params_suffix = ""          # used to find the exp.ini and inform column names
                    short_params_suffix = ""    # used to name results directory
                    for param, value in params.items():
                        params_suffix = params_suffix + f"_{value}-{param}"
                        short_params_suffix = short_params_suffix + f"_{value}-{param}"
                    params_suffix += f"_RUN{run}"
                    modified_ini_file = ini_variants_base + params_suffix
                    protocol_runner_path = runner_paths[protocol_name]
                    print(f"\t ---------- {protocol_name} running experiment: {experiment_name + params_suffix} ----------")
                    os.system(f"{python} {protocol_runner_path} {modified_ini_file}") # Finally runs the exp
                    
                    # Generate output.csv and individual vector csvs for all tracked vectors for this exp/protocol/params combo
                    exp_results_dir = os.getenv('HOME') + f"/raynet/_experiments/{experiment_name}/ini_variants/results"
                    generate_exp_csvs(exp_results_dir, protocol_name, params_str=params_suffix, short_params_str=short_params_suffix, run=run)
        
"""
Automation script for running many experiments with different parameter combinations and protocols
- Each param in the "params" field will be substituted into the assoociated .ini file (e.g. BANDWIDTH_PLACEHOLDER)
- All param combos will be automatically explored via itertools.product, and each combo will generate a unique config file
- The "meta" section contains other useful information that will not be used for replacement or naming or the itertools product, but can be used for other experiment-specific stuff
"""
if __name__ == "__main__":
    random.seed(47901741)
    experiments_to_run = {
        "responsiveness": {
            "protocols": ["Orca", "Cubic", "CleanSlate"],
            "params": {
                "QSIZE": ["1bdp"], # Based on the average BDP of the ranges given
                },
            "meta": {
                "runs" : 100,
                "bw_range" : (10, 20),
                "rtt_range" : (10, 100),
                }
            },
        "competing-flows": {
            "protocols": ["Cubic", "Orca", "CleanSlate"],
            "params": {
                "BANDWIDTH" : ["10Mbps"],
                "DELAY"     : ["10ms", "20ms", "30ms", "40ms", "50ms", "60ms", "70ms", "80ms","90ms", "100ms"],    
                "QSIZE": [".2bdp", "1bdp", "4bdp"],
                },
           "meta": {
               "runs" : 10,
               }
            },
        "single-flow": {
            "protocols": ["Cubic", "Orca", "CleanSlate"],
            "params": {
                "BANDWIDTH" : ["10Mbps"],
                "DELAY"     : ["10ms", "20ms", "30ms", "40ms", "50ms", "60ms", "70ms", "80ms","90ms", "100ms"],    
                "QSIZE": [".2bdp", "1bdp", "4bdp"],
                },
           "meta": {
               "runs" : 10,
               }
            },
    }
    
    run_experiments(experiments_to_run)