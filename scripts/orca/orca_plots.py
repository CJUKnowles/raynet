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
    Naively plots the given csv and exports in the same directory.
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

if __name__ == "__main__":
    filePath = os.getenv('HOME') + "/raynet/_experiments/experiment1/results/output.csv"
    exp = "Dumbbell"
    protocol = "Orca"
    argNum = 0
    vectorsToExtract = ["throughput", "srtt", "pacerate", "intervalDuration", "cwnd", "action"]
    extracted = False
    
    # for arg in sys.argv[1:]:
    #     if(argNum == 0):
    #         filePath = str(arg)
    #     elif(argNum == 1):
    #         exp = str(arg)
    #     elif(argNum == 2):
    #         protocol = str(arg) #Protocol Name
    #     elif(argNum == 3):
    #         run = int(arg) #Run
    #     argNum = argNum + 1
    
    # Generate index file (.vci) and results (.csv)
    results_dir = filePath.rsplit("/", 1)[0]
    vec_file = results_dir + f"/{protocol}-#0.vec"
    cmd = f"""
                source ~/omnetpp/setenv &&
                opp_scavetool i "{vec_file}" &&
                opp_scavetool export "{vec_file}" -F CSV-R -o "{results_dir}/output.csv"
            """
    subprocess.Popen(cmd, shell=True, executable="/bin/bash").communicate(timeout=40)
    
    extracted_paths = []
    rawResults = getResults(filePath)
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
                subprocess.Popen("mkdir -p " + os.getenv('HOME') + '/raynet/results/' + exp + "/csvs", shell=True).communicate(timeout=40) 
                subprocess.Popen("mkdir -p " + os.getenv('HOME') + '/raynet/results/' + exp + "/csvs/" + protocol, shell=True).communicate(timeout=40)
                subprocess.Popen("mkdir -p " + os.getenv('HOME') + '/raynet/results/' + exp + "/csvs/" + protocol + "/" + str(modName), shell=True).communicate(timeout=40)
                csv_path = os.getenv('HOME') + '/raynet/results/'+ exp +'/csvs/' + protocol + "/" + str(modName) + "/" + vec + '.csv'
                finallist.to_csv(csv_path, index=False)
                dumb_plot(csv_path, output_name=vec)
                extracted = True
    termTime.sleep(1)
    """
        General-0-20260316-16:14:24-3659738,vector,OrcaNet.server[0].tcp.conn-23,throughput:vector,,,"1.002590405333 2.002590405333 3.002590405333 4.002590405333 5.002590405333 6.002590405333 7.002590405333 8.002590405333 9.002590405333 10.002590405333 11.002590405333 12.002590405333 13.002590405333 14.002590405333 15.002590405333 16.002590405333 17.002590405333 18.002590405333 19.002590405333 20.002590405333 21.002590405333 22.002590405333 23.002590405333 24.002590405333 25.002590405333 26.002590405333 27.002590405333 28.002590405333 29.002590405333 30.002590405333 31.002590405333 32.002590405333 33.002590405333 34.002590405333 35.002590405333 36.002590405333 37.002590405333 38.002590405333 39.002590405333 40.002590405333 41.002590405333 42.002590405333 43.002590405333 44.002590405333 45.002590405333 46.002590405333 47.002590405333 48.002590405333 49.002590405333 50.002590405333 51.002590405333 52.002590405333 53.002590405333 54.002590405333 55.002590405333 56.002590405333 57.002590405333 58.002590405333 59.002590405333 60.002590405333 61.002590405333 62.002590405333 63.002590405333 64.002590405333 65.002590405333 66.002590405333 67.002590405333 68.002590405333 69.002590405333 70.002590405333 71.002590405333 72.002590405333 73.002590405333 74.002590405333 75.002590405333 76.002590405333 77.002590405333 78.002590405333 79.002590405333 80.002590405333 81.002590405333 82.002590405333 83.002590405333 84.002590405333 85.002590405333 86.002590405333 87.002590405333 88.002590405333 89.002590405333 90.002590405333 91.002590405333 92.002590405333 93.002590405333 94.002590405333 95.002590405333 96.002590405333 97.002590405333 98.002590405333 99.002590405333 100.002590405333 101.002590405333 102.002590405333 103.002590405333 104.002590405333 105.002590405333 106.002590405333 107.002590405333 108.002590405333 109.002590405333 110.002590405333 111.002590405333 112.002590405333 113.002590405333 114.002590405333 115.002590405333 116.002590405333 117.002590405333","2196736 2069759 1985280 2002176 2196480 1782528 2306304 1892351 1790976 2264064 1976832 1858560 1968384 2128896 1867008 1867008 2061312 2035968 1875456 1892352 1892352 2112000 1993728 1900800 2027520 1993728 1993728 2002176 1867008 1951488 2019072 1900799 1892352 1976832 1993728 1985280 1976832 1959936 1934592 1943040 2052864 1968384 1875456 1976832 1976832 1926144 1883904 1968384 1968384 1934592 1976832 1993728 1951488 1900800 2069760 1993728 1917696 1968384 1926144 1951488 1900800 1951488 1943040 1926144 1959936 2010624 1951488 1892352 1976832 2027520 1883904 1976832 1926144 1934592 1875456 1968384 1934592 1959936 2019072 1993728 1943040 1900800 2010624 1993728 1926144 1951488 1917696 1951488 1926144 1926144 1917696 1926144 1926144 1985280 1968384 1943040 1959936 2010624 1968384 1909248 1909248 1959936 1959936 1900800 1892352 1934592 2010624 2002176 1985280 1926144 2010624 2027520 1968384 1909248 1926144 1959936 1968384"
        General-0-20260316-16:14:24-3659738,attr,OrcaNet.server[0].tcp.conn-23,throughput:vector,interpolationmode,sample-hold,,
        General-0-20260316-16:14:24-3659738,attr,OrcaNet.server[0].tcp.conn-23,throughput:vector,recordingmode,vector,,
    """