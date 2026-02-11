import itertools
import os

if __name__ == "__main__":
    
    #ENVS = ["OmnetGymApiEnv"]
    ENVS = ["OmnetGymApiEnv"]
    WORKERS = [15] # 16 is stable. 32+ has issues, 64+ crashes my laptop lol
    SEEDS =  [6142140] 
    
    # For each parameter combo, change to the training script directory and run the script in a new process from there.
    for params in itertools.product(ENVS, WORKERS, SEEDS):
        os.chdir(f"{os.getenv('HOME')}/raynet/scripts/orca/")
        os.system(f"python3 orca_training.py {params[0]} {params[1]} {params[2]}")
