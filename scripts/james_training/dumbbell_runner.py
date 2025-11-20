import itertools
import os

if __name__ == "__main__":
    
    #ENVS = ["OmnetGymApiEnv"]
    ENVS = ["OmnetGymApiEnv"]
    WORKERS = [8] # 16 is healthy. 32+ has issues, 64+ crashes my laptop lol
    SEEDS =  [61420]
    
    # For each parameter combo, change to the cartpole directory and run the script in a new process from there.
    for params in itertools.product(ENVS, WORKERS, SEEDS):
        os.chdir(f"{os.getenv('HOME')}/raynet/scripts/james_training")
        os.system(f"python3 dumbbell_experiment.py {params[0]} {params[1]} {params[2]}")
