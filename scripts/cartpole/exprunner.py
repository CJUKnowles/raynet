import itertools
import os

if __name__ == "__main__":
    
    #ENVS = ["OmnetGymApiEnv"]
    ENVS = ["CartPole-v1"]
    WORKERS = [1,2,4,8,16]
    SEEDS =  [42]
    
    # For each parameter combo, change to the cartpole directory and run the script in a new process from there.
    for params in itertools.product(ENVS, WORKERS, SEEDS):
        if params[0] != 'ns3-v0':
            os.chdir(f"{os.getenv('HOME')}/raynet/scripts/cartpole")
            os.system(f"python3 cartpoleexp.py {params[0]} {params[1]} {params[2]}")
        else:
            os.chdir(f"{os.getenv('HOME')}/ns-allinone-3.38/ns-3.38/contrib/opengym/examples/cartpole")
            os.system(f"python3 cartpoleexp.py {params[0]} {params[1]} {params[2]}")
