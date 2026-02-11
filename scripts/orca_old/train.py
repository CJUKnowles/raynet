from build.omnetbind import OmnetGymApi
from nnmodels import KerasBatchNormModel
import gymnasium as gym
from gymnasium import spaces, logger
import numpy as np
import math
from ray.tune.registry import register_env
#from ray.rllib.algorithms.td3 import TD3Config
from ray.rllib.algorithms.sac import SACConfig
import ray
import pandas as pd
import os
from collections import deque
import time
from ray.rllib.models import ModelCatalog


ModelCatalog.register_custom_model("bn_model",KerasBatchNormModel)

# Utility Function
def uniform(low=0, high=1):
    return np.random.uniform(low, high)

# Instance of the OmnetGymAPI environment. Basically defines a single agent.
class OmnetGymApiEnv(gym.Env):

    # Initialise variables, do not start the simulation
    def __init__(self, env_config):
        self.env_config = env_config
        self.stacking = env_config['stacking']
        self.action_space = spaces.Box(low=np.array([-2.0], dtype=np.float32), high=np.array([2.0], dtype=np.float32), dtype=np.float32)
        self.obs_min = np.tile(np.array([-1000000000,  
                                 -1000000000,   
                                 -1000000000,   
                                 -1000000000,
                                 -1000000000,
                                 -1000000000,
                                 -1000000000], dtype=np.float32), self.stacking)

        self.obs_max = np.tile(np.array([10000000000, 
                                 10000000000, 
                                 10000000000, 
                                 10000000000,
                                 10000000000,
                                 10000000000,
                                 10000000000],dtype=np.float32), self.stacking)
        self.currentRecord = None
        self.observation_space = spaces.Box(low=self.obs_min, high=self.obs_max, dtype=np.float32)
        self.runner = OmnetGymApi()
        self.obs = deque(np.zeros(len(self.obs_min)),maxlen=len(self.obs_min))
        self.agentId = None
    
    # Reset (and start) the simulation at a particular ste[p]
    def reset(self, *, seed=None, options=None):
        self.obs = deque(np.zeros(len(self.obs_min)), maxlen=len(self.obs_min)) # initaialize an observation of all zeros

        # Grab network parameters
        linkrate_range = self.env_config["linkrate_range"]
        rtt_range = self.env_config["rtt_range"]
        buffer_range = self.env_config["buffer_range"]

        # Grab random network parameter values from a range (?)
        linkrate = uniform(low=linkrate_range[0], high=linkrate_range[1])
        rtt = uniform(low=rtt_range[0], high=rtt_range[1])/2.0
        buffer = uniform(low=buffer_range[0], high=buffer_range[1])

        # Open and modify the original ini file (reset it)
        original_ini_file = self.env_config["iniPath"]
        with open(original_ini_file, 'r') as fin:
            ini_string = fin.read()
        ini_string = ini_string.replace("DELAY_PLACEOLDER", f'{round(rtt,2)}ms')
        ini_string = ini_string.replace("LINKRATE_PLACEHOLDER", f'{round(linkrate)}Mbps')
        ini_string = ini_string.replace("Q_PLACEHOLDER", str(round(buffer)))
        ini_string = ini_string.replace("HOME",  os.getenv('HOME'))

        # Create a new worker ini file and use it to start a runner
        worker_ini_file = original_ini_file + f".worker{os.getpid()}_{self.env_config.worker_index}"
        with open(worker_ini_file, 'w') as fout:
            fout.write(ini_string)
        self.runner.initialise(worker_ini_file)

        # Reset agent variables based on the reset runner
        obs = self.runner.reset() # Reset the runner itself. This is the same as the reset function this code exists in, but a layer deeper.
        if len(obs.keys()) > 1: # Multiple agents were found. Not supported here.
            print(f"************ ERROR: expected only 1 flow, but {len(obs.keys())} were found.") 
        self.agentId = list(obs.keys())[0]
        obs = obs[self.agentId]
        self.currentRecord = obs
        self.obs.extend(obs)
        obs = np.asarray(list(self.obs),dtype=np.float32)
        return obs, {}

    """
    - Recieves an action from the RL algorithm (SAC, in this case, returns an action from its policy)
    - Tell the agent to take said action after formatting it
    - Get observations and rewards from the agent based on the action taken
    - Format the obs/rewards and return it...?
    """
    def step(self, action):

        # Sanitize the action ------------------------
        action = 2**action                  # The action is 2^(value), some weird orca quirk
        actions = {self.agentId: action}    # Creates a single-entry dictionary mapping an agent to an action. This may be done bcause a dictionary is required for multi-action setups, which doesn't apply here.
        if math.isnan(action):
            # Something broke. Not sure why this doesn't just return
            print("====================================== action passed is nan =========================================")


        # Take the step with the action ------------------ (!!!!)
        """
        Step return values:
            obs - array of observations, indexed by each agent's ID
            rewards - array of reward values, indexed by each agent's ID
            dones - array of true/false done states for each agent, indexed by each agent's ID
            info_ - unknown, come back to this
        """
        obs, rewards, dones, info_= self.runner.step(actions) # tell the runner to step! (rollout worker?) and return observation/reward
        

        # Stop the runner if this agent is done
        if dones[self.agentId]:
             self.runner.shutdown()
             self.runner.cleanup()

        # Handle the reward
        if math.isnan(rewards[self.agentId]):
            print("====================================== reward returned is nan =========================================")
        reward = round(rewards[self.agentId],4)

        # Handle the observation
        if any(np.isnan(np.asarray(obs[self.agentId], dtype=np.float32))):
            print("==================================== == obs returned is nan =========================================")
        obs = obs[self.agentId]                             
        self.currentRecord = obs
        self.obs.extend(obs)                                # Add the observation to the list for this agent(?)
        obs = np.asarray(list(self.obs),dtype=np.float32)   # Grab an array of all observations so far

        # Report this agent as being done if the sim is complete
        if info_['simDone']:
             dones[self.agentId] = True
        
        # Return step info to RLlib for future use/training
        return  obs, reward, dones[self.agentId],False, {}

# Funcion that returns the specified environment
def OmnetGymApienv_creator(env_config):
    return OmnetGymApiEnv(env_config)  # return an env instance


if __name__ == '__main__':
    # Set the config each agent will begin with (?)
    register_env("OmnetGymApiEnv", OmnetGymApienv_creator)
    env_config={"iniPath": os.getenv('HOME') + "/raynet/configs/orca/orcaConfigStatic.ini",
            "stacking": 10,
            "linkrate_range": [64,64],
            "rtt_range": [16, 16],
            "buffer_range": [250, 250],}

    # Define the training algorithm
    algo = (
        SACConfig()
        .env_runners(num_env_runners=1)
        .resources(num_gpus=0)
        .environment("OmnetGymApiEnv", env_config=env_config) # "ns3-v0"
        .build_algo())

    # Log progress
    t_start = time.time()
    now = time.time()
    while True:
        print(f"Total elpsed: {(now - t_start)}")
        result = algo.train()
        print(result['num_env_steps_sampled'])
        if result['num_env_steps_sampled'] >= 1000000:
                break
        now = time.time()
    ray.shutdown()
    # analysis = ray.tune.run(
    #     "TD3", name="orca",stop={"training_iteration": 200000}, config=config, checkpoint_freq=50)