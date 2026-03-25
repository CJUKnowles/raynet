import sys, os
from ray.runtime_env import RuntimeEnv
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from ray.tune.registry import register_env
from ray.rllib.callbacks.callbacks import RLlibCallback
import pprint
import ray
from ray import tune
from ray.tune import Tuner
from ray.air import CheckpointConfig
import random
import math
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.algorithms.sac.sac import SACConfig
from ray.rllib.algorithms.sac.sac import SAC
import os
import time
from random import randint
from ray.tune.analysis import ExperimentAnalysis
import GPUtil
from collections import deque

class OmnetGymApiEnv(gym.Env):
    def __init__(self,env_config):
        """
        Initialize the training environment configuration
        - This mostly involves setting spcaes (bounds, shapes, types) for actions and observations.
        - These bounds are needed for RL algorithms provided by RLlib- They limit the problem space and are also used for normalization.
        """
        sys.path.insert(0, os.path.join(os.getenv('HOME'), "raynet", "build"))
        from omnetbind import OmnetGymApi
        self.runner = OmnetGymApi()
        
        self.env_config = env_config
        self.step_count = 0 # just for debugging
        self.random_seed = os.getpid() # Ensures each ray worker generates different parameters
        random.seed(self.random_seed)
        # Initialize env parameters to some reasonable defaults (these should be quickly overwritten in reset())
        self.stacking = self.env_config["stacking"]

        self.has_reset = False

        # Define the action space (possible values for actions)
        self.action_space = spaces.Box(low=-2, high=2, shape=(1,), dtype=np.float32) # Orca: A float value from -2.0 to 2.0. Will be used to alter cwnd via (cwnd = 2^action * cwnd).

        # Define the observation space (expected values/types for each observation feature)
        self.obs_min = np.tile(np.array(
                     [0,                            # Throughput
                      0,                            # Pacerate
                      0,                            # Lossrate
                      0,                            # number of acks
                      0,                            # Interval duration
                      0,                            # srtt
                      0                             # Delay metric
                      ], dtype=np.float32), self.stacking)
        self.obs_max = np.tile(np.array(
                     [1,                            # Throughput
                      10,                           # Pacerate
                      10,                           # Lossrate
                      10,                           # Number of ACKs
                      1,                            # Interval duration
                      1,                            # srtt
                      1,                            # Delay metric
                      ], dtype=np.float32), self.stacking)
        self.observation_space = spaces.Box(
            low=self.obs_min, 
            high=self.obs_max, 
            dtype=np.float32) # A 4-dimensional array, each feature is a float value with its own bounds
        
        # Create empty observation history deque
        self.num_observations = 7
        self.obs_history = deque(np.zeros(self.stacking*self.num_observations),maxlen=self.stacking*self.num_observations)
        
       
    def reset(self, *, seed=None, options=None):
        # Reset the observation history to empty
        self.obs_history = deque(np.zeros(self.stacking*self.num_observations),maxlen=self.stacking*self.num_observations)
        
        # Dynamically generate new simulation config
        original_ini_file = self.env_config["iniPath"]
        ini_variants_base = f"{self.env_config["iniPath"].rsplit("/", 1)[0]}/ini_variants/{self.env_config["iniPath"].rsplit("/", 1)[1]}"
        with open(original_ini_file, 'r') as fin:
            ini_string = fin.read()
        ini_string = ini_string.replace("HOME",  os.getenv('HOME'))
        # TODO: Include these strings in the .ini somewhere that actually makes them alter the experiment
        with open(ini_variants_base + f".worker{os.getpid()}", 'w') as fout:
            fout.write(ini_string)
        
        # Start a new simulation runner on the modified ini file
        self.runner.initialise(ini_variants_base + f".worker{os.getpid()}", "Orca")
        obs = self.runner.reset()
        
        # Pull the initial observation and store return it to the trainer
        obs = obs['Orca']
        for i in range(self.stacking):
            self.obs_history.extend(obs)    # Reset only - fill the obs_history with copies of first obs instead of 0's
        self.obs_history.extend(obs)
        return_obs_history = np.asarray(list(self.obs_history),dtype=np.float32)
        return  return_obs_history, {}

    def step(self, actions):
        # Forward the action to our agent
        actions = actions.item() # TODO: Make sure this is right. Your types and shapes are a bit sketchy atm
        action = {'Orca': actions}
        obs, rewards, terminateds, info_ = self.runner.step(action)
        self.obs_history.extend(obs['Orca'])
        
        # Extract obs/rewards from our agent
        obs = np.asarray(list(obs['Orca']), dtype=np.float32)    # also formats the RLAgent's obs so RLlib can understand it
        return_obs_history = np.asarray(list(self.obs_history),dtype=np.float32)
        #TODO: Append this to self.obs_history and return that instead. 
        reward = rewards['Orca']                               # Get the reward our RLAgent is reporting
        sim_truncated = False
        
        # Check if this training episode is complete
        if terminateds['Orca']:      # TERMINATED - The RLAgent has reported itself as done (within the context of the MDP.) End the simulation.
            print(terminateds)
            self.runner.shutdown()
            self.runner.cleanup()
        if info_['simDone']:            # TRUNCATED - Environment/simulation has finished before the agent reported as done (usually a timelimit in the .ini)
            sim_truncated = True
        
        # Debug
        printFreq = 1
        if self.step_count % printFreq == -1:
            print("-")
            print(f"{printFreq} step(s) completed (Agent total: {self.step_count}):")
            print("\tObservations:")
            print(f"\t\tThroughput: {obs[0]:.2f}%             \t\t(Normalized, per interval)")
            print(f"\t\tPacing Rate: {obs[1]:.2f}%        \t\t(Normalized, per interval)")
            print(f"\t\tLoss Rate: {obs[2]:.2f}%          \t\t(Normalized, per interval)")
            print(f"\t\tACKs: {obs[3]:.2f}x              \t\t(Multiplier of cwnd, per interval)") #? Identical to goodput(throughput) if normalized. 
            print(f"\t\tInterval time: {obs[4]:.2f}s      \t\t(Raw, per interval)") #? Identical to delay if normalized?
            print(f"\t\tSRTT: {obs[5]:.2f}%                   \t\t(Normalized, current)") #? Basically same as delay? slightly longer time horizon
            print(f"\t\tDelay: {obs[6]:.2f}%                    \t\t(Log, current)") #? Maybe normalize?
            
            print(f"\tRewards:")
            print(f"\t\tREWARD: {reward:.5f}                  \t(Raw, per interval)")
        self.step_count += 1
        
        # OBS, REWARD, IS_TERMINATED, IS_TRUNCATED, EXTRA_INFO
        return  return_obs_history, reward, terminateds['Orca'], sim_truncated, {}
        
# Generates the OmnetGymApiEnv for the calling ray worker
def omnetgymapienv_creator(env_config):
    return OmnetGymApiEnv(env_config)  # return an env instance

register_env("OmnetGymApiEnv", omnetgymapienv_creator)

if __name__ == '__main__':
    env_name = "Orca-inference"
    register_env(env_name, omnetgymapienv_creator)
    
    load_from_checkpoint = True
    checkpoint_load_dir = os.getenv('HOME') + "/ray_results/Orca-1.2/SAC_Orca-1.2_2026-03-23_19-08-55zoflfxd9/checkpoints/checkpoint_51"
    env_config = {"iniPath": sys.argv[1],
                  "stacking": 10}
    
    ray.init(local_mode=True)
    config = (
            SACConfig()
            # .resources(num_gpus=len(gpus), num_gpus_per_learner_worker=1)
            .env_runners(explore=False) #, rollout_fragment_length=1000
            .environment(env_name, env_config=env_config) # "OmnetGymApiEnv
            )
    algo = config.build()
    
    # Convert betas? (solution found online, fixes a crash when loading a checkpoint)
    def betas_tensor_to_float(learner):
        for param_grp_key in learner._optimizer_parameters.keys():
            param_grp = param_grp_key.param_groups[0]
            param_grp["betas"] = tuple(beta.item() for beta in param_grp["betas"])
    if (load_from_checkpoint):
        algo.restore(checkpoint_load_dir)
        algo.learner_group.foreach_learner(betas_tensor_to_float)
    
    # Main loop!
    steps = 0
    check_in_freq = 100
    while True:
        result = algo.training_step()
        if steps % check_in_freq == 0:
            print(f"Performing step: {steps}")
        steps += 1