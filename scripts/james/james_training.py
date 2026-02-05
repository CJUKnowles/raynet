import sys, os
from ray.runtime_env import RuntimeEnv
from build.omnetbind import OmnetGymApi
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from ray.tune.registry import register_env
import ray
from ray import tune
from ray.tune import Tuner
from ray.air import CheckpointConfig
import random
import math
from ray.rllib.algorithms.ppo.ppo import AlgorithmConfig
from ray.rllib.algorithms.ppo.ppo import PPOConfig
import os
import time
from random import randint
from ray.tune.analysis import ExperimentAnalysis
import GPUtil
import eval_utils

class OmnetGymApiEnv(gym.Env):
    def __init__(self,env_config):
        """
        Initialize the training environment configuration
        - This mostly involves setting spcaes (bounds, shapes, types) for actions and observations.
        - These bounds are needed for RL algorithms provided by RLlib- They limit the problem space and are also used for normalization.
        """
        self.runner = OmnetGymApi()
        self.env_config = env_config
        #self.max_episode_len = 5 # don't think I need this

        # Define the action space (possible values for actions)
        #self.action_space = spaces.Discrete(2)
        self.action_space = spaces.Box(low=0.1, high=5.0, shape=(1,), dtype=np.float64) # A single float value ranging from 0.0 -> 2.0

        # Define the observation space (expected values/types for each observation feature)
        low_bounds = [0, 0, 0, -np.finfo(np.float64).max,]
        high_bounds= [np.finfo(np.float64).max, 10, np.finfo(np.float64).max,  np.finfo(np.float64).max,]
        low_bounds = np.array(low_bounds, dtype=np.float64)
        high_bounds = np.array(high_bounds, dtype=np.float64)
        self.observation_space = spaces.Box(low=low_bounds, high=high_bounds, dtype=np.float64) # A 4-dimensional array, each feature is a float value with its own bounds
       
    def reset(self, *, seed=None, options=None):
        #print("\tRESET BEING CALLED")

        original_ini_file = self.env_config["iniPath"]
        # Replace HOME with absolute paths in the simulation ini file
        with open(original_ini_file, 'r') as fin:
            ini_string = fin.read().replace("HOME",  os.getenv('HOME'))
        with open(original_ini_file + f".worker{os.getpid()}", 'w') as fout:
            fout.write(ini_string)
        # Start a new simulation runner on the modified ini file
        self.runner.initialise(original_ini_file + f".worker{os.getpid()}")
        obs = self.runner.reset()
        obs = np.asarray(list(obs['JamesCC']),dtype=np.float64)
        return  obs, {}

    def step(self, actions):
        """
        Receive an action from the policy (provided by RLlib), forward it to the RayNet RLAgent, and return the result to RLlib for further training.
        - This experiment(?) script is not responsible for determining the action/policy, it is just a middleman between RLlib and the RayNet RLAgent.
        - Actions/observations exist in a dictionary to support multi-agent environments.
        - This experiment only support single-agent environments, so observations/rewards are immediately extracted from the dictionary
        """
        # Forward the action (provided by RLlib) to OMNeT++ (and eventually our RLAgent JamesCC), and retrieve the RLAgent's reported result
        actions = actions.item() # TODO: Make sure this is right. Your types and shapes are a bit sketchy atm
        action = {'JamesCC': actions}               
        obs, rewards, terminateds, info_ = self.runner.step(action)
        # Extra the relevant obs/rewards from the environment info (only the info relevent to our single-agent)
        obs = np.asarray(list(obs['JamesCC']), dtype=np.float64)    # also formats the RLAgent's obs so RLlib can understand it
        reward = round(rewards['JamesCC'], 4)                                 # Get the reward our RLAgent is reporting
        sim_truncated = False
  
        # Debug stuff
        # print("\t\t\tSTEP: ")
        # print("\t\t\tobs: ", obs)
        # print("\t\t\trewards: ", rewards)
        # print("\t\t\tterminateds: ", terminateds)
        # print("\t\t\tinfo_: ", info_)
        if math.isnan(reward):
            print("Warning: NaN reward returned!")
        # Check if this training episode is complete
        if terminateds['JamesCC']:      # TERMINATED - The RLAgent has reported itself as done (within the context of the MDP.) End the simulation.
            self.runner.shutdown()
            self.runner.cleanup()
        if info_['simDone']:            # TRUNCATED - Environment/simulation has finished before the agent reported as done (usually a timelimit in the .ini)
            sim_truncated = True
        sim_truncated=False

        # OBS, REWARD, IS_TERMINATED, IS_TRUNCATED, EXTRA_INFO
        return  obs, reward, terminateds['JamesCC'], sim_truncated, {"test": "this is a test! Can the JamesCC see this info?"}


# Generates the OmnetGymApiEnv for the calling ray worker
def omnetgymapienv_creator(env_config):
    return OmnetGymApiEnv(env_config)  # return an env instance

register_env("OmnetGymApiEnv", omnetgymapienv_creator)

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        #raise Exception("This script expects arguments ENV, NUM_WORKERS, SEED. Please provide arguments or use a runner.py")
        env = "OmnetGymApiEnv"
        num_workers = 1
        seed = 918284
        steps_to_train = 1000
    else:
        env = sys.argv[1]               # OmnetGymApiEnv, CartPole-v1
        num_workers = int(sys.argv[2])  # 1, 2, 4, 8, 16
        seed = int(sys.argv[3])         # any num
        steps_to_train = 1000000
    gpus = GPUtil.getGPUs()
    print("Num GPUs Available:", len(gpus))
    
    random.seed(seed)
    np.random.seed(seed)

    ray.init(num_cpus=64, num_gpus=1)

    env_config = {"iniPath": os.getenv('HOME') + "/raynet/configs/james/james.ini"}

    config = (
            PPOConfig()
            .resources(num_gpus=1)
            .env_runners(num_env_runners=num_workers)
            .learners(num_gpus_per_learner=1)
            .environment(env, env_config=env_config) # "OmnetGymApiEnv
            .training()       
            #.build_algo()
            )

    checkpoint_config = CheckpointConfig(
        
    )
    
    exp:ExperimentAnalysis = ray.tune.run(
        "PPO",
        name="James_training",
        stop={"num_env_steps_sampled_lifetime": steps_to_train},
        config=config
        #TODO: Add some sort of CheckpointFrequency
    )
    
    trials_dfs = exp.trial_dataframes # Returns a dict of dfs. Each df represents a trial, and contains rows of training iterations. Used for time series plots.
    trials_results = exp.results_df # Returns a df in which each row represents a trial, and contains aggregate/summary information about it. Used for scalar plots.
    #results = exp.dataframe()
    
    results_path = exp.experiment_path
    
    for trial_id, trial_df in trials_dfs.items():
        print(f"Creating plot for trial {trial_id}")
        print(trial_df)
        print(trial_df[['env_runners/episode_return_max', 'env_runners/episode_return_min', 'env_runners/episode_return_mean', 'env_runners/num_episodes']])
        eval_utils.plot_experiment_summary(trial_df, exp.experiment_path, f"{trial_id}_time_series.pdf")
    





    # Manual training. Include examples of this too, but the tuner is way easier to use
    
    # algo = (
    #         PPOConfig()
    #         .resources(num_gpus=1)
    #         .env_runners(num_env_runners=num_workers)
    #         .learners(num_gpus_per_learner=1)
    #         .environment(env, env_config=env_config) # "OmnetGymApiEnv
    #         .training()       
    #         .build_algo()
    #         )
    
    # # Run experiments and log progress
    # t_start = time.time()
    # now = time.time()
    # training_iteration_count = 0
    # max_rl_steps = 1000
    # while True:
    #     result = algo.train() # Runs a single training iteration (usually some arbitrary number of steps, like 100)
    #     training_iteration_count += 1
    #     print(f"Training iteration complete! ------------------------------")
    #     print(f"Time elapsed: {(now - t_start)}")
    #     print(f"Training iterations completed so far: {training_iteration_count}")
    #     print(f"RL steps performed so far: ({result['num_env_steps_sampled_lifetime']}/{max_rl_steps})")
    #     print(f"-----------------------------------------------------------\n")
    #     if result['num_env_steps_sampled_lifetime'] >= max_rl_steps:
    #         break
    #     now = time.time()
    # t_complete = time.time()
    # print("Training completed in "+ str(t_complete - t_start) + " seconds")
    # ray.shutdown()
    
    