import sys, os
from ray.runtime_env import RuntimeEnv
from build.omnetbind import OmnetGymApi
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
import eval_utils
from collections import deque

class OmnetGymApiEnv(gym.Env):
    def __init__(self,env_config):
        """
        Initialize the training environment configuration
        - This mostly involves setting spcaes (bounds, shapes, types) for actions and observations.
        - These bounds are needed for RL algorithms provided by RLlib- They limit the problem space and are also used for normalization.
        """
        #self.spec = gym.envs.registration.EnvSpec(id="OmnetppEnv", entry_point=self.__init__,max_episode_steps=400)
        self.runner = OmnetGymApi()
        self.env_config = env_config
        self.step_count = 0 # just for debugging
        self.random_seed = os.getpid() # Ensures each ray worker generates different parameters
        random.seed(self.random_seed)
        # Initialize env parameters to some reasonable defaults (these should be quickly overwritten in reset())
        self.bw = self.env_config["bottleneck_bw_range"][0]
        self.base_rtt = self.env_config["minimum_rtt_range"][1]
        self.buffer_size = self.env_config["bottleneck_buffer_range"][0]
        self.max_steps_range = self.env_config["max_steps_range"][1]

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
                      ], dtype=np.float32), 10)
        self.obs_max = np.tile(np.array(
                     [1,                            # Throughput
                      10,                           # Pacerate
                      10,                           # Lossrate
                      10,                           # Number of ACKs
                      1,                            # Interval duration
                      1,                            # srtt
                      1,                            # Delay metric
                      ], dtype=np.float32), 10)
        self.observation_space = spaces.Box(
            low=self.obs_min, 
            high=self.obs_max, 
            dtype=np.float32) # A 4-dimensional array, each feature is a float value with its own bounds
        
        # Create empty observation history deque
        self.obs_history = deque(np.zeros(len(self.obs_min)),maxlen=len(self.obs_min))
        
       
    def reset(self, *, seed=None, options=None):
        # Reset the observation history to empty
        self.obs_history = deque(np.zeros(len(self.obs_min)),maxlen=len(self.obs_min))
        
        # Grab environment parameter ranges
        bottleneck_bw_range = self.env_config["bottleneck_bw_range"]
        base_rtt_range = self.env_config["minimum_rtt_range"]
        bottleneck_buffer_range = self.env_config["bottleneck_buffer_range"]
        max_steps_range = self.env_config["max_steps_range"]
        
        # Randomize environment parameters. Save them for obs normalization later.
        self.bw = round(np.random.uniform(low=bottleneck_bw_range[0], high=bottleneck_bw_range[1]))
        self.base_rtt = round(np.random.uniform(low=base_rtt_range[0], high=base_rtt_range[1]),2)
        self.buffer_size = round(np.random.uniform(low=bottleneck_buffer_range[0], high=bottleneck_buffer_range[1]))
        self.max_steps = round(np.random.uniform(low=max_steps_range[0], high=max_steps_range[1]))

        # print("ORCA_BOTTLENECK_BW: ", f"{self.bw}Mbps")
        # print("ORCA_BASE_RTT: ", f"{self.base_rtt}ms")
        # print("ORCA_BOTTLENECK_BUFFER_SIZE: ", f"{self.buffer_size}b")
        # print("MAX_RL_STEPS: ", f"{self.max_steps}")
        
        # Modify the base config .ini with a proper home directory and the random environment parameters
        original_ini_file = self.env_config["iniPath"]
        with open(original_ini_file, 'r') as fin:
            ini_string = fin.read()
        ini_string = ini_string.replace("HOME",  os.getenv('HOME'))
        ini_string = ini_string.replace("ORCA_BOTTLENECK_BW", f"{self.bw}Mbps")
        ini_string = ini_string.replace("ORCA_BASE_RTT", f"{self.base_rtt/2.0}ms")  # Delay goes both ways, divide by two
        ini_string = ini_string.replace("ORCA_BOTTLENECK_BUFFER_SIZE", f"{self.buffer_size}b")
        ini_string = ini_string.replace("MAX_RL_STEPS", f"{self.max_steps}")
        # TODO: Include these strings in the .ini somewhere that actually makes them alter the experiment
        with open(original_ini_file + f".worker{os.getpid()}", 'w') as fout:
            fout.write(ini_string)
        
        # Start a new simulation runner on the modified ini file
        self.runner.initialise(original_ini_file + f".worker{os.getpid()}")
        obs = self.runner.reset()
        obs = obs['Orca']
        self.obs_history.extend(obs)
        return_obs_history = np.asarray(list(self.obs_history),dtype=np.float32)
        #TODO: return history of observations, not just this observation. Compile a self.obs_history and return that instead.
        return  return_obs_history, {}

    def step(self, actions):
        """
        Receive an action from the policy (provided by RLlib), forward it to the RayNet RLAgent, and return the result to RLlib for further training.
        - This experiment(?) script is not responsible for determining the action/policy, it is just a middleman between RLlib and the RayNet RLAgent.
        - Actions/observations exist in a dictionary to support multi-agent environments.
        - This experiment only support single-agent environments, so observations/rewards are immediately extracted from the dictionary
        """
        # Forward the action (provided by RLlib) to OMNeT++ (and eventually our RLAgent Orca), and retrieve the RLAgent's reported result
        actions = actions.item() # TODO: Make sure this is right. Your types and shapes are a bit sketchy atm
        action = {'Orca': actions}               
        obs, rewards, terminateds, info_ = self.runner.step(action)
        self.obs_history.extend(obs['Orca'])
        
        # Extra the relevant obs/rewards from the environment info (only the info relevent to our single-agent)
        obs = np.asarray(list(obs['Orca']), dtype=np.float32)    # also formats the RLAgent's obs so RLlib can understand it
        return_obs_history = np.asarray(list(self.obs_history),dtype=np.float32)
        #TODO: Append this to self.obs_history and return that instead. 
        reward = rewards['Orca']                               # Get the reward our RLAgent is reporting
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
        if terminateds['Orca']:      # TERMINATED - The RLAgent has reported itself as done (within the context of the MDP.) End the simulation.
            self.runner.shutdown()
            self.runner.cleanup()
        if info_['simDone']:            # TRUNCATED - Environment/simulation has finished before the agent reported as done (usually a timelimit in the .ini)
            sim_truncated = True
        
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
    env_name = "Orca-1.1"
    register_env(env_name, omnetgymapienv_creator)
    num_workers = 13 # Must be >= 1. A value of 0 will spawn a single worker that does not reset if issues occur. 1+ allows resets.
    seed = 91456211
    # bottleneck_bandwidth_range = (6, 192)            # Orca: 6Mbps-192Mbps
    # minimum_rtt_range = (4, 400)                     # Orca: 4ms-400ms
    # bottleneck_buffer_range = (3000, 96000000)       # Orca: 3KB-96MB, expressed in terms of bits
    max_steps_range = (5000, 5000)                   # Custom: Randomize ending time slightly so threads desync, to make log outputs less sparse
    bottleneck_bandwidth_range = (6, 6)            
    minimum_rtt_range = (5, 5)
    bottleneck_buffer_range = (5280000, 5280000) 
    load_from_checkpoint = True
    checkpoint_load_dir = os.getenv('HOME') + "/ray_results/Orca-1.0/SAC_Orca_2026-03-13_14-42-05b6jzob3t/checkpoints/checkpoint_107"
    steps_to_train = 5000000
    env_config = {"iniPath": os.getenv('HOME') + "/raynet/configs/orca/orca.ini",
                  "bottleneck_bw_range": bottleneck_bandwidth_range,
                  "minimum_rtt_range": minimum_rtt_range,
                  "bottleneck_buffer_range": bottleneck_buffer_range,
                  "max_steps_range": max_steps_range}
    random.seed(seed)
    np.random.seed(seed)
    gpus = GPUtil.getGPUs()
    print("GPUs Available:", gpus)
    ray.init(num_cpus=16, num_gpus=len(gpus))
    config = (
            SACConfig()
            .resources(num_gpus=len(gpus))
            .env_runners(num_env_runners=num_workers) #, rollout_fragment_length=1000
            .learners(num_learners=1, num_gpus_per_learner=len(gpus), num_cpus_per_learner=1)
            .environment(env_name, env_config=env_config) # "OmnetGymApiEnv
            .training(store_buffer_in_checkpoints=True)
            #.training(optimizer={"foreach": False, "capturable": True})
            ##.evaluation(evaluation_interval=1000, evaluation_duration_unit="timesteps")
            ##.fault_tolerance(restart_failed_sub_environments=True)
            #.training(training_intensity=1000)  # num_steps_sampled_before_learning_starts=0 training_intensity=1000
            # .build_algo()
            )
    algo = config.build()
    
    # Convert betas? (solution found online, fixes a crash when loading a checkpoint)
    def betas_tensor_to_float(learner):
        for param_grp_key in learner._optimizer_parameters.keys():
            param_grp = param_grp_key.param_groups[0]
            param_grp["betas"] = tuple(beta.item() for beta in param_grp["betas"])
    if (load_from_checkpoint):
        #algo.load_checkpoint(os.getenv('HOME') + "/ray_results/JAMESTEST")
        algo.restore(checkpoint_load_dir)
        algo.learner_group.foreach_learner(betas_tensor_to_float)
    
    
    pprint.pprint(algo.config)
    # Main training loop!
    iteration = 0
    checkpoint = 0
    iterations_per_checkpoint = 1000
    while True:
        result = algo.train()   # Perform a single training iteration (many steps, usually shorter than an episode. Changes depending on training parameters.)
        iteration += 1
        print(f"Iteration {iteration} complete")
        if (iteration % iterations_per_checkpoint == 0):
            checkpoint_dir = algo.logdir + f"/checkpoints/checkpoint_{checkpoint}"
            algo.save_checkpoint(checkpoint_dir) # Somehow get the directory from this?
            print(f"Saved checkpoint to {checkpoint_dir}")
            checkpoint += 1

    # old -------------------------------
    #algo = SAC.from_checkpoint(os.getenv('HOME') + "/ray_results/orca/SAC_OmnetGymApiEnv_8fe1c_00000_0_2026-03-04_01-57-55/checkpoint_000021")
    # ray.tune.run(
    #     "SAC",
    #     name="orca",
    #     stop={"num_env_steps_sampled_lifetime": steps_to_train},
    #     config=config,
    #     restore=os.getenv('HOME') + "/ray_results/orca/SAC_OmnetGymApiEnv_8fe1c_00000_0_2026-03-04_01-57-55/checkpoint_000021",
    #     #resume=True,
    #     checkpoint_config=CheckpointConfig(checkpoint_frequency=1000, checkpoint_at_end=True),

    # )
    
    # trials_dfs = exp.trial_dataframes # Returns a dict of dfs. Each df represents a trial, and contains rows of training iterations. Used for time series plots.
    # trials_results = exp.results_df # Returns a df in which each row represents a trial, and contains aggregate/summary information about it. Used for scalar plots.
    # #results = exp.dataframe()
    
    # results_path = exp.experiment_path
    
    # for trial_id, trial_df in trials_dfs.items():
    #     print(f"Creating plot for trial {trial_id}")
    #     eval_utils.plot_experiment_summary(trial_df, exp.experiment_path, f"{trial_id}_time_series.pdf")