
import sys, os
from ray.runtime_env import RuntimeEnv
import gymnasium as gym
from gymnasium import spaces, logger
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
from ray.rllib.algorithms.sac.sac import SACConfig
import os
import time
from random import randint
from ray.tune.analysis import ExperimentAnalysis
import GPUtil
from collections import deque, defaultdict
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class OmnetGymApiEnv(MultiAgentEnv):
    
    # def get_observation_space(self, agent_id):
    #     if agent_id.startswith("Astrea"):
    #         return spaces.Box(
    #             low=self.obs_min, 
    #             high=self.obs_max,
    #             dtype=np.float32) # A 4-dimensional array, each feature is a float value with its own bounds
    #     else:
    #         raise ValueError(f"bad agent id: {agent_id}!")
    
    # def get_action_space(self, agent_id):
    #     print("HI HI HELLO HI")
    #     if agent_id.startswith("Astrea"):
    #         return gym.spaces.Box(low=np.array(-2.0, dtype=np.float32), high=np.array(2.0, dtype=np.float32), dtype=np.float32)
    #     else:
    #         raise ValueError(f"bad agent id: {agent_id}!")
    
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
        self.stacking = self.env_config["stacking"]
        self.random_seed = os.getpid() # Ensures each ray worker generates different parameters
        random.seed(self.random_seed)
        # Initialize env parameters to some reasonable defaults (these should be quickly overwritten in reset())
        self.bw = self.env_config["bottleneck_bw_range"][0]
        self.base_rtt = self.env_config["minimum_rtt_range"][1]
        self.buffer_size = self.env_config["bottleneck_buffer_range"][0]
        self.max_steps = self.env_config["max_steps_range"][1]
        self.num_flows = self.env_config["num_flows_range"][0]
        self.has_reset = False
        
        self.obs_size = 7   # How many values are in a given obs
        self.obs_min = np.tile(np.array(
                     [0,                            # Throughput
                      0,                            # number of acks
                      0,                            # Interval duration
                      0,                            # srtt
                      0,                             # Delay metric
                      0,
                      0,
                      ], dtype=np.float32), self.stacking)
        self.obs_max = np.tile(np.array(
                    [1,                           # Throughput
                    10000000000,                  # Throughput max
                    10,                           # Latency Ratio
                    1,                            # Min Latency
                    10,                            # relative cwnd
                    1,                            # Loss rate
                    1                             # Inflight
                    ], dtype=np.float32), self.stacking)
        self.agent_dones = {}
        self.agent_trucateds = {}
        
        obs_spaces = {}
        action_spaces = {}
        self.possible_agents = []
        for i in range(self.env_config["num_flows_range"][1]):
            self.possible_agents.append(f"`Astrea`{i}")
            obs_spaces[f"Astrea{i}"] = spaces.Box(low=self.obs_min, high=self.obs_max, dtype=np.float32)
            action_spaces[f"Astrea{i}"] = spaces.Box(low=-1.0, high=1.0, dtype=np.float32)
        
        self.observation_space = spaces.Dict(obs_spaces)
        self.action_space = spaces.Dict(action_spaces)
        
       
    def reset(self, *, seed=None, options=None):
        # Reset the observation history to empty
        self.obs_history = defaultdict(
            lambda: deque(
                [np.zeros(self.obs_size, dtype=np.float32) for _ in range(self.stacking)],
                maxlen=self.stacking
            )
        )
        
        # Grab environment parameter ranges
        bottleneck_bw_range = self.env_config["bottleneck_bw_range"]
        base_rtt_range = self.env_config["minimum_rtt_range"]
        bottleneck_buffer_range = self.env_config["bottleneck_buffer_range"]
        max_steps_range = self.env_config["max_steps_range"]
        num_flows_range = self.env_config["num_flows_range"]
        
        # Randomize environment parameters. Save them for obs normalization later.
        self.bw = round(np.random.uniform(low=bottleneck_bw_range[0], high=bottleneck_bw_range[1]))
        self.base_rtt = round(np.random.uniform(low=base_rtt_range[0], high=base_rtt_range[1]),2)
        self.buffer_size = round(np.random.uniform(low=bottleneck_buffer_range[0], high=bottleneck_buffer_range[1]))
        self.max_steps = round(np.random.uniform(low=max_steps_range[0], high=max_steps_range[1]))
        self.num_flows = round(np.random.uniform(low=num_flows_range[0], high=num_flows_range[1]))

        print("ORCA_BOTTLENECK_BW: ", f"{self.bw}Mbps")
        print("ORCA_BASE_RTT: ", f"{self.base_rtt}ms")
        print("ORCA_BOTTLENECK_BUFFER_SIZE: ", f"{self.buffer_size}b")
        print("MAX_RL_STEPS: ", f"{self.max_steps}")
        
        # Modify the base config .ini with a proper home directory and the random environment parameters
        original_ini_file = self.env_config["iniPath"]
        ini_variants_base = f"{self.env_config["iniPath"].rsplit("/", 1)[0]}/ini_variants/{self.env_config["iniPath"].rsplit("/", 1)[1]}"
        with open(original_ini_file, 'r') as fin:
            ini_string = fin.read()
        ini_string = ini_string.replace("HOME",  os.getenv('HOME'))
        ini_string = ini_string.replace("BOTTLENECK_BW", f"{self.bw}Mbps")
        ini_string = ini_string.replace("BASE_RTT", f"{self.base_rtt/2.0}ms")  # Delay goes both ways, divide by two
        ini_string = ini_string.replace("BOTTLENECK_BUFFER_SIZE", f"{self.buffer_size}b")
        ini_string = ini_string.replace("MAX_RL_STEPS", f"{self.max_steps}")
        ini_string = ini_string.replace("NUM_FLOWS", f"{self.num_flows}")
        # TODO: Include these strings in the .ini somewhere that actually makes them alter the experiment
        with open(ini_variants_base + f".worker{os.getpid()}", 'w') as fout:
            fout.write(ini_string)
        
        # Start a new simulation runner on the modified ini file
        self.runner.initialise(ini_variants_base + f".worker{os.getpid()}", "General")
        obs = self.runner.reset()
        print("Reset obs: ")
        print(obs)
        print("")
        
        # Append the most recent observations to their agents' histories. Store the updated histories in obs and return.
        for agent, agent_obs in obs.items():
            self.obs_history[agent].append(np.asarray(agent_obs, dtype=np.float32))
            obs[agent] = np.concatenate(self.obs_history[agent])
        return  obs, {}

    def step(self, actions):
        """
        Receive an action from the policy (provided by RLlib), forward it to the RayNet RLAgent, and return the result to RLlib for further training.
        - This experiment(?) script is not responsible for determining the action/policy, it is just a middleman between RLlib and the RayNet RLAgent.
        - Actions/observations exist in a dictionary to support multi-agent environments.
        - This experiment only support single-agent environments, so observations/rewards are immediately extracted from the dictionary
        """

        # Convert the policy's action dict(str:np.float32) to dict(str:float) so omnet can use it
        for agent_id, action in actions.items():
            actions[agent_id] = float(np.asarray(action).item())
        obs, rewards, terminateds, info_ = self.runner.step(actions)
        
    
        # Append the most recent observations to their agents' histories. Store the updated histories in obs
        for agent, agent_obs in obs.items():
            self.obs_history[agent].append(np.asarray(agent_obs, dtype=np.float32))
            obs[agent] = np.concatenate(self.obs_history[agent])
        
        if terminateds["__all__"] or info_['simDone']:
            print("Episode complete, shutting down simulation env")
            self.runner.shutdown()
            self.runner.cleanup()
            
        # OBS, REWARD, IS_TERMINATED, IS_TRUNCATED, EXTRA_INFO
        return  obs, rewards, terminateds, terminateds, {}
        
# Generates the OmnetGymApiEnv for the calling ray worker
def omnetgymapienv_creator(env_config):
    env = OmnetGymApiEnv(env_config)
    return env  # return an env instance

if __name__ == '__main__':
    env_name = "Astrea-1.3"
    register_env(env_name, omnetgymapienv_creator)
    seed = 91456211
    
    
    num_workers = 15 # Must be >= 1. A value of 0 will spawn a single worker that does not reset if issues occur. 1+ allows resets.
    # Environment Params
    max_steps_range = (2000, 2000)
    bottleneck_bandwidth_range = (5, 20)
    minimum_rtt_range = (5, 100)
    bottleneck_buffer_range = (25000, 2000000)
    num_flows_range = (2,5)
    
    # num_workers = 2 # Must be >= 1. A value of 0 will spawn a single worker that does not reset if issues occur. 1+ allows resets.
    # # Environment Params
    # max_steps_range = (5000, 5000)
    # bottleneck_bandwidth_range = (10, 10)
    # minimum_rtt_range = (25, 25)
    # bottleneck_buffer_range = (2000000, 2000000)
    # num_flows_range = (2,2)
    
    # Training Params
    load_from_checkpoint = False
    checkpoint_load_dir = os.getenv('HOME') + "/ray_results/SAC_Astrea-1.2_2026-04-14_01-12-53pspebvlh/checkpoints/checkpoint_3"
    stacking = 5
    
    env_config = {"iniPath": os.getenv('HOME') + "/raynet/simlibs/Astrea/src/training/AstreaTraining.ini",
                  "bottleneck_bw_range": bottleneck_bandwidth_range,
                  "minimum_rtt_range": minimum_rtt_range,
                  "bottleneck_buffer_range": bottleneck_buffer_range,
                  "max_steps_range": max_steps_range,
                  "num_flows_range": num_flows_range,
                  "stacking": stacking} # how many observations to keep in an obs_history
    random.seed(seed)
    np.random.seed(seed)
    gpus = GPUtil.getGPUs()
    print("GPUs Available:", gpus)
    
    obs_min = np.tile(np.array(
                     [0,                           
                      0,                           
                      0,                           
                      0,                            
                      0,                            
                      0,
                      0
                      ], dtype=np.float32), stacking)
    obs_max = np.tile(np.array(
                    [1,                           # Throughput
                    10000000000,                  # Throughput max
                    10,                           # Latency Ratio
                    1,                            # Min Latency
                    10,                            # relative cwnd
                    1,                            # Loss rate
                    1                             # Inflight
                    ], dtype=np.float32), stacking)
        
    ray.init(num_cpus=16, num_gpus=len(gpus))
    config = (
            SACConfig()
            .resources(num_gpus=len(gpus), num_gpus_per_learner_worker=1)
            .env_runners(num_env_runners=num_workers, 
                         num_cpus_per_env_runner=1, 
                         num_envs_per_env_runner=1, 
                         explore=True, 
                         # batch_mode="complete_episodes"
                         ) #, rollout_fragment_length=1000
            .environment(env_name, env_config=env_config, disable_env_checking=True) # "OmnetGymApiEnv
            .multi_agent(
                policies={
                    "shared_policy": (
                        None,
                        spaces.Box(low=obs_min, high=obs_max, dtype=np.float32),    # Obs space
                        spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),  # action space
                        {}
                    ),
                },
                policy_mapping_fn=lambda agent_id, episode, **kwargs: "shared_policy"
            )
            #.fault_tolerance(restart_failed_sub_environments=True, ignore_env_runner_failures=False)
            .training(
              replay_buffer_config={"type": "MultiAgentEpisodeReplayBuffer",
                                    "replay_sequence_length": 1, # Observations are already stacked, so only look at one obs when sampling from the buffer.
                                    "capacity": max_steps_range[1] * num_workers * 5},
              store_buffer_in_checkpoints=True,
              gamma=0.98,
              tau=0.005,
              actor_lr=0.00005,
              critic_lr=0.001,
            )
    )
    print(config.is_multi_agent)
    
    algo = config.build()
    
    
    # Convert betas? (solution found online, fixes a crash when loading a checkpoint)
    def betas_tensor_to_float(learner):
        for param_grp_key in learner._optimizer_parameters.keys():
            param_grp = param_grp_key.param_groups[0]
            param_grp["betas"] = tuple(beta.item() for beta in param_grp["betas"])
    if (load_from_checkpoint):
        algo.restore(checkpoint_load_dir)
        algo.learner_group.foreach_learner(betas_tensor_to_float)
    
    pprint.pprint(algo.config)
    # Main training loop!
    iteration = 0
    checkpoint = 0
    iterations_per_checkpoint = 200
    while True:
        result = algo.train()   # Perform a single training iteration (many steps, usually shorter than an episode. Changes depending on training parameters.)
        iteration += 1
        print(f"Iteration {iteration} complete")
        if (iteration % iterations_per_checkpoint == 0):
            checkpoint_dir = algo.logdir + f"/checkpoints/checkpoint_{checkpoint}"
            algo.save_checkpoint(checkpoint_dir) # Somehow get the directory from this?
            print(f"Saved checkpoint to {checkpoint_dir}")
            checkpoint += 1