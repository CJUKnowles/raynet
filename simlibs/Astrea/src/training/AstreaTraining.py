
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
from ray.rllib.algorithms.ppo.ppo import PPOConfig
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
        self.max_steps_range = self.env_config["max_steps_range"][1]
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
                    [1,                            # Throughput
                    10,                           # Number of ACKs
                    1,                            # Interval duration
                    1,                            # srtt
                    1,                            # Delay metric
                    0,
                    0
                    ], dtype=np.float32), self.stacking)
        
        
        obs_spaces = {}
        action_spaces = {}
        self.possible_agents = []
        for i in range(100):
            self.possible_agents.append(f"`Astrea`{i}")
            obs_spaces[f"Astrea{i}"] = spaces.Box(low=self.obs_min, high=self.obs_max, dtype=np.float32)
            action_spaces[f"Astrea{i}"] = spaces.Box(low=-1.0, high=1.0, dtype=np.float32)
        
        self.observation_space = spaces.Dict(obs_spaces)
        self.action_space = spaces.Dict(action_spaces)
        
       
    def reset(self, *, seed=None, options=None):
        if self.has_reset:
            return
        self.has_reset = True
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
        ini_variants_base = f"{self.env_config["iniPath"].rsplit("/", 1)[0]}/ini_variants/{self.env_config["iniPath"].rsplit("/", 1)[1]}"
        with open(original_ini_file, 'r') as fin:
            ini_string = fin.read()
        ini_string = ini_string.replace("HOME",  os.getenv('HOME'))
        ini_string = ini_string.replace("ORCA_BOTTLENECK_BW", f"{self.bw}Mbps")
        ini_string = ini_string.replace("ORCA_BASE_RTT", f"{self.base_rtt/2.0}ms")  # Delay goes both ways, divide by two
        ini_string = ini_string.replace("ORCA_BOTTLENECK_BUFFER_SIZE", f"{self.buffer_size}b")
        ini_string = ini_string.replace("MAX_RL_STEPS", f"{self.max_steps}")
        # TODO: Include these strings in the .ini somewhere that actually makes them alter the experiment
        with open(ini_variants_base + f".worker{os.getpid()}", 'w') as fout:
            fout.write(ini_string)
        
        # Start a new simulation runner on the modified ini file
        self.runner.initialise(ini_variants_base + f".worker{os.getpid()}", "General")
        obs = self.runner.reset()
        
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
        
        # Check if this training episode is complete
        if terminateds['__all__']:      # TERMINATED - The RLAgent has reported itself as done (within the context of the MDP.) End the simulation.
            self.runner.shutdown()
            self.runner.cleanup()
        if info_['simDone']:            # TRUNCATED - Environment/simulation has finished before the agent reported as done (usually a timelimit in the .ini)
            sim_truncated = True
        else:
            sim_truncated = False    
        
        # TODO: Implement trucation. This was simple with single-agent, harder with multiple. Maybe use a copy of terminateds and replace the values.
        # OBS, REWARD, IS_TERMINATED, IS_TRUNCATED, EXTRA_INFO
        return  obs, rewards, terminateds, terminateds, {}
        
# Generates the OmnetGymApiEnv for the calling ray worker
def omnetgymapienv_creator(env_config):
    env = OmnetGymApiEnv(env_config)
    return env  # return an env instance



if __name__ == '__main__':
    env_name = "Astrea-evaltest"
    register_env(env_name, omnetgymapienv_creator)
    num_workers = 1 # Must be >= 1. A value of 0 will spawn a single worker that does not reset if issues occur. 1+ allows resets.
    seed = 91456211
    # bottleneck_bandwidth_range = (6, 192)            # Orca: 6Mbps-192Mbps
    # minimum_rtt_range = (4, 400)                     # Orca: 4ms-400ms
    # bottleneck_buffer_range = (3000, 96000000)       # Orca: 3KB-96MB, expressed in terms of bits
    max_steps_range = (5000, 5000)                   # Custom: Randomize ending time slightly so threads desync, to make log outputs less sparse
    bottleneck_bandwidth_range = (6, 6)            
    minimum_rtt_range = (5, 5)
    bottleneck_buffer_range = (5280000, 5280000) 
    load_from_checkpoint = False
    checkpoint_load_dir = os.getenv('HOME') + "/ray_results/SAC_OmnetGymApiEnv_2026-03-10_01-19-546lihpmj1/checkpoints/checkpoint_22"
    steps_to_train = 20000000
    stacking = 3
    env_config = {"iniPath": os.getenv('HOME') + "/raynet/simlibs/Astrea/src/training/AstreaTraining.ini",
                  "bottleneck_bw_range": bottleneck_bandwidth_range,
                  "minimum_rtt_range": minimum_rtt_range,
                  "bottleneck_buffer_range": bottleneck_buffer_range,
                  "max_steps_range": max_steps_range,
                  "stacking": stacking} # how many observations to keep in an obs_history
    random.seed(seed)
    np.random.seed(seed)
    gpus = GPUtil.getGPUs()
    print("GPUs Available:", gpus)
    
    obs_min = np.tile(np.array(
                     [0,                            # Throughput
                      0,                            # number of acks
                      0,                            # Interval duration
                      0,                            # srtt
                      0,                             # Delay metric
                      0,
                      0
                      ], dtype=np.float32), stacking)
    obs_max = np.tile(np.array(
                [1,                            # Throughput
                10,                           # Number of ACKs
                1,                            # Interval duration
                1,                            # srtt
                1,                            # Delay metric
                1,
                1
                ], dtype=np.float32), stacking)
        
    #dummy_env = omnetgymapienv_creator(env_config)
    config = (
            PPOConfig()
            .resources(num_gpus=len(gpus))
            .env_runners(num_env_runners=num_workers) #, rollout_fragment_length=1000
            .learners(num_learners=1, num_gpus_per_learner=len(gpus), num_cpus_per_learner=1)
            .environment(env_name, env_config=env_config, disable_env_checking=True) # "OmnetGymApiEnv
            .multi_agent(
                policies={
                    "shared_policy": (
                        None,
                        spaces.Box(low=obs_min, high=obs_max, dtype=np.float32),    # Obs space
                        spaces.Box(low=-2, high=.8, shape=(1,), dtype=np.float32),  # action space
                        {}
                    ),
                },
                policy_mapping_fn=lambda agent_id, episode, **kwargs: "shared_policy"
            )
            #.training(optimizer={"foreach": False, "capturable": True})
            ##.evaluation(evaluation_interval=1000, evaluation_duration_unit="timesteps")
            ##.fault_tolerance(restart_failed_sub_environments=True)
            #.training(training_intensity=1000)  # num_steps_sampled_before_learning_starts=0 training_intensity=1000
            # .build_algo()
            )
    print(config.is_multi_agent)
    
    algo = config.build_algo()
    
    
    # config = {
    #     "env": env_name,
    #     "env_config": env_config,
    #     "evaluation_config": {
    #         "explore": False
    #     },
    #     "num_workers": num_workers,
    #     "multiagent": {
    #         "count_steps_by": "agent_steps"
    #     }
    # }
    # ray.init(num_cpus=16, num_gpus=len(gpus))
    
    # cls = get_trainable_cls("SAC")
    # env = omnetgymapienv_creator(config['env_config'])
    # agent = cls(env=env_name, config=config)
    
    
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
    
