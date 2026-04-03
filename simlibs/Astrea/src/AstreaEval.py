
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
        
        
        obs_spaces = {}
        action_spaces = {}
        self.possible_agents = []
        for i in range(1000):
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
        self.runner.initialise(ini_variants_base + f".worker{os.getpid()}", "Astrea")
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
        
        print(info_)
        # Shutdown if ANY agent reports as done (lazy but works for this use case)
        if True in terminateds.values():      # TERMINATED - The RLAgent has reported itself as done (within the context of the MDP.) End the simulation.
            print("An Astrea agent has repoted as done, shutting down.")
            self.runner.shutdown()
            self.runner.cleanup()
        # if info_['simDone']:            # TRUNCATED - Environment/simulation has finished before the agent reported as done (usually a timelimit in the .ini)
        #     print("Simulation reported as complete, shutting down")
        #     self.runner.shutdown()
        #     self.runner.cleanup()  
        
        # TODO: Implement trucation. This was simple with single-agent, harder with multiple. Maybe use a copy of terminateds and replace the values.
        # OBS, REWARD, IS_TERMINATED, IS_TRUNCATED, EXTRA_INFO
        return  obs, rewards, terminateds, terminateds, {}
        
# Generates the OmnetGymApiEnv for the calling ray worker
def omnetgymapienv_creator(env_config):
    env = OmnetGymApiEnv(env_config)
    return env  # return an env instance



if __name__ == '__main__':
    env_name = "Astrea-inference"
    register_env(env_name, omnetgymapienv_creator)
    load_from_checkpoint = False
    checkpoint_load_dir = os.getenv('HOME') + "/ray_results/Astrea-1.0/PPO_Astrea-1.0_2026-04-02_21-58-40jrwrh5ro/checkpoints/checkpoint_2"
    steps_to_train = 20000000
    stacking = 5
    env_config = {"iniPath": sys.argv[1],
                  "stacking": stacking} # how many observations to keep in an obs_history
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
    
    ray.init(local_mode=True)
    config = (
        PPOConfig()
        # .env_runners(explore=False)
        .environment(env_name, env_config=env_config)   # "OmnetGymApiEnv
        #.training(num_steps_sampled_before_learning_starts=9999999)
        .multi_agent(
            policies={
                "shared_policy": (
                    None,
                    spaces.Box(low=obs_min, high=obs_max, dtype=np.float32),    # Obs space
                    spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),  # action space
                    {}
                ),
            },
            policy_mapping_fn=lambda agent_id, episode, **kwargs: "shared_policy"
        )
        )
    print(config.is_multi_agent)

    algo = config.build_algo()


    # Convert betas? (solution found online, fixes a crash when loading a checkpoint)
    def betas_tensor_to_float(learner):
        for param_grp_key in learner._optimizer_parameters.keys():
            param_grp = param_grp_key.param_groups[0]
            param_grp["betas"] = tuple(beta.item() for beta in param_grp["betas"])
    if (load_from_checkpoint):
        #algo.load_checkpoint(os.getenv('HOME') + "/ray_results/JAMESTEST")
        algo.restore(checkpoint_load_dir)
        algo.learner_group.foreach_learner(betas_tensor_to_float)

    # Main loop!
    steps = 0
    check_in_freq = 100
    while True:
        result = algo.training_step()
        
        print(f"Performing step: {steps}")
        if steps % check_in_freq == 0:
            print(f"Performing step: {steps}")
        steps += 1
    
    # config = (
    #         PPOConfig()
    #         .resources(num_gpus=len(gpus))
    #         .env_runners(num_env_runners=1) #, rollout_fragment_length=1000
    #         .environment(env_name, env_config=env_config, disable_env_checking=True) # "OmnetGymApiEnv
    #         .multi_agent(
    #             policies={
    #                 "shared_policy": (
    #                     None,
    #                     spaces.Box(low=obs_min, high=obs_max, dtype=np.float32),    # Obs space
    #                     spaces.Box(low=-2, high=.8, shape=(1,), dtype=np.float32),  # action space
    #                     {}
    #                 ),
    #             },
    #             policy_mapping_fn=lambda agent_id, episode, **kwargs: "shared_policy"
    #         )
    #         )
    # print(config.is_multi_agent)
    
    # algo = config.build_algo()
    
    
    # # Convert betas? (solution found online, fixes a crash when loading a checkpoint)
    # def betas_tensor_to_float(learner):
    #     for param_grp_key in learner._optimizer_parameters.keys():
    #         param_grp = param_grp_key.param_groups[0]
    #         param_grp["betas"] = tuple(beta.item() for beta in param_grp["betas"])
    # if (load_from_checkpoint):
    #     algo.restore(checkpoint_load_dir)
    #     algo.learner_group.foreach_learner(betas_tensor_to_float)
    
    # pprint.pprint(algo.config)
    # # Main training loop!
    # iteration = 0
    # checkpoint = 0
    # iterations_per_checkpoint = 1000
    # while True:
    #     result = algo.train()   # Perform a single training iteration (many steps, usually shorter than an episode. Changes depending on training parameters.)
    #     iteration += 1
    #     print(f"Iteration {iteration} complete")
    #     if (iteration % iterations_per_checkpoint == 0):
    #         checkpoint_dir = algo.logdir + f"/checkpoints/checkpoint_{checkpoint}"
    #         algo.save_checkpoint(checkpoint_dir) # Somehow get the directory from this?
    #         print(f"Saved checkpoint to {checkpoint_dir}")
    #         checkpoint += 1