import debugpy
import sys
import os
import argparse

from config import Config
from iiwa_sample_joint_vel_env import IiwaJointVelEnv
from iiwa_sample_env import IiwaSampleEnv
from simulator_vec_env import SimulatorVecEnv
from stable_baselines3 import PPO, A2C
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback


if "--debug" in sys.argv:
    debugpy.listen(5678)
    debugpy.wait_for_client()

os.environ["DEBUG_POLICY_MODE"] = "True"
os.environ["PLANAR-ACTION-SPACE"] = "True"


def get_env(config_dict, env_dict): 
    env_key = config_dict['env_key']

    def create_env(id=0):
        # 'dart' should be always included in the env_key when there is a need to use a dart based environment
        if env_key == 'iiwa_joint_vel':
            env = IiwaJointVelEnv(max_ts=250, id=id, config=config_dict)
        elif env_key == 'iiwa_sample_dart_unity_env':
            env = IiwaSampleEnv(max_ts=env_dict['max_time_step'],
                                orientation_control=env_dict['orientation_control'],
                                use_ik=env_dict['use_inverse_kinematics'],
                                ik_by_sns=env_dict['linear_motion_conservation'],
                                enable_render=env_dict['enable_dart_viewer'],
                                state_type=config_dict['state'],
                                env_id=id)
        return env

    num_envs = config_dict['num_envs']
    env = [create_env for i in range(num_envs)]
    env = SimulatorVecEnv(env, config_dict)
    return env




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--p", help="Path to the model")
    args = parser.parse_args()

    # Check if path was provided and file exists
    # if args.p and os.path.isfile(args.p):
    if args.p:
        path_to_model = "/home/vtprl/agent/logged_policies/experiment_1_r3m_resnet18_A2C_default_mlp_network_1500000_steps_24_envs_run__3/model.zip"
    else:
        print("Invalid or no path provided. Please specify a path to the model using --p flag.")
        sys.exit(1)
  
    config_dict = Config.get_config_dict()
    env_dict = Config.get_dart_env_dict()
    env = get_env(config_dict, env_dict)
    


    model = A2C('MlpPolicy', env, verbose=1, learning_rate=0.001, gamma=0.96)
    model.load(path_to_model) 
    model.learn(total_timesteps=1500000)

