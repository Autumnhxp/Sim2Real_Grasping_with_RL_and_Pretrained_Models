import debugpy
import sys
import os

os.environ["SIM-TO-REAL"] = "True"
os.environ["SAVE-OBSERVATION-IMAGE"] = "False"
os.environ["PLANAR-ACTION-SPACE"] = "True"
os.environ["DEBUG_POLICY_MODE"] = "True"
os.environ["NUMERIC-MODE"] = "False"  # == True, requires use_images = False in config.py and <EnableObservationImage> == False in config.xml
# print value of environment variable NUMERIC-MODE
os.environ["RANDOMIZE_TARGET"] = "True"

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
from stable_baselines3.common.vec_env import VecNormalize


if "--debug" in sys.argv:
    debugpy.listen(5678)
    debugpy.wait_for_client()


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
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
    return env




if __name__ == "__main__":


    # Open tensorboard: tensorboard --logdir ./tensorboard_logging/ --bind_all

    config_dict = Config.get_config_dict()
    env_dict = Config.get_dart_env_dict()
    env = get_env(config_dict, env_dict)

    experiment_name = "test_convergence_with_planaer_without_orientation_control_randomized_target_num_steps_10"

    # Experiment with n_steps

    model = A2C('MlpPolicy', env, ent_coef=0.01, vf_coef=0.5, verbose=1, n_steps=10, learning_rate=0.001, policy_kwargs = dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])]), gamma=0.96, tensorboard_log="/home/vtprl/agent/tensorboard_logging/")
    model.learn(total_timesteps=1500000, tb_log_name=experiment_name)
    tensorboard_experiment_path = model._logger.dir
    print("Experiment ended. Results & model saved to: ", tensorboard_experiment_path)
    print(model.policy)
    model.save(tensorboard_experiment_path + "/model")
    env.save(tensorboard_experiment_path + "/vec_normalize.pkl")

    # Loading the model and the normalization parameters
    # env = VecNormalize.load(tensorboard_experiment_path + "/vec_normalize.pkl", env)
    # model = A2C.load(tensorboard_experiment_path + "/model", env=env)

    # Notes on hyperparams:
    # ent_coef: Regulates exploration, higher means more
    # n_steps: Less steps made training more stable
    # net_arc: Number of neurons in the hidden layers of the policy and value function networks; Default is 64; More neurons significantly improved convergence (converged within 1 episode), yet slower performance
    # vf_coef: Regulates how much the value function is weighted in the loss function, default is 0.5; Lower share of vf_coef showed slower convergence in numeric experiemtn
