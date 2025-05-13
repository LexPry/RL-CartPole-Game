import os

import gymnasium as gym
from stable_baselines3 import PPO
from tqdm import tqdm

from Wrappers.IntervalSurvivalRewardWrapper import IntervalSurvivalRewardWrapper
from utils import Utils


def setup_environment(env_name_registered, max_steps=6000,
                      render_mode="rgb_array"):
    try:
        gym.register(id=env_name_registered, entry_point="gymnasium.envs.classic_control:CartPoleEnv",
                     max_episode_steps=max_steps)
        print(f"Environment {env_name_registered} registered with max_episode_steps={max_steps}.")
    except gym.error.Error as e:
        print(f"Environment {env_name_registered} might already be registered: {e}")

    base_env = gym.make(env_name_registered, render_mode=render_mode)
    print(f"Created environment '{env_name_registered}' with spec: {base_env.spec}")  # Check the spec
    env = IntervalSurvivalRewardWrapper(base_env)
    return env


def train_agent(env, utils, ent_coef_val, gamma_val, total_timesteps_val):
    log_dir = os.path.join(utils.get_log_dir_path(), "PPO")
    model_save_path = utils.get_model_save_path()
    existing_model_path = utils.get_latest_model_load_path()

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir + "/", ent_coef=ent_coef_val, gamma=gamma_val)

    if os.path.isfile(existing_model_path):
        print(f"Loading model from: {existing_model_path}")
        model.set_parameters(existing_model_path)
        print("Model parameters loaded successfully.")
        model.ent_coef = ent_coef_val
        model.gamma = gamma_val
    else:
        print("No existing model found. Training from scratch.")

    print(f"Starting training for {total_timesteps_val} timesteps...")
    model.learn(total_timesteps=total_timesteps_val, progress_bar=False)
    print("Training complete.")
    model.save(model_save_path)
    print(f"Model saved to: {model_save_path}")
    return model


def evaluate_agent(model, num_episodes=5, num_steps_per_episode=7000):
    vec_env = model.get_env()
    for i in range(num_episodes):
        obs = vec_env.reset()
        print(f"\nStarting evaluation episode {i + 1}/{num_episodes}")
        for _ in tqdm(range(num_steps_per_episode), desc=f"Episode {i + 1}"):
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, info = vec_env.step(action)
            vec_env.render("human")
            if done:
                print(f"\nEpisode {i + 1} complete, finished after {_ + 1} steps.")
                break
    print("Evaluation complete.")


if __name__ == "__main__":
    # --- Configuration ---
    ENT_COEF = 0.01
    GAMMA = 0.99
    TOTAL_TIMESTEPS = 50_000
    REGISTERED_ENV_NAME = 'CartPole-v1-LongerIntervals'  # Use a more descriptive name
    BASE_ENV_ID = 'CartPole-v1'  # The actual base env ID for gym.make
    MAX_EPISODE_STEPS = 6000
    NUM_EVAL_EPISODES = 3
    STEPS_PER_EVAL_EPISODE = 7000  # e.g., 7000 steps = 140 seconds

    # initialize utils class
    utils = Utils()

    # --- Setup Environment ---
    # Pass BASE_ENV_ID to gym.make and REGISTERED_ENV_NAME to gym.register
    # The environment_name in your original code for gym.make should be the base ID.
    env = setup_environment(REGISTERED_ENV_NAME, max_steps=MAX_EPISODE_STEPS)

    # --- Train Agent ---
    model = train_agent(env, utils, ent_coef_val=ENT_COEF, gamma_val=GAMMA, total_timesteps_val=TOTAL_TIMESTEPS)

    # --- Evaluate Agent ---
    evaluate_agent(model, num_episodes=NUM_EVAL_EPISODES, num_steps_per_episode=STEPS_PER_EVAL_EPISODE)

    env.close()
