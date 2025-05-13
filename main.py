import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from tqdm import tqdm

from Wrappers.IntervalSurvivalRewardWrapper import IntervalSurvivalRewardWrapper
from utils import Utils


def setup_environment(env_name_registered, max_steps=6000,  # ~120 secs
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


def create_eval_agent(registered_env_name,
                      render_mode="rgb_array",
                      max_steps=6000):
    eval_env = gym.make(registered_env_name,
                        render_mode=render_mode,
                        max_episode_steps=max_steps)

    print(f"Eval environment created with spec: {eval_env.spec}")
    return eval_env


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


def train_agent(env,
                utils_obj: Utils,
                agent_name_str: str,
                ent_coef_val: float,
                env_name_str,
                gamma_val: float,
                total_timesteps_val: int,
                eval_env_obj: gym.Env):
    # log path for training
    safe_env_name_for_log = env_name_str.replace("/", "_").replace(":", "_")
    expiriment_log_name = f"{agent_name_str}_{safe_env_name_for_log}_train"

    # path for the model saved my train_agent at the end
    train_model_save_path = utils_obj.get_train_agent_model_save_path(
        agent_name=agent_name_str, env_name=env_name_str, timestamp=False
    )

    # Path for loading a previously train_agent saved model
    existing_train_model_path = utils_obj.get_latest_train_agent_model_load_path(
        model_prefix=f"{agent_name_str}_{safe_env_name_for_log}"
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=str(utils_obj.get_log_dir_path()),
        ent_coef=ent_coef_val,
        gamma=gamma_val,
    )

    if existing_train_model_path and existing_train_model_path.is_file():
        print(f"Loading parameters for train_agent model from: {existing_train_model_path}")
        model = PPO.load(str(existing_train_model_path), env=env, ent_coef=ent_coef_val, gamma=gamma_val)
        print("Model parameters loaded successfully.")
    else:
        print(
            f"No existing model found at {existing_train_model_path if existing_train_model_path else 'default search location'}. Training from scratch.")

    # --- EvalCallback Setup ---
    eval_callback_log_path = str(
        utils_obj.get_log_dir_path(experiment_name=f"{agent_name_str}_{safe_env_name_for_log}_eval"))
    # This is the DIRECTORY where EvalCallback will save "best_model.zip"
    eval_callback_best_model_dir = str(utils_obj.get_eval_callback_save_dir())

    eval_callback = EvalCallback(eval_env_obj,  # Use the passed eval_env_obj
                                 best_model_save_path=eval_callback_best_model_dir,
                                 log_path=eval_callback_log_path,
                                 n_eval_episodes=10,
                                 eval_freq=10_000,  # Adjust as needed
                                 deterministic=True,
                                 render=False)

    print(f"Starting training for {total_timesteps_val} timesteps...")
    model.learn(total_timesteps=total_timesteps_val, progress_bar=False, callback=eval_callback)
    print("Training complete.")

    # Save the final state of the model from this training run
    model.save(str(train_model_save_path))
    print(f"Final trained model saved to: {train_model_save_path}")

    # Now, load and return the BEST model found by EvalCallback for evaluation
    best_model_from_eval_path = utils_obj.get_path_for_loading_eval_best_model()
    if best_model_from_eval_path.is_file():
        print(f"Loading best model (from EvalCallback) from: {best_model_from_eval_path}")
        best_model = PPO.load(str(best_model_from_eval_path), env=env)
        return best_model
    else:
        print(f"Best model from EvalCallback not found at {best_model_from_eval_path}. Returning last trained model.")
        return model


if __name__ == "__main__":
    # --- Configuration ---
    ENT_COEF = 0.01
    GAMMA = 0.99
    TOTAL_TIMESTEPS = 60_000
    REGISTERED_ENV_NAME = 'CartPole-v1-LongerIntervals'
    BASE_ENV_ID = 'CartPole-v1'  # The actual base env ID for gym.make
    MAX_EPISODE_STEPS = 8000
    NUM_EVAL_EPISODES = 3
    STEPS_PER_EVAL_EPISODE = 10_000

    # initialize utils class
    utils = Utils()

    # --- Setup Environment ---
    env = setup_environment(REGISTERED_ENV_NAME, max_steps=MAX_EPISODE_STEPS)

    # --- Setup Eval Agent ---
    eval_env = create_eval_agent(REGISTERED_ENV_NAME)
    print(f"Eval environment created with spec: {eval_env.spec}")

    # --- Train Agent ---
    model = train_agent(env, utils, ent_coef_val=ENT_COEF, gamma_val=GAMMA, total_timesteps_val=TOTAL_TIMESTEPS,
                        eval_env_obj=eval_env, agent_name_str="PPO", env_name_str=BASE_ENV_ID)

    # --- Evaluate Agent ---
    evaluate_agent(model, num_episodes=NUM_EVAL_EPISODES, num_steps_per_episode=STEPS_PER_EVAL_EPISODE)

    env.close()
    eval_env.close()
