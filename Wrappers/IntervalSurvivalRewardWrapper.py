import gymnasium as gym
import math
import numpy as np
import stable_baselines3.common.vec_env


class IntervalSurvivalRewardWrapper(gym.Wrapper):
    """
    A wrapper for CartPole that changes the reward structure.
    Instead of +1 per step, it gives a reward at the end of the episode
    equal to floor(minutes survived) + 1.
    """

    def __init__(self, env, reward_interval_sec: float = 10.0, reward_per_interval: float = 0.10):
        super().__init__(env)
        self.reward_interval_sec = reward_interval_sec
        self.reward_per_interval = reward_per_interval

        # try to get the simulation time step from the environment
        # Default for cartpole is 0.02 seconds per step
        try:
            self.sim_seconds_per_step = self.env.unwrapped.tau
        except AttributeError:
            print("Warning: could not get simulation time step from environment. Defaulting to 0.02 seconds.")
            self.sim_seconds_per_step = 0.02

        self.episode_sim_time_elapsed = 0.0
        self.num_intervals_rewarded_this_episode = 0

    def reset(self, **kwargs):
        """Resets the environment and episode timer"""
        self.episode_sim_time_elapsed = 0.0
        self.num_intervals_rewarded_this_episode = 0

        # Ensure seed and options are passed to the wrapped environment's reset
        observation, info = self.env.reset(**kwargs)
        return observation, info

    def step(self, action):
        """
        Steps the environment and updates the episode timer
        """
        # step the environment
        observation, reward, terminated, truncated, info = self.env.step(action)

        self.episode_sim_time_elapsed += self.sim_seconds_per_step

        current_step_interval_reward = 0.0
        total_intervals_achieved_so_far = math.floor(self.episode_sim_time_elapsed / self.reward_interval_sec)

        if total_intervals_achieved_so_far > self.num_intervals_rewarded_this_episode:
            newly_achieved_intervals = total_intervals_achieved_so_far - self.num_intervals_rewarded_this_episode
            current_step_interval_reward = float(newly_achieved_intervals * self.reward_per_interval)
            self.num_intervals_rewarded_this_episode = total_intervals_achieved_so_far

            # Add useful info for logging or debugging
            if 'custom_reward_info' not in info:
                info['custom_reward_info'] = {}
            info['custom_reward_info']['intervals_newly_rewarded_count'] = newly_achieved_intervals
            info['custom_reward_info'][
                'total_intervals_rewarded_this_episode'] = self.num_intervals_rewarded_this_episode
            info['custom_reward_info']['reward_for_intervals_this_step'] = current_step_interval_reward
            info['custom_reward_info']['current_episode_sim_time_sec'] = round(self.episode_sim_time_elapsed, 2)

        final_reward_for_this_step = current_step_interval_reward + 0.001

        return observation, final_reward_for_this_step, terminated, truncated, info
