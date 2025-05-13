import math
from typing import Tuple, Any, Dict  # For more precise type hinting (Python 3.9+)
import gymnasium as gym
class IntervalSurvivalRewardWrapper(gym.Wrapper):
    """
    A Gymnasium Wrapper that modifies the reward structure of a wrapped environment.

    This wrapper replaces the original environment's reward with a custom one designed to
    incentivize survival over time. It provides:
    1. A periodic reward (`reward_per_interval`) every time the agent survives for
       a specified duration (`reward_interval_sec`).
    2. A small constant positive reward (0.001) added at every step, regardless of
       whether an interval is completed.

    This wrapper is intended to be used with environments where survival time is a key
    objective, such as CartPole. It tracks the cumulative simulation time within an
    episode to determine when interval rewards should be given.
    """

    def __init__(self, env, reward_interval_sec: float = 10.0, reward_per_interval: float = 0.10):
        """
        Initializes the IntervalSurvivalRewardWrapper.

        :param env: The Gymnasium environment to wrap.
        :param reward_interval_sec: The duration in seconds for each survival interval.
                                    A reward is given upon completion of each interval.
        :param reward_per_interval: The amount of reward given when a full survival
                                    interval is completed.
        """
        super().__init__(env)
        self.reward_interval_sec = reward_interval_sec
        self.reward_per_interval = reward_per_interval

        # Attempt to get simulation seconds per step from the environment's tau attribute.
        # Defaults to 0.02 (common for environments like CartPole) if tau is not found.
        try:
            self.sim_seconds_per_step = self.env.unwrapped.tau
        except AttributeError:
            print(f"Warning: Environment {env} does not have 'tau' attribute. Defaulting sim_seconds_per_step to 0.02.")
            self.sim_seconds_per_step = 0.02

        self.episode_sim_time_elapsed: float = 0.0
        self.num_intervals_rewarded_this_episode: int = 0

    def reset(self, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        """
        Resets the wrapped environment and internal state of this wrapper.

        Specifically, it resets the episode's simulated time elapsed and the count
        of rewarded intervals for the new episode. It then calls the `reset` method
        of the wrapped environment, passing through any provided keyword arguments.

        :param kwargs: Arbitrary keyword arguments passed to the wrapped environment's `reset` method.
        :return: A tuple containing the initial observation and information dictionary
                 from the wrapped environment.
        """
        self.episode_sim_time_elapsed = 0.0
        self.num_intervals_rewarded_this_episode = 0

        # Ensure seed and options are passed to the wrapped environment's reset
        observation, info = self.env.reset(**kwargs)
        return observation, info

    def step(self, action):
        """
        Steps the wrapped environment with the given action, calculates a custom reward,
        and updates internal timers.

        The custom reward consists of:
        - A bonus (`reward_per_interval`) if a new `reward_interval_sec` is completed during this step.
        - A constant small reward (0.001) added to every step's reward.

        The original reward from the wrapped environment is replaced.
        Diagnostic information about interval rewards is added to the `info` dictionary.

        :param action: The action to take in the wrapped environment.
        :return: A tuple containing the observation, calculated custom reward,
                 terminated flag, truncated flag, and info dictionary.
        """
        # Step the underlying environment; its reward is ignored by this wrapper.
        observation, original_reward, terminated, truncated, info = self.env.step(action)

        self.episode_sim_time_elapsed += self.sim_seconds_per_step

        current_step_interval_reward = 0.0
        # Calculate how many full intervals should have been rewarded by the current elapsed time
        total_intervals_achieved_so_far = math.floor(self.episode_sim_time_elapsed / self.reward_interval_sec)

        # If new intervals have been completed in this step
        if total_intervals_achieved_so_far > self.num_intervals_rewarded_this_episode:
            newly_achieved_intervals = total_intervals_achieved_so_far - self.num_intervals_rewarded_this_episode
            current_step_interval_reward = float(newly_achieved_intervals * self.reward_per_interval)
            self.num_intervals_rewarded_this_episode = total_intervals_achieved_so_far

            # Add useful diagnostic info for logging or debugging
            if 'custom_reward_info' not in info:  # Initialize if not present
                info['custom_reward_info'] = {}
            info['custom_reward_info']['intervals_newly_rewarded_count'] = newly_achieved_intervals
            info['custom_reward_info'][
                'total_intervals_rewarded_this_episode'] = self.num_intervals_rewarded_this_episode
            info['custom_reward_info'][
                'reward_for_intervals_this_step'] = current_step_interval_reward  # The part from intervals only
            info['custom_reward_info']['current_episode_sim_time_sec'] = round(self.episode_sim_time_elapsed, 2)

        # Final reward includes the interval reward (if any) plus a small constant bonus for every step
        final_reward_for_this_step = current_step_interval_reward + 0.001

        return observation, final_reward_for_this_step, terminated, truncated, info
