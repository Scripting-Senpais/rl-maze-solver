# Import necessary libraries
import gymnasium as gym  # Gymnasium is a reinforcement learning environment toolkit
import numpy as np       # NumPy for numerical operations
from gymnasium import ObservationWrapper  # Base class for creating custom observation wrappers

# Define a custom observation wrapper class for a robotic environment
class RoboGymObservationWrapper(ObservationWrapper):
    
    def __init__(self, env):
        # Initialize the wrapper with the provided environment
        super(RoboGymObservationWrapper, self).__init__(env)

    def reset(self):
        """
        Reset the environment and return the processed observation and additional info.
        This function is automatically called at the beginning of each new episode.
        """
        observation, info = self.env.reset()              # Reset the base environment
        observation = self.process_observation(observation)  # Process observation to flatten and combine parts
        return observation, info

    def step(self, action):
        """
        Take a step in the environment using the provided action.
        Returns the processed observation, reward, done flag, truncated flag, and info dict.
        """
        observation, reward, done, truncated, info = self.env.step(action)  # Step the base environment
        observation = self.process_observation(observation)  # Process the observation
        return observation, reward, done, truncated, info

    def process_observation(self, observation):
        """
        Process the observation dictionary from the environment by concatenating:
        - 'observation' (usually the current state)
        - 'achieved_goal' (goal currently achieved by the agent)
        - 'desired_goal' (goal the agent is supposed to reach)
        into a single flat NumPy array.

        Returns:
            A single concatenated NumPy array that combines all relevant observation components.
        """
        obs_map = observation['observation']           # Core observation (e.g., position, velocity)
        obs_achieved_goal = observation['achieved_goal']  # Current goal status
        obs_desired_goal = observation['desired_goal']    # Target goal

        # Concatenate all components into one flat array for the agent to process easily
        obs_concatenated = np.concatenate((obs_map, obs_achieved_goal, obs_desired_goal))

        return obs_concatenated