import gymnasium as gym
import numpy as np
from gym_robotics_custom import RoboGymObservationWrapper

if __name__ == '__main__':
    env_name = "AntMaze_UMaze-v4"
    max_episode_steps = 1000

    STRIGHT_MAZE = [[1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1]]
    
    env = gym.make(env_name, max_episode_steps=max_episode_steps, render_mode="human", maze_map=STRIGHT_MAZE)
    env = RoboGymObservationWrapper(env)

    observation, info = env.reset()

    for i in range(100):
        action = env.action_space.sample()
        env.step(action)

    print(observation)