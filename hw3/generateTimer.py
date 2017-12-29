import gym
import pandas as pb
import tensorflow as tf
import time
import dqn_timer

def train_timer(num_action, eff_range):
    env = gym.make('Pong-ram-v0')

    timer = []

    for sp in range(num_action):
        env.reset()
        for _ in range(100):
            last_obs, _, _, _ = env.step(2)
            if last_obs[51]==38:
                break
        for t in range(100):
            control_output = dqn_timer.controller(last_obs[51], sp)
            last_obs, _, _, _ = env.step(control_output)
            agent_pos = last_obs[51]
            if agent_pos <= sp+38+eff_range and agent_pos >= sp+38-eff_range:
                timer.append(t)
                break

    return timer


"""
def main():
    # Run training
    timer = train_timer(166, 9)
    print(timer)

if __name__ == "__main__":
    main()
"""
