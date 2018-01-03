import gym
import pandas as pb
import tensorflow as tf
import time
import dqn_timer

def train_timer(num_action, eff_range):
    env = gym.make('Pong-ram-v0')

    timer = []

    for action in range(num_action):
        env.reset()
        sp = action * 16 + 38
        for _ in range(100):
            last_obs, _, _, _ = env.step(2)
            if last_obs[51]==38:
                break
        for _ in range(5):
            last_obs, _, _, _ = env.step(0)
        for t in range(100):
            control_output = dqn_timer.controller(last_obs[51], sp, t)
            last_obs, _, _, _ = env.step(control_output)
            agent_pos = last_obs[51]
            if agent_pos <= sp+eff_range and agent_pos >= sp-eff_range:
                timer.append(t)
                break

    return timer


"""
def main():
    # Run training
    timer = train_timer(11, 8)
    print(timer)

if __name__ == "__main__":
    main()
"""
