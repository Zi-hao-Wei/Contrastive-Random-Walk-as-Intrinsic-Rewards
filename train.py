import gym
import control_flags as flag
import matplotlib.pyplot as plt
import numpy as np
import random
import argparse

env_name = "MountainCar-v0"

env = gym.make(env_name)
num_states = env.observation_space.shape[0]
num_actions = env.action_space.n
num_features = 64
s_low_b = env.observation_space.low
s_high_b = env.observation_space.high


print("Number of states:{}".format(num_states))
print("Number of actions:{}".format(num_actions))


def test_env_working():
    test_env = gym.make(env_name)
    for _ in range(300):
        test_env.render()
        action = test_env.action_space.sample()
        test_env.step(action)
    test_env.close()


if __name__ == "__main__":
    random.seed(0)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', type=str, default="crw",help='Methods chosen from crw, greedy, ucb, and rnd')
    args = parser.parse_args()
    
    if args.methods=="crw":
        from train_crw import Agent
    elif args.methods=="greedy":
        from train_greedy import Agent
    elif args.methods == "ucb":
        from train_ucb import Agent
    elif args.methods == "rnd":
        from train_rnd import Agent
    else:
        print("Wrong Method Args")
        exit(0) 

    if flag.CHECK_ENV_WORKS:
        test_env_working()
        print("Environment works.")
        exit(0)

    plt.style.use("ggplot")
    plt.figure()
    plt.ylim(-200,-110)
    
    #Test Feature Space
    if args.methods=="crw":
        agent = Agent(env, num_actions, num_states, num_features,1,128)
    else:
        agent = Agent(env,num_actions,num_states,num_features)
        
    running_reward = agent.run()
    episodes = np.arange(agent.max_episodes)
    
    plt.plot(episodes, running_reward)
    with open("Running_Rewards.npy","wb") as f:
        np.save(f,running_reward)

    plt.savefig("running_reward.png")
