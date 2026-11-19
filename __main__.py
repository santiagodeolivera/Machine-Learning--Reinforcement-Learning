# A RL agent is trained on a concrete initial environment.
# The agent must support deciding the action to take depending on an observation.
# The agent must support receiving rewards for certain actions.
def train_rl_agent(agent, env, n_episodes):
    for _ in range(n_episodes):
        obs = env.reset()
        action = agent.decide_action(obs)
        while True:
            obs, reward, done = env.step(action)
            if done:
                break
            # TODO: make sure the previous environment is available
            # agent.receive_reward(prev_env, env, action, reward)
            action = agent.decide_action(obs)
