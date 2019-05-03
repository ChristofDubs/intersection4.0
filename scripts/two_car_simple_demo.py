import context
from intersection4.environments.two_car.two_car_env import TwoCarEnv
from intersection4.agents.two_car.two_car_simple import Agent

env = TwoCarEnv()
agent = Agent()
for i_episode in range(100):
    total_reward = 0
    old_state = env.reset()
    for t in range(100):
        env.render()
        action = agent.get_epsilon_greedy_action(old_state)
        new_state, reward, done, info = env.step(action)
        agent.save_data(old_state, action, reward)

        old_state = new_state
        total_reward += reward

        if done:
            env.render()
            agent.on_termination(new_state)
            print(
                "Episode {} finished after {} timesteps. Total reward: {}".format(
                    i_episode, t + 1, total_reward))
            break
