import gym
import numpy as np

# Create the CartPole environment
env = gym.make('CartPole-v1')

# Q-learning hyperparameters
learning_rate = 0.1
discount_factor = 0.99
epsilon = 0.1
num_episodes = 1000

# Initialize the Q-table with zeros
num_states = env.observation_space.shape[0]
num_actions = env.action_space.n
Q_table = np.zeros((num_states, num_actions))

# Q-learning algorithm
for episode in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(Q_table[state, :])  # Exploit

        next_state, reward, done, _ = env.step(action)
        
        # Q-value update using the Q-learning equation
        Q_table[state, action] = (1 - learning_rate) * Q_table[state, action] + \
                                 learning_rate * (reward + discount_factor * np.max(Q_table[next_state, :]))

        state = next_state

# Evaluate the trained policy
total_reward = 0
num_eval_episodes = 100
for _ in range(num_eval_episodes):
    state = env.reset()
    done = False
    
    while not done:
        action = np.argmax(Q_table[state, :])
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state

average_reward = total_reward / num_eval_episodes
print(f"Average reward: {average_reward}")
