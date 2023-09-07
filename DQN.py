import numpy as np
import random
import gym
import tensorflow as tf

# Create the CartPole environment
env = gym.make('CartPole-v1')

# DQN hyperparameters
learning_rate = 0.001
discount_factor = 0.99
epsilon_initial = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
replay_buffer_size = 10000
batch_size = 32
num_episodes = 1000
target_update_frequency = 10

# Initialize Q-networks and target network
input_shape = env.observation_space.shape[0]
output_shape = env.action_space.n

q_network = tf.keras.Sequential([
    tf.keras.layers.Dense(24, activation='relu', input_shape=(input_shape,)),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(output_shape)
])

target_network = tf.keras.Sequential([
    tf.keras.layers.Dense(24, activation='relu', input_shape=(input_shape,)),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(output_shape)
])
target_network.set_weights(q_network.get_weights())

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Experience replay buffer
replay_buffer = []

# DQN algorithm
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    epsilon = max(epsilon_min, epsilon_initial * (epsilon_decay ** episode))
    
    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            q_values = q_network.predict(state[None, :])
            action = np.argmax(q_values)  # Exploit
        
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        
        # Store transition in replay buffer
        replay_buffer.append((state, action, reward, next_state, done))
        
        # Update Q-network using minibatch from replay buffer
        if len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            targets = rewards + discount_factor * (1 - np.array(dones)) * \
                      np.amax(target_network.predict(np.array(next_states)), axis=1)
            
            with tf.GradientTape() as tape:
                q_values = q_network(np.array(states))
                selected_q_values = tf.reduce_sum(q_values * tf.one_hot(actions, output_shape), axis=1)
                loss = tf.reduce_mean(tf.square(targets - selected_q_values))
            
            gradients = tape.gradient(loss, q_network.trainable_variables)
            optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))
        
        state = next_state
    
    # Update target network periodically
    if episode % target_update_frequency == 0:
        target_network.set_weights(q_network.get_weights())
    
    print(f"Episode: {episode}, Total Reward: {total_reward}")
