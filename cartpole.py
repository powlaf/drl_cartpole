import os
import random
import gym
import numpy as np
from collections import deque
from tensorflow import keras
from keras.models import Model, load_model
from keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Model, Sequential


class GymEnvironment:

    def __init__(self, env_id, max_timesteps=300):
        self.max_timesteps = max_timesteps
        self.env = gym.make(env_id)
        self.env._max_episode_steps = max_timesteps

        self.target_reward = 250

    def trainDQN(self, agent, no_episodes):
        # Initialize performance trackers and episode
        train_reward, train_epsilon, train_losses, test_reward = [], [], [], []
        e = 0

        ## FOR EACH EPISODE
        while e < no_episodes:
            # Initialize performance trackers
            episode_reward = 0
            episode_loss = []

            # Get initial state
            state = self.env.reset().reshape([1, 4])

            # ALL TIMESTEPS
            for t in range(1, self.max_timesteps+1):
                # Update performance tracker (total timesteps alive)
                episode_reward = t
                # self.env.render()

                # Take an action and get feedback
                action = agent.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, 4])

                # Adjust the reward to -100 if the cart pole is out of the boundaries
                if not done or t == self.max_timesteps:
                    reward = reward
                else:
                    reward = -100

                # Record the transition, train the agents' weights and perform epsilon decay
                agent.record(state, action, reward, next_state, done)
                episode_loss.append(agent.update_weights(t))
                agent.epsilon_decay(e)

                # Go to the next state and stop if the episode has terminated
                state = next_state
                if done:
                    break

            # Evaluation & Convergence:
            # Test every 5 episodes for 10 episodes. The algorithm has converges if the mean reward is high enough.
            if e % 5 == 0 and e > 2:
                timesteps_test = self.testDQN(agent, 10)
                test_reward.append(timesteps_test)
                if np.mean(timesteps_test) > self.target_reward:
                    print('Algorithm converged at episode {} with an average of {} - amazing job baby dqn'.format(e, np.mean(timesteps_test)))
                    break
                else
                  print('AVERAGE SCORE FROM 10 TEST SESSIONS: {}'.format(np.mean(timesteps_test)))

            # Collect performance data and print progress
            print("Training episode: {}/{}, score: {}, e: {:.2}".format(e + 1, no_episodes, t, agent.epsilon))
            if len(agent.memory) > agent.min_memory:
                train_reward.append(episode_reward)
                train_epsilon.append(agent.epsilon)
                train_losses.append(episode_loss)
                e += 1

        # Save model
        agent.model.save_weights("cartpole-v0.h5", overwrite=True)

        return train_reward, train_epsilon, train_losses, test_reward

    def testDQN(self, agent, no_episodes):
        # Initalize performance tracker
        test_rewards = []

        # ALL EPISODES
        for e in range(1, no_episodes + 1):
            # self.env.render()
            # Get initial state
            state = self.env.reset().reshape([1, 4])

            # ALL TIME STEPS
            for t in range(1, self.max_timesteps + 1):
                # Perform an action, observe environment and go to next state
                action = np.argmax(agent.model.predict(state))
                next_state, reward, done, _ = self.env.step(action)
                state = np.reshape(next_state, [1, 4])

                # Update performance
                reward_episode = t

                # Stop if environment is terminal
                if done:
                    break

            print("Testing episode: {}/{}, score: {}".format(e, no_episodes, t))

            # Collect performance parameter
            test_rewards.append(reward_episode)
        return test_rewards

    def adjust_reward(self, reward, next_state):
        if reward > 0:
            x = 0.5
            v = 0.5
            if next_state[0][0] > 0:
                x -= next_state[0][0] / 4.8
            elif next_state[0][0] < 0:
                x += next_state[0][0] / 4.8

            if next_state[0][2] > 0:
                v -= next_state[0][2] / 0.418
            elif next_state[0][2] < 0:
                v += next_state[0][2] / 0.418

            reward = x + v
        else:
            reward = -100

        return reward


class DQN_Agent:
    def __init__(self, no_of_states, no_of_actions, load_old_model):
        self.state_size = no_of_states
        self.action_size = no_of_actions
        self.load_old_model = load_old_model

        # Hyperparameter
        self.gamma = 0.99  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_dec = 0.999
        self.lr = 0.00025

        #  Primary and target network models
        self.target_model_time = 10
        self.model = self.nn_model(self.state_size, self.action_size)
        self.target_model = self.nn_model(self.state_size, self.action_size)
        self.target_model.set_weights(self.model.get_weights())

        # Memory and batch
        self.memory = deque(maxlen=2000)
        self.min_memory = 1000
        self.batch_size = 100

    def nn_model(self, state_size, action_size, load_old_model=0):
        # Create neural network with 3 hidden layers
        model = Sequential([
            Input(shape=(state_size,)),
            Dense(512, activation='relu', kernel_initializer='he_uniform'),
            Dense(256, activation='relu', kernel_initializer='he_uniform'),
            Dense(64, activation='relu', kernel_initializer='he_uniform'),
            Dense(action_size, activation='linear', kernel_initializer='he_uniform'),
        ])

        model.compile(loss="mse", optimizer=RMSprop(learning_rate=self.lr, rho=0.95, epsilon=0.01), metrics=["accuracy"])

        # Load existing model
        if load_old_model == 1:
            model = self.model.load_weights("cartpole-v0.h5")

        return model

    def select_action(self, state):
        # Epsilon greedy: either select an action randomly or let the primary network predict the optimal action
        if random.uniform(0, 1) < self.epsilon:
            act = random.randint(0, 1)
        else:
            act = np.argmax(self.model.predict(state))

        return act

    def record(self, state, action, reward, next_state, done):
        # Save transition to memory
        self.memory.append((state, action, reward, next_state, done))

    def epsilon_decay(self, e):
        # Decay the epsilon every training timestep
        if len(self.memory) > self.min_memory and e > 0:
                self.epsilon = max(self.epsilon*self.epsilon_dec, self.epsilon_min)

    def update_weights(self, t):
        # Wait until memory has reached a certain size
        if len(self.memory) >= self.min_memory:
          
            # Sample random (mini-)batch from memory
            batch = random.sample(self.memory, self.batch_size)

            # Get states, actions, rewards, next states and dones from sample and reformat
            states, actions, rewards, next_states, dones = [], [], [], [], []
            for i in range(self.batch_size):
                states.append(batch[i][0])
                actions.append(batch[i][1])
                rewards.append(batch[i][2])
                next_states.append(batch[i][3])
                dones.append(batch[i][4])
            states = np.array(states).reshape(self.batch_size, self.state_size)
            next_states = np.array(next_states).reshape(self.batch_size, self.state_size)

            # Predict the predicted QValue with the primary and the target QValue with the target network
            q_predicted = self.model.predict(states)  # [Q(s',0,w-), Q(s',1,w-)] -> predicted QValue
            q_target = self.target_model.predict(next_states)  # [Q(s, 0, w), Q(s, 1, w)] -> target QValue

            # Calculate the update q value for this state: r + gamma * (1-d) * max a' Q(s', a', w-)
            # And choose the updated q value for the performed action
            for i in range(self.batch_size):
                q_predicted[i][actions[i]] = rewards[i] + self.gamma * (1-int(dones[i])) * (np.amax(q_target[i]))

            # Train the primary neural network
            history = self.model.fit(states, q_predicted, batch_size=self.batch_size, verbose=0)

            # Update the target network
            if t % self.target_model_time == 0:
                self.target_model.set_weights(self.model.get_weights())

            return history.history['loss']


if __name__ == "__main__":

    # Initialize environment and agent
    no_of_states = 4
    no_of_actions = 2
    load_old_model = 0

    environment = GymEnvironment('CartPole-v0')
    agent = DQN_Agent(no_of_states, no_of_actions, load_old_model)

    # Train your agent
    train_episodes = 100
    train_reward, train_epsilon, train_loss, test_reward = environment.trainDQN(agent, train_episodes)

    # Run your agent
    test_episodes = 100
    run_reward = environment.testDQN(agent, test_episodes)

    # Simulation & Visualization
    visualize_agent = False
    if visualize_agent == True:
        env = gym.make('CartPole-v0')
        env._max_episode_steps = 500
        load_model = 1
        state_size = 4
        action_size = 2
        agent = DQN_Agent(state_size, action_size, load_model)
        state = env.reset().reshape(1, env.observation_space.shape[0])
        for e in range(10):
            for _ in range(500):
                env.render()
                action = act = np.argmax(agent.model.predict(state))
                next_state, reward, done, _ = env.step(action)
                if done: break
                state = next_state.reshape(1, env.observation_space.shape[0])
        env.close()
