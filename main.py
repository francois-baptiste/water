#!/usr/bin/env python3 -i
from __future__ import print_function
import ple
import numpy as np
import keras
import random
import copy
from pprint import pprint
import matplotlib.pyplot as plt
import os

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam, RMSprop

CREEPS = 3
EPHOCS = 50


class Agent():
    """Q learning agent."""

    def __init__(self, actions, max_memory=5000, gamma=.95, load=None):
        """Initialize the agent.
        
        :param list actions: A list with all the available actions
        :param int max_memory: The upper limit for out memory
        :param int gamma: Discount factor for the future rewards
        :param int epsilor: Exploration rate. How likely we are to choose
                            a random action instead of the best one
        :param int epsilon_min: The minimum value for the epsilon
        :param int epsilon_decat: How will the epsilon decay over time
        :param int learning_rate: The learning rate of the model
        """
        self._actions = actions
        self._max_memory = max_memory
        self._gamma=gamma
        self._epsilon = 1.0  # exploration rate
        self._epsilon_min = 0.01
        self._epsilon_decay = 0.995
        self._learning_rate = 0.001

        self._model = None
        if load:
            self.load(load)
        else:
            self.prepare_model()

        # memory of frames
        self._memory = []

    def save(self):
        """Save the model."""
        dirs = [int(i.replace("model_", "")) for i in os.listdir("models")]
        if not dirs:
            dirs = [0]
        name = "models/model_{}".format(max(dirs)+1)
        print("Saving ", name)
        self._model.save(name)

    def load(self, name):
        """Load a model"""
        self._model = keras.models.load_model(name)

    @property
    def memory(self):
        """Return a copy of the internal memory"""
        return copy.deepcopy(self._memory)

    @property
    def max_memory(self):
        """Return the max_memory limit."""
        return self._max_memory

    @property
    def actions(self):
        """Return the actions."""
        return copy.deepcopy(self._actions)
    
    def add_memory(self, memory):
        """Add a new state to the memory."""
        
        # make sure that the states are processed
        state, action, reward, next_state, done = memory
        self._memory.append((
            self.preprocess_state(state),
            action,
            reward,
            self.preprocess_state(next_state),
            done,
        ))

        while len(self._memory) > self._max_memory:
            # remove the latest memory until we get back to the maximum
            # capacity
            self._memory.pop(0)


    def prepare_model(self):
        """Prepare the NN model."""
        self._model = Sequential()
        self._model.add(
            Dense(164,
                  kernel_initializer='lecun_uniform',
                  # 3 coordinates for every creep (x, y, type)
                  # and two more for the player position
                  input_shape=(CREEPS*3+2,)
                 )
            )
        self._model.add(Activation('relu'))

        self._model.add(Dense(150,
                              kernel_initializer='lecun_uniform'))
        self._model.add(Activation('relu'))

        self._model.add(Dense(150,
                              kernel_initializer='lecun_uniform'))
        self._model.add(Activation('relu'))

        self._model.add(Dense(len(self._actions),
                              kernel_initializer='lecun_uniform'))

        #linear output so we can have range of real-valued outputs
        self._model.add(Activation('linear'))

        self._model.compile(loss='mse', optimizer=RMSprop())

    @staticmethod
    def preprocess_state(game_state):
        """Process the state in a simpler format for the network.

        Basically have an array with triples like this:
          (x, y, type)
        where type is -1 if that is a bad creep or +1 for a good creep

        The last 2 elements are the position of out player.
        """
        out = []
        for creep_x, creep_y in game_state['creep_pos']['BAD']:
            out.extend([creep_x, creep_y, -1])
        for creep_x, creep_y in game_state['creep_pos']['GOOD']:
            out.extend([creep_x, creep_y, 1])
        out.extend([game_state['player_x'], game_state['player_y']])
        # return np.array([out])[np.newaxis]
        return np.array([out])


    def pick_action(self, reward, obs):
        
        # with a probability choose a random action
        if np.random.rand() <= self._epsilon:
            return random.choice(self._actions)

        # if we don't choose a random action

        # pre-process the state
        state = self.preprocess_state(obs)

        # predict the Q values for all the actins
        pred = self._model.predict(state)

        # chose the action with the highest Q value
        return self._actions[np.argmax(pred)]

    def get_sample(self, batch_size):
        if batch_size > len(self._memory):
            return self._memory
        return random.sample(self._memory, batch_size)

    def replay(self, batch_size=1000):
        """Replay the memory and train the model."""
        # print("Replay {} from {}".format(batch_size, len(self._memory)))
        minibatch = self.get_sample(batch_size)

        x_train = None
        y_train = None
        for state, action, reward, next_state, done in minibatch:
            # if the game is done assume only the immediate reward
            target = reward

            # if the game is not done
            if not done:
                # add the current reward + ne next reward predicted by the network
                # discounted by a factor (gamma)
                target = reward + self._gamma * np.argmax(self._model.predict(next_state))

            # compute the output for the current state
            target_f = self._model.predict(state)

            # change only the spot for the reward that we choose
            # with the target reward
            # import pdb; pdb.set_trace()
            target_f[0, self._actions.index(action)] = target

            # train the model so that starting from the base state
            # it will approximate the reward for the action that it took
            # closer to the target (what we estimate in practice by running the
            # simulation)
            # self._model.fit(state, target_f, epochs=1, verbose=0)
            if x_train is None:
                x_train = state
            else:
                x_train = np.append(x_train, state, axis=0)

            if y_train is None:
                y_train = target_f
            else:
                y_train = np.append(y_train, target_f, axis=0)
        self._model.fit(
            x_train,
            y_train,
            epochs=1,
            verbose=0)

        # if the exploration factor is still not at minimum
        if self._epsilon > self._epsilon_min:
            # decay the exploration factor once
            self._epsilon *= self._epsilon_decay

if __name__ == "__main__":
    print("Start ")
    game = ple.games.waterworld.WaterWorld(
        width=480, height=480, num_creeps=CREEPS, 
    )
    reward_values={'positive': 500.0, 'negative': -500.0, 'tick': -0.01, 'loss': -5000.0, 'win': 100000.0}
    # env = ple.PLE(game, fps=30*40, display_screen=False, reward_values=reward_values)
    env = ple.PLE(game, fps=30, display_screen=True, reward_values=reward_values)
    print("rewards :", game.rewards)

    agent = Agent(actions=env.getActionSet(),
                  load='models/model_59')

    # agent = Agent(actions=env.getActionSet())


    env.init()

    # nb_frames = 2000
    nb_frames = EPHOCS * 1000
    # max_noops = 20
    rewards = []
    rewards_a = []
    scores = []
    reward = 0.0
    old_reward = 0.0
    
    # start our training loop
    for epoch in range(1, EPHOCS+1):
        for i in range(1000):
            # if the game is over, reset the game
            # we don't care about the "won" games and we don't count them
            # as the game can continue indefinitely we are only concerned with the
            # amount of frames
            if env.game_over():
                print("Finished a game !")
                env.saveScreen("screen_capture_{}_{}.png".format(epoch, i))
                env.reset_game()

            # get the action
            current_state = game.getGameState()
            action = agent.pick_action(reward, current_state)

            reward = env.act(action)
            old_reward += reward

            # keep the rewars
            rewards.append(old_reward)
            rewards_a.append(reward)
            scores.append(game.getScore())

            next_state = game.getGameState()
            agent.add_memory((current_state, action, reward, next_state, env.game_over()))
                
            agent.replay()
        agent.save()
        print("Run ", epoch)



    # plt.plot(old_reward)
    plt.plot(scores)
    plt.ylabel("Rewards")
    # plt.show()
    print("Done ----")
    agent.save()
