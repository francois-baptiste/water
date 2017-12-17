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
import time
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['GOTO_NUM_THREADS'] = '8'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['openmp'] = 'True'

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam, RMSprop

FPS = 30

WIDTH = 480
HEIGHT = 480

CREEPS = 3
EPHOCS = 30
LOAD = None
# model 80 e destul destul de bun
# model 159 a fost antrenat o noapte
# LOAD = 'models/model_159'

class Agent():
    """Q learning agent."""

    def __init__(self, actions, max_memory=5000, gamma=.95, load=None, game=None):
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
        self._game = game
        self._actions = actions
        self._max_memory = max_memory
        self._gamma = gamma
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
        self._model.add(Dropout(0.2))

        self._model.add(Dense(150,
                              kernel_initializer='lecun_uniform'))
        self._model.add(Activation('relu'))
        self._model.add(Dropout(0.2))

        self._model.add(Dense(150,
                              kernel_initializer='lecun_uniform'))
        self._model.add(Activation('relu'))
        self._model.add(Dropout(0.2))

        self._model.add(Dense(len(self._actions),
                              kernel_initializer='lecun_uniform'))

        #linear output so we can have range of real-valued outputs
        self._model.add(Activation('linear'))

        self._model.compile(loss='mse', optimizer=RMSprop())

    def preprocess_state(self, game_state):
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

    def replay(self, batch_size=100):
        """Replay the memory and train the model."""

        if batch_size > len(self._memory):
            # to avoid overt fitting we don't reply
            return
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
            epochs=2,
            verbose=0)

        # if the exploration factor is still not at minimum
        if self._epsilon > self._epsilon_min:
            # decay the exploration factor once
            self._epsilon *= self._epsilon_decay


class Sensors(Agent):

    @staticmethod
    def distance_line_creep(a, b,c, x, y):
        """Distance between a line and a creep."""
        return abs(a*x + b*y + c) / ((a**2 + b**2) ** 0.5)

    def preprocess_state(self, game_state):
        """Process the state in a simpler format for the network.

        Basically have an array with triples like this:
          (x, y, type)
        where type is -1 if that is a bad creep or +1 for a good creep

        The last 2 elements are the position of out player.
        """

        my_x = game_state['player_x'] # -1 to normalize the graph
        my_y = HEIGHT - game_state['player_y']

        sensors = [
            (1, -1, 0), # f(x) = x
            (1, 1, 0), # f(x) = -x
            (4, -1, 0), # f(x) = 4x
            (4, 1, 0), # f(x) = -4x

            (.25, -1, 0), # f(x) = 0.25*x
            (.25, 1, 0), # f(x) = -0.25*x
            (0, -1, 1), # axa x
            (-1, 0, 1), # axa y
        ]
        # move the lines to intersect with my current position 
        sensors = [ (a, b, c+my_x) for a, b, c in sensors]

        def dist(x, y):
            """Distance from the agent."""
            return ((float(my_x) - float(x))**2 + (float(my_y) - float(y))**2) ** 0.5

        pairs = [('BAD', -1), ('GOOD', +1)]
        state = []
        for f in [lambda x: my_x > x,
                lambda x : my_x <= x]:
            for a, b, c in sensors:
                # the distance for the closest creep that intersects with that line
                
                # first get all creeps that intersect
                intersects = []
                for creep_type, sign in pairs:
                    for x, y in game_state['creep_pos'][creep_type]:
                        y = HEIGHT - y # normalize the graph 
                        # if the distance between the line and the creep center is less then the
                        # creep radius
                        if self.distance_line_creep(a, b, c, x, y) < self._game.AGENT_RADIUS and f(x):
                            intersects.append((x, y, dist(x, y), sign))
                
                # we have a list with the intersections
                # now we need to see the distance
                if intersects:
                    intersects.sort(key=lambda x: x[3])
                    x, y, d, s = intersects[0]
                    state.append(s * d)
                else:
                    state.append(0)
        state = np.array([state]) 
        # import pdb; pdb.set_trace()
        return state

    def prepare_model(self):
        """Prepare the NN model."""
        self._model = Sequential()
        self._model.add(
            Dense(164,
                  kernel_initializer='lecun_uniform',
                  # 3 coordinates for every creep (x, y, type)
                  # and two more for the player position
                  input_shape=(16,)
                 )
            )
        self._model.add(Activation('relu'))
        self._model.add(Dropout(0.2))

        self._model.add(Dense(150,
                              kernel_initializer='lecun_uniform'))
        self._model.add(Activation('relu'))
        self._model.add(Dropout(0.2))

        self._model.add(Dense(150,
                              kernel_initializer='lecun_uniform'))
        self._model.add(Activation('relu'))
        self._model.add(Dropout(0.2))

        self._model.add(Dense(len(self._actions),
                              kernel_initializer='lecun_uniform'))

        #linear output so we can have range of real-valued outputs
        self._model.add(Activation('linear'))

        self._model.compile(loss='mse', optimizer=RMSprop())


def neighbors(arr,x,y,n=100):
    ''' Given a 2D-array, returns an nxn array whose "center" element is arr[x,y]'''
    arr=np.roll(np.roll(arr,shift=-x+1,axis=0),shift=-y+1,axis=1)
    return arr[:n,:n]

if __name__ == "__main__":
    print("Start ")
    game = ple.games.waterworld.WaterWorld(
        width=WIDTH, height=HEIGHT, num_creeps=CREEPS, 
    )

    reward_values={'positive': 500.0, 'negative': -500.0, 'tick': -0.01, 'loss': -5000.0, 'win': 5000.0}

    if not LOAD:
        env = ple.PLE(game, fps=FPS*4000, display_screen=False, reward_values=reward_values)
    else:
        env = ple.PLE(game, fps=FPS, display_screen=True, reward_values=reward_values)
    print("rewards :", game.rewards)


    # agent = Agent(actions=env.getActionSet(),
    #               load=LOAD, game=game)

    agent = Sensors(actions=env.getActionSet(),
                    load=LOAD, game=game)

    env.init()

    rewards_a = []
    scores = []
    reward = 0.0


    # start our training loop
    start_e = time.time()
    for epoch in range(1, EPHOCS+1):
        for i in range(1000):

            if FPS == 1:
                time.sleep(5)
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

            # keep the rewars
            rewards_a.append(reward)
            scores.append(game.getScore())

            next_state = game.getGameState()
            agent.add_memory((current_state, action, reward, next_state, env.game_over()))
            if not LOAD:
                agent.replay()
        agent.save()
        print("Run {}, took: {:.4f}".format(epoch, time.time() - start_e))
        start_e = time.time()



    plt_rewards_a, = plt.plot(rewards_a,  label='reward per action')
    plt_score, = plt.plot(scores, label='game score')
    plt.legend(handles=[plt_rewards_a, plt_score])
    plt.ylabel("Rewards")
    plt.ylabel("frame count")
    plt.show()
    print("Done ----")
    agent.save()
