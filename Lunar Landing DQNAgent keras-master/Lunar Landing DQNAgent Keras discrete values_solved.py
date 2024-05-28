#pip install rl-agents==0.1.1
#pip install tensorflow
#pip install gym
#pip install keras
#pip install keras-rl2
#pip install pygame
#pip install tensorflow==2.15
#conda install swig 
#conda install -c conda-forge box2d-py 

import gym

#Game being used
# ->https://gymnasium.farama.org/environments/box2d/lunar_lander/
env = gym.make('LunarLander-v2')
states = env.observation_space.shape[0]

#Number of actions available.
actions = env.action_space

print(states)
print(actions.n)

#Env setting with random behaviour
episodes = 10
for episode in range(1,episodes+1):
    state = env.reset()
    done = False
    score = 0
        
    while not done:
        env.render()
        n_state, reward, done, info = env.step(env.action_space.sample())
        score+=reward
    print('Episode:{} Score:{}'.format(episode,score))
    
env.close()

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.optimizers.legacy import Adam

from keras import __version__
tf.keras.__version__ = __version__

#https://www.tensorflow.org/agents/api_docs/python/tf_agents/agents/DdpgAgent?hl=en
#from rl.agents import SARSAAgent #check diferent agents -> https://keras-rl.readthedocs.io/en/latest/
from rl.agents import DQNAgent #check diferent agents -> https://keras-rl.readthedocs.io/en/latest/
from rl.policy import EpsGreedyQPolicy
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

def build_model(states, actions):
    model = Sequential() 
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape)) 
    model.add(Dense(200,activation='relu'))
    model.add(Dense(200,activation='relu'))
    model.add(Dense(actions,activation='linear'))    
    return model

model = build_model(states,actions.n)

model.summary()

def build_agent(model, actions):
    policy = EpsGreedyQPolicy()
    #earlystop = EarlyStopping(monitor = 'episode_reward', min_delta=.1, patience=5, verbose=1, mode='auto') 
    memory = SequentialMemory(limit=50000,window_length=1)
    #callbacks = [earlystop] 
    nb_steps_warmup = 1000 
    target_model_update = .02 
    #gamma = .99 
    #epochs = training_steps/1000 
    #decay = float(lr/epochs) 
    dqn = DQNAgent(model=model, nb_actions=actions, memory=memory, nb_steps_warmup=nb_steps_warmup, target_model_update = target_model_update, policy=policy)
    return dqn

 #Adam._name = 'hey' ## use in case of error mentioning this parameter as null-
dqn = build_agent(model,actions.n)

lr = .0001 
dqn.compile(Adam(learning_rate=lr), metrics=['mae'],)

training_steps = 1000000

dqn.fit(env, nb_steps=training_steps, visualize=False, verbose=1) 

scores = dqn.test(env, nb_episodes=100, visualize=False)
print(np.mean(scores.history['episode_reward']))

#lets test again with more 15 episodes
_ = dqn.test(env, nb_episodes=15, visualize=True)
env.close()

dqn.save_weights('LunarLander-v2_weights.h5f',overwrite=True)