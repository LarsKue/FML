from typing import List
import numpy as np
from collections import deque

from .features import feature_matrix, feature_vector
import events as e



ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup_training(self):
    self.discount = 0.95

    self.alpha = 0.01
    self.alpha_decrease = 0.999

    self.epsilon = 0.15
    self.epsilon_decrease = 0.99
    self.min_epsilon = 0.10

    self.states_count = {}
    self.exploration_parameter = 10
    self.total_state_count = 0
    

    self.weights = np.random.uniform(-1, 1, 12).T
    self.weights = np.zeros(12)
    self.weights = np.array([2, 2, -1, 3, -3, 1, -1, 3, -1, 1, 1, 1], dtype=float).T
    self.weights = np.array([1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 1, 1], dtype=float).T


    self.previous_action = None
    
    self.experience = []
    self.batch_size = 50

    self.weights_updates = np.zeros((1000, 12))
    self.update_number = 0
    
    

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    if old_game_state == None:
        return

    """
    new_state_feature_matrix = feature_matrix(self, new_game_state)
    maxa_nextQ = np.max(self.weights.dot(new_state_feature_matrix))
    old_state_feature_vector = feature_vector(self, old_game_state, self_action)

    reward = calculate_reward(events)

    if reward != 0:
        self.epsilon *= self.epsilon_decrease
        self.alpha *= self.alpha_decrease
    
        self.epsilon = max(self.epsilon, 0.05)
        self.alpha = max(self.alpha, 0.05)


    delta = reward + self.discount * maxa_nextQ - self.weights.dot(old_state_feature_vector)
    self.weights += self.alpha * delta * old_state_feature_vector
    """

    if True or calculate_reward(events) != 0 or np.random.uniform() < 0.5:
        self.experience.append([
            feature_matrix(self, old_game_state), 
            self_action, 
            calculate_reward(events),
            feature_matrix(self, new_game_state),
            False,
            old_game_state['round']
        ])



    self.previous_action = self_action
    
    #print("\n")
    #print("action:", self.experience[-1][1])
    #print("reward:", self.experience[-1][2])
    #print(self.experience[-1][0])



def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    
    """
    old_state_feature_vector = feature_vector(self, last_game_state, last_action)
    reward = calculate_reward(events)


    delta = reward - self.weights.dot(old_state_feature_vector)
    self.weights += self.alpha * delta * old_state_feature_vector
    """

    if len(self.experience) > 0:
        self.experience.append([
            self.experience[-1][3], 
            last_action, 
            calculate_reward(events), 
            feature_matrix(self, last_game_state),
            True,
            last_game_state['round']
        ])


    #if last_game_state['step'] < 20 and last_game_state['round'] > 500:
    #    self.experience = self.experience[:-last_game_state['step']]

    #n_step_sarsa(self, 4)
    #n_step_sarsa(self, 3)

    if last_game_state['round'] > 0 and last_game_state['round'] % 10 == 0:
        experienceReplayQLinFApp(self, self.batch_size)
        self.batch_size += 10
        self.batch_size = min(self.batch_size, 500)

        self.weights_updates[self.update_number] = self.weights
        self.update_number += 1

        #self.alpha *= self.alpha_decrease
        #self.alpha = max(self.alpha, 0.001)

        n = self.update_number
        self.alpha = 1/(0.3*n+75)


    if last_game_state['round'] > 0 and last_game_state['round'] % 50 == 0:
        self.epsilon *= self.epsilon_decrease
        self.epsilon = max(self.epsilon, 0.05)

        self.experience = []

    #if last_game_state['round'] > 0 and last_game_state['round'] % 100 == 0:
    #    print("\n")
    #    sorted_states_count = [(count, state) for state, count in self.states_count.items()]
    #    sorted_states_count.sort(reverse=True)
    #    for count, state in sorted_states_count:
    #        state_ = " ".join([state[i:i+6] for i in range(0, len(state), 6)])
    #        print(f"{state_}:\t{count/self.total_state_count:0.3f}\t{self.epsilon*np.exp(-self.exploration_parameter * count / self.total_state_count) + self.min_epsilon:0.3f}\t{count}")

    if last_game_state['round'] > 0 and last_game_state['round'] % 1000 == 0:
        np.save('weights_updates.npy', self.weights_updates)



    
    if last_game_state['round'] > 0 and last_game_state['round'] % 10 == 0:
        print("")
        for w in self.weights:
            print(f"{w:0.3f}, ", sep="", end="")
            
        #print("\n")
        #for state, count in self.states_count.items():
        #    print(f"{state}:\t{count}")
        print("\n", self.epsilon, self.alpha)



def calculate_reward(events: List[str]):
    reward = 0

    for event in events:
        if event == e.INVALID_ACTION:
            reward += -200
        elif event == e.BOMB_DROPPED:
            reward += 5
        elif event == e.CRATE_DESTROYED:
            reward += 20
        elif event == e.COIN_FOUND:
            reward += 100
        elif event == e.COIN_COLLECTED:
            reward += 300
        elif event == e.KILLED_OPPONENT:
            reward += 500
        elif event == e.KILLED_SELF:
            reward += -100              # sd == KILLED_SELF + GOT_KILLED => reward(sd) = -500
        elif event == e.GOT_KILLED:
            reward += -400
        #elif event == e.SURVIVED_ROUND:
        #    reward += 20

    return reward / 100


def epsilon_greedy(self, game_state: dict):
    features = feature_matrix(self, game_state)

    state_string = np.array2string(features.flatten(), separator='', formatter={'float_kind': lambda x: f"{x:0.0f}"})[1:-2]


    epsilon = self.epsilon
    if state_string in self.states_count:
        epsilon *= np.exp(-self.exploration_parameter * self.states_count[state_string] / self.total_state_count)
        self.states_count[state_string] += 1
    else:
        self.states_count[state_string] = 1
    epsilon += self.min_epsilon
    self.total_state_count += 1
    #print("\n")
    
    if np.random.uniform() < epsilon:
        a = np.random.choice(ACTIONS)
        #print(state_string, np.round(epsilon, 3), "random:", a)
        return a
    else:
        #self.epsilon *= self.epsilon_decrease


        q = self.weights.dot(features)

        best_actions = np.argwhere(q == np.amax(q)).flatten()

        a = ACTIONS[np.random.choice(best_actions)]

        #if a == 'WAIT' and game_state['round'] > 10:
        #    print(features)
        #    print(np.round(self.weights, 2))
        #    print(np.round(q, 2))

        #print(state_string, np.round(epsilon, 3), " greedy:", a)

        return a
        



def act_train(self, game_state: dict):
    action = epsilon_greedy(self, game_state)
    return action




def n_step_sarsa(self, n):

    """
    print("\n")
    print("weights:", np.round(self.weights, 2))
    for i, step in enumerate(self.experience):
        print("step:", i)
        print("feature_matrix:\n", step[0])
        print("action:", step[1])
        print("reward:", step[2])
        print("\n")
    """

    
    #print("\n")
    #for step in self.experience:
    #    print(step[1], step[2])
    self.epsilon *= self.epsilon_decrease
    self.alpha *= self.alpha_decrease
    self.epsilon = max(self.epsilon, 0.05)
    self.alpha = max(self.alpha, 0.05)
    
    action_indices = {
        'UP': 0,
        'RIGHT':1,
        'DOWN':2,
        'LEFT':3,
        'WAIT':4,
        'BOMB':5,
    }
    episode = self.experience
    T = len(episode)
    rewards = np.array([step[2] for step in episode])
    gammas = np.array([self.discount**c for c in range(0, n)])
    gamma_pow_n = self.discount**n

    #print(rewards)
    current_weights = self.weights

    for t in range(T - n):
        G = (rewards[t+1:t+n+1] * gammas).sum()
        action_index_t = action_indices[episode[t][1]]
        action_index_t_n = action_indices[episode[t+n][1]]
        G = G + gamma_pow_n * current_weights.dot(episode[t+n][0][:,action_index_t_n])
        self.weights += self.alpha * (G - current_weights.dot(episode[t][0][:,action_index_t])) * episode[t][0][:,action_index_t]

        #print("action:", ACTIONS[action_index_t])
        #print("Return:", (rewards[t+1:t+n+1] * gammas).sum())
        #print("vector:", episode[t][0][:,action_index_t])
        #print("new weights:", np.round(self.weights, 2))
        
        
    #print(np.round(self.weights, 2))


def experienceReplayQLinFApp(self, batch_size):
    action_indices = {
        'UP': 0,
        'RIGHT':1,
        'DOWN':2,
        'LEFT':3,
        'WAIT':4,
        'BOMB':5,
    }

    batch_size = min(batch_size, len(self.experience))

    probability = np.array([1 if step[2] != 0 else 0.5 for step in self.experience])
    probability /= probability.sum()
    batch_indices = np.random.choice(np.arange(len(self.experience)), size=batch_size, replace=False, p=probability)
    #batch_indices = np.arange(batch_size)


    #for (s, a, r, s_, t) in self.experience:
    #    print("\n")
    #    print(s)
    #    print(a)
    #    print(r)
    #    print(s_)
    #    print(t)



    """
    batch = [self.experience[i] for i in batch_indices]

    w = self.weights

    for (old_feature_matrix, action, reward, new_feature_matrix, is_terminal) in batch:
        delta = reward
        if not is_terminal:
            delta = delta + self.discount * np.max(self.weights.dot(new_feature_matrix)) - self.weights.dot(old_feature_matrix[:,action_indices[action]])
        w = w + self.alpha / batch_size * delta * old_feature_matrix[:,action_indices[action]]


    self.weights = w
    
    """
    w = self.weights

    n = 4

    for i in batch_indices:
        (old_feature_matrix, action, reward, new_feature_matrix, is_terminal, episode) = self.experience[i]
        r = reward
        for j in range(1, n+1):
            if i+j >= len(self.experience) or self.experience[i+j][5] != episode:
                break
            r += self.experience[i+j][2] * self.discount**j
        j -= 1

        delta = r
        if not is_terminal:
            delta = delta + self.discount**j * np.max(self.weights.dot(self.experience[i+j][3])) - self.weights.dot(old_feature_matrix[:,action_indices[action]])


        #if old_feature_matrix[:,action_indices[action]][7] == 1 and old_feature_matrix[:,action_indices[action]][6] == 0 and self.weights[3] > 0 and self.weights[5] > 0:
        #    print("\ngood bomb that destroys crates")
        #    print(r)
        #    print(np.max(self.weights.dot(new_feature_matrix)))
        #    print(self.weights.dot(old_feature_matrix[:,action_indices[action]]))
        #    print(delta)
        #    print(self.alpha / batch_size * delta * old_feature_matrix[:,action_indices[action]])


        #print("\n", i)
        #print(r)
        #print(self.weights.dot(new_feature_matrix))
        #print(self.weights.dot(old_feature_matrix))
        #print(delta)
        #print(w)
        #print(old_feature_matrix[:,action_indices[action]])
        
        w = w + self.alpha / batch_size * delta * old_feature_matrix[:,action_indices[action]]
        

    self.weights = w
