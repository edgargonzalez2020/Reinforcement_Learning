import numpy as np
import random
import sys
import csv

class Board:
    def __init__(self, arr, reward):
        self.reward = reward
        self.board = arr
        self.init_board()
        self.rows = len(self.board)
        self.cols = len(self.board[0])
    def init_board(self):
        for x in range(len(self.board)):
            for y in range(len(self.board[x])):
                if self.board[x][y] == '.':
                    self.board[x][y] = Block(self.reward,False)
                elif self.board[x][y] == 'X':
                    self.board[x][y] = Block(0, False, is_wall = True)
                else:
                    self.board[x][y] = Block(float(self.board[x][y]), True)
    def __repr__(self):
        string = ''
        for x in range(self.rows):
            for y in range(self.cols):
                string += f'{self.board[x][y].value}'
            string += '\n'
        return string


class Environment:
    def __init__(self, path, reward, gamma, K, boundary):
        self.path = path
        self.reward = float(reward)
        self.gamma = float(gamma)
        self.number_of_moves = int(K)
        self.boundary = float(boundary)
        self.init_board()
        self.rows = self.board.rows
        self.cols = self.board.cols

        # directions
        self.up = (-1, 0)
        self.down = (1, 0)
        self.left = (0, -1)
        self.right = (0, 1)
    def init_board(self):
        with open(self.path, 'r') as f:
            data = list(csv.reader(f))
            self.data = data
            self.board = Board(self.data, self.reward)
    def __getitem__(self, tup):
        x,y = tup
        return self.board.board[x][y]
    def get_random_state(self):
        move_set = [ (x,y)
                     for x in range(self.rows)
                     for y in range(self.cols)
                     if not self[x,y].is_wall and not self[x,y].is_terminal
                    ]
        return move_set[random.randint(0, len(move_set) - 1)]


class Block:
    def __init__(self, value, is_terminal, is_wall = False):
        self.reward = value
        self.is_terminal = is_terminal
        self.is_wall = is_wall


def q_learning_update(env, old_state, new_state, action, Q, N):
    if env[new_state].is_terminal:
        Q[new_state][None] = env[new_state].reward
    if old_state is not None:
        if N[old_state][action] != 0:
            N[old_state][action] += 1
        else:
            N[old_state][action] = 1
        c = 1 / N[old_state][action]
        old_utility = Q[old_state][action]
        old_reward = env[old_state].reward
        max_utility = max(Q[new_state].values())
        value = ((1-c) * old_utility) + c * (old_reward * max_utility)
        Q[old_state][action] = value



def f_function(state, action,Q, N, boundary):
    return 1 if N[state][action] < boundary else Q[state][action]


def get_action(env, state, Q, N):
    movements = [env.up, env.down, env.left, env.right]
    utilities = []
    for x in movements:
        utilities.append(f_function(state, x, Q, N, env.boundary))
    return movements[int(np.argmax(utilities))]


def move_agent(env, state, action):
    x = state[0] + action[0]
    y = state[1] + action[1]
    if x >= 0 and x < env.rows and y >= 0 and y < env.cols:
        if env[x,y].is_wall:
            x, y = state
    else:
        x, y = state
    return x, y



def agent_model_q_learning(env):
    movements = [env.up, env.down, env.left, env.right]
    Q = { (k,v): { x: 0 for x in movements } for k in range(env.rows) for v in range(env.cols) }
    N = { (k,v): { x: 0 for x in movements } for k in range(env.rows) for v in range(env.cols) }
    current_moves = 0
    while current_moves < env.number_of_moves:
        old_state = None
        action = None
        new_state = env.get_random_state()
        while True:
            current_moves += 1
            q_learning_update(env, old_state, new_state, action, Q, N)
            if env[new_state].is_terminal:
                break
            action = get_action(env, new_state, Q, N)
            old_state = new_state
            new_state = move_agent(env, old_state, action)
    return Q



def main():
    if len(sys.argv) < 6:
        print('usage: [path_to_file] [non_terminal_reward] [gamma] [number_of_moves] [boundary]')
    else:
        env = Environment(*sys.argv[1:])
        agent_model_q_learning(env)

if __name__ == '__main__':
    main()
