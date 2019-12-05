#Edgar Gonzalez
#1001336686
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

class Block:
    def __init__(self, value, is_terminal, is_wall = False):
        self.reward = value
        self.is_terminal = is_terminal
        self.is_wall = is_wall

class Environment:
    def __init__(self, path, reward, gamma, K, learning_boundary):
        self.path = path
        self.reward = float(reward)
        self.gamma = float(gamma)
        self.number_of_moves = int(K)
        self.learning_boundary = float(learning_boundary)
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
    def random_start_state(self):
        states = []
        for x in range(self.rows):
            for y in range(self.cols):
                if not self[x,y].is_terminal and not self[x,y].is_wall:
                    states.append((x,y))
        return states[random.randint(0, len(states) - 1)]

def q_learning_update(env, old_state, action, new_state, Q, N):
    if env[new_state].is_terminal:
        Q[new_state][None] = env[new_state].reward
    if old_state is not None:
        if N[old_state][action] != 0:
            N[old_state][action] += 1
        else:
            N[old_state][action] = 1
        c = 1 / N[old_state][action]
        prev_q = Q[old_state][action]
        reward = env[old_state].reward
        max_utility = max(Q[new_state].values())
        value = ((1 - c) * prev_q) + c * (reward + env.gamma * max_utility)
        Q[old_state][action] = value

def f(state,action, Q, N, boundary):
    return 1 if N[state][action] < boundary else Q[state][action]

def get_action(env, state, Q, N):
    movements = [ env.up, env.down, env.left, env.right ]
    poss = []
    for x in movements:
        poss.append( f(state, x, Q, N, env.learning_boundary) )
    return movements[ int(np.argmax(poss)) ]

def calculate_new_state(env, old_state, action):
    up = {
        env.up: env.up,
        env.down: env.down,
        env.left: env.left,
        env.right: env.right
    }
    left = {
        env.up: env.left,
        env.down: env.right,
        env.left: env.down,
        env.right: env.up
    }
    right = {
        env.up: env.right,
        env.down: env.left,
        env.left: env.up,
        env.right: env.down

    }
    movements = [*[action for _ in range(8)], right[action], left[action]]
    action = movements[random.randint(0,9)]
    new_x = old_state[0] + action[0]
    new_y = old_state[1] + action[1]
    if new_x >= 0 and new_x < env.rows and new_y >= 0 and new_y < env.cols:
        if env[new_x, new_y].is_wall:
            new_x, new_y = old_state[0], old_state[1]
    else:
        new_x, new_y = old_state[0], old_state[1]
    return new_x, new_y

def agent_model_q_learning(env):
    movements = [env.up, env.down, env.left, env.right]
    Q = { (k,v) : { i: 0 for i in movements } for k in range(env.rows) for v in range(env.cols) }
    N = { (k,v) : { i: 0 for i in movements } for k in range(env.rows) for v in range(env.cols) }
    moves = 0
    while moves < env.number_of_moves:
        old_state = None
        action = None
        new_state = env.random_start_state()
        while True:
            q_learning_update(env, old_state, action, new_state, Q, N)
            if env[new_state].is_terminal:
                break
            action = get_action(env, new_state, Q, N)
            old_state = new_state
            new_state = calculate_new_state(env, old_state, action)
            moves += 1
    return Q

def main():
    if len(sys.argv) < 6:
        print('usage: [path_to_file] [non_terminal_rewards] [gamma] [number_of_moves] [learning_boundary]')
    else:
        env = Environment(*sys.argv[1:])
        Q = agent_model_q_learning(env)
        for x in range(env.rows):
            string = ''
            for y in range(env.cols):
                if env[x,y].is_terminal:
                    string += f'{env[x,y].reward:6.3f},'
                    continue
                string += f'{max(Q[x,y].values()):6.3f},'
            if string[-1] == ',':
                string = string[:-1]
            print(string)

if __name__ == '__main__':
    main()
