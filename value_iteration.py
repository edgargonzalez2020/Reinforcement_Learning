#Edgar Gonzalez
#1001336686
import copy
import csv
import sys
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
                    self.board[x][y] = Block(self.reward, False)
                elif self.board[x][y] == 'X':
                    self.board[x][y] = Block(0, False, is_wall = True)
                else:
                    self.board[x][y] = Block(float(self.board[x][y]), True)
    def __repr__(self):
        string = ''
        for x in range(self.rows):
            for y in range(self.cols):
                string += f'{self.board[x][y].value} '
            string += '\n'
        return string


class Block:
  def __init__(self, value, is_terminal, is_wall = False):
    self.value = value
    self.is_terminal = is_terminal
    self.is_wall = is_wall

class ValueIteration:
    def __init__(self, path, reward, gamma, K):
        self.path = path
        self.reward = float(reward)
        self.gamma = float(gamma)
        self.k = int(K)
        self.init_board()
        self.rows = self.board.rows
        self.cols = self.board.cols

        #directions
        self.up = (-1,0)
        self.down = (1, 0)
        self.left = (0, -1)
        self.right = (0, 1)
    def init_board(self):
        with open(self.path, "r") as f:
          data = list(csv.reader(f))
          self.data = data
          self.board = Board(self.data, self.reward)
    def __getitem__(self, tup):
        x,y = tup
        return self.board.board[x][y]
def calculate_action(x, y, env, utility, action):
   new_state = (x + action[0], y + action[1])
   if new_state[0] >= 0 and new_state[0] < env.board.rows and new_state[1] >= 0 and new_state[1] < env.board.cols:
       if env[new_state[0],new_state[1]].is_wall:
           new_state = (x, y)
   else:
       new_state = (x,y)
   return utility[new_state]
def calculate_utility(x, y, env, utility):
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
    probabilities = [(up, 0.8), (left, 0.1), (right, 0.1)]
    movements = [env.up, env.down, env.left, env.right]
    totals = {k: 0 for k in movements}
    for k in movements:
        total = 0
        for moves, prob in probabilities:
            total += prob * calculate_action(x,y,env,utility, moves[k])
        totals[k] = total
    return max(totals.values())
def value_iteration(env):
    U_p = {(x,y): 0 for x in range(env.rows) for y in range(env.cols)}
    for i in range(env.k):
        U = copy.deepcopy(U_p)
        for x in range(env.rows):
            for y in range(env.cols):
                state = (x,y)
                if not env[x,y].is_wall and not env[x,y].is_terminal:
                    U_p[state] = env.board.board[x][y].value + env.gamma * calculate_utility(*state, env, U)
                if env[x,y].is_terminal:
                    U_p[state] = env[x,y].value
    return U_p
def main():
    if len(sys.argv) < 5:
        print("usage: [path_to_environment_file] [non_terminal_reward] [gamma] [K]")
    else:
        program = ValueIteration(*sys.argv[1:])
        U = value_iteration(program)
        for x in range(program.rows):
            string = ''
            for y in range(program.cols):
                string += f'{U[x,y]:6.3f},'
            if string[-1] == ',':
                string = string[:-1]
            print(string)

if __name__ == '__main__':
    main()
