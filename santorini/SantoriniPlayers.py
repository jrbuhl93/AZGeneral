import numpy as np

class RandomSantoriniPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a

class HumanSantoriniPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        n = self.game._board.n
        valid = self.game.getValidMoves(board, 1)
        for i in range(len(valid)):
            if valid[i]:
                print("[", int(i/n), int(i%n), end="] ")
        print("")
        while True:
            input_move = input()
            input_a = input_move.split(" ")
            if len(input_a) == 2:
                try:
                    x,y = [int(i) for i in input_a]
                    if ((0 <= x) and (x < n) and (0 <= y) and (y < n)) or \
                            ((x == n) and (y == 0)):
                        a = n * x + y if x != -1 else n ** 2
                        if valid[a]:
                            break
                except ValueError:
                    # Input needs to be an integer
                    'Invalid integer'
            print('Invalid move')
        return a