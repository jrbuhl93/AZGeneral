import sys
import numpy as np

sys.path.append('..')
from Game import Game
from .Connect4Logic import Board


class Connect4Game(Game):
    """
    Connect4 Game class implementing the alpha-zero-general Game interface.
    """

    def __init__(self, height=None, width=None, win_length=None, np_pieces=None):
        Game.__init__(self)
        self._base_board = Board(height, width, win_length, np_pieces)

    def getInitBoard(self):
        return self._base_board.np_pieces

    def getBoardSize(self):
        return (self._base_board.height, self._base_board.width)

    def getActionSize(self):
        return self._base_board.width

    def getNextState(self, board, player, action):
        """Returns a copy of the board with updated move, original board is unmodified."""
        b = self._base_board.with_np_pieces(np_pieces=np.copy(board))
        b.add_stone(action, player)
        return b.np_pieces, -player

    def getValidMoves(self, board, player):
        "Any zero value in top row in a valid move"
        return self._base_board.with_np_pieces(np_pieces=board).get_valid_moves()

    def getGameEnded(self, board, player):
        b = self._base_board.with_np_pieces(np_pieces=board)
        winstate = b.get_win_state()
        if winstate.is_ended:
            if winstate.winner is None:
                # draw has very little value.
                return 1e-4
            elif winstate.winner == player:
                return +1
            elif winstate.winner == -player:
                return -1
            else:
                raise ValueError('Unexpected winstate found: ', winstate)
        else:
            # 0 used to represent unfinished game.
            return 0

    def getCanonicalForm(self, board, player):
        # Flip player from 1 to -1
        return board * player

    def getBoardInput(self, board):
        network_input = np.zeros([len(board), len(board[0]), 2])
        for m in range(len(board)):
            for n in range(len(board[m])):
                if board[m][n] == 1:
                    network_input[m][n][0] = 1
                elif board[m][n] == -1:
                    network_input[m][n][1] = 1
        return network_input

    def getSymmetries(self, board, pi):
        """Board is left/right board symmetric"""
        return [(self.getBoardInput(board), pi), (self.getBoardInput(board[:, ::-1]), pi[::-1])]

    def getRandomSymmetry(self, board):
        idx = np.random.choice(2,1)
        return [board, board[:, ::-1]][idx[0]]

    def stringRepresentation(self, board):
        return board.tostring()

    @staticmethod
    def display(board):
        m = board.shape[0]
        n = board.shape[1]

        print("   ", end="")
        for y in range(n):
            print (y,"", end="")
        print("")
        print("  ", end="")
        for _ in range(n):
            print ("-", end="-")
        print("--")
        for y in range(m):
            print(" ", "|",end="")
            for x in range(n):
                piece = board[y][x]    # get the piece to print
                if piece == -1: print("X ",end="")
                elif piece == 1: print("O ",end="")
                else:
                    if x==n:
                        print("-",end="")
                    else:
                        print("- ",end="")
            print("|")

        print("  ", end="")
        for _ in range(n):
            print ("-", end="-")
        print("--")


        # print(" -----------------------")
        # print(' '.join(map(str, range(len(board[0])))))
        # print(board)
        # print(" -----------------------")