import sys
import numpy as np

sys.path.append('..')
from Game import Game
from .SantoriniLogic import Board

class SantoriniGame(Game):
    """
    Santorini Game class implementing the alpha-zero-general Game interface.
    """

    game_state = 'addingPieces'
    selected_piece = None

    def __init__(self, size=5):
        Game.__init__(self)
        self._board = Board(size)

    def getInitBoard(self):
        return self._board.pieces

    def getActionSize(self):
        return (self._board.n * self._board.n) + 1

    def getNextState(self, board, player, action):
        b = Board(len(board))
        b.pieces = np.copy(board)
        move = (int(action/b.n), action%b.n)

        print(f'Player: {player}')
        print(f'Game state: {self.game_state}')
        print(f'Mover: {move}')

        if self.game_state == 'addingPieces':
            b.add_piece(move, player)
            if b.has_placed_all_pieces():
                self.game_state = 'selectingPiece'
            return (b.pieces, -player)
        elif self.game_state == 'selectingPiece':
            self.selected_piece = move
            self.game_state = 'movingPiece'
            return (b.pieces, player)
        elif self.game_state == 'movingPiece':
            b.move_piece(move, self.selected_piece, player)
            self.selected_piece = move
            self.game_state = 'building'
            return (b.pieces, player)
        elif self.game_state == 'building':
            b.build(move)
            self.game_state = 'selectingPiece'
            return (b.pieces, -player)

        return None


    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        valids = [0]*self.getActionSize()
        b = Board(len(board))
        b.pieces = np.copy(board)

        if self.game_state == 'addingPieces':
            legal_moves = b.get_legal_add_stone_moves(player)
        elif self.game_state == 'selectingPiece':
            legal_moves = b.get_legal_select_piece_moves(player)
        elif self.game_state == 'movingPiece':
            legal_moves = b.get_legal_movement_moves(player, self.selected_piece)
        elif self.game_state == 'building':
            legal_moves = b.get_legal_build_moves(player, self.selected_piece)

        if len(legal_moves)==0:
            valids[-1]=1
            return np.array(valids)
        for x, y in legal_moves:
            valids[b.n*x+y]=1
        return np.array(valids)

    def getGameEnded(self, board, curPlayer):
        b = Board(len(board))
        b.pieces = np.copy(board)
        winstate = b.get_win_state(curPlayer)
        if winstate.is_ended:
            self.game_state = 'addingPieces'
            if winstate.winner is None:
                # draw has very little value.
                return 1e-4
            elif winstate.winner == curPlayer:
                return +1
            elif winstate.winner == -curPlayer:
                return -1
            else:
                raise ValueError('Unexpected winstate found: ', winstate)
        else:
            # 0 used to represent unfinished game.
            return 0

    def getCanonicalForm(self, board, player):
        n = len(board)
        canonical_board = np.copy(board)
        if player == -1:
            for x in range(n):
                for y in range(n):
                    if board[x][y][0] == 1:
                        canonical_board[x][y][0] = 0
                        canonical_board[x][y][2] = 1

                    if board[x][y][1] == 1:
                        canonical_board[x][y][1] = 0
                        canonical_board[x][y][3] = 1

                    if board[x][y][2] == 1:
                        canonical_board[x][y][2] = 0
                        canonical_board[x][y][0] = 1

                    if board[x][y][3] == 1:
                        canonical_board[x][y][3] = 0
                        canonical_board[x][y][1] = 1

        game_state_array = [0] * 50
        if self.selected_piece is not None:
            x, y = self.selected_piece
            if self.game_state == 'movingPiece':
                game_state_array[n*x+y] = 1
            elif self.game_state == 'building':
                game_state_array[n*x+y+25] = 1

        full_canonical_board = []
        for x in range(n):
            full_canonical_row = []
            for y in range(n):
                full_canonical_row.append(np.concatenate((canonical_board[x][y], game_state_array), axis=0))
            full_canonical_board.append(full_canonical_row)

        return full_canonical_board

    @staticmethod
    def display(board):
        n = len(board)
        print("   ", end="")
        for y in range(n):
            print(" ", end="")
            print(y, end="  ")
        print("")
        print("-----------------------")
        for x in range(n):
            print(x, "|", end="")    # print the row #
            for y in range(n):
                square_value = board[x][y]    # get the square value to print
                print(SantoriniGame.get_square_content(square_value), end=" ")
            print("|")

        print("-----------------------")

    @staticmethod
    def get_square_content(square_value):
        content = '--'
        if square_value[0] == 1:
            content = 'M1'

        if square_value[1] == 1:
            content = 'F1'
        
        if square_value[2] == 1:
            content = 'M2'

        if square_value[3] == 1:
            content = 'F2'

        if square_value[4] == 1:
            content = content + '0'

        if square_value[5] == 1:
            content = content + '1'

        if square_value[6] == 1:
            content = content + '2'

        if square_value[7] == 1:
            content = content + '3'

        if square_value[8] == 1:
            content = content + '4'

        return content