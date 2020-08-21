from collections import namedtuple
import numpy as np

WinState = namedtuple('WinState', 'is_ended winner')

class Board():

    def __init__(self, n=5):
        # Set up initial board configuration

        self.n = n
        # Create the empty board array
        self.pieces = [None]*self.n
        for i in range(self.n):
            square = np.zeros(11)
            square[4] = 1
            self.pieces[i] = [square]*self.n  # P1M P1F P2M P2F     0H 1H 2H 3H 4H

    # add [][] indexer syntax to the Board
    def __getitem__(self, index): 
        return self.pieces[index]

    def add_piece(self, move, curPlayer):
        x, y = move
        for idx in range(4):
            if self[x][y][idx] == 1:
                print("Very bad")
                return

        if curPlayer == 1:
            if self._number_of_placed_pieces() < 2:
                self[x][y][0] = 1
            else:
                self[x][y][1] = 1
        elif curPlayer == -1:
            if self._number_of_placed_pieces() < 2:
                self[x][y][2] = 1
            else:
                self[x][y][3] = 1

    def select_piece(self, move):
        mx, my = move

        self[mx][my][9] = 1

    def move_piece(self, move, curPlayer):
        ox, oy = self._get_selected_piece()
        mx, my = move

        if curPlayer == 1:
            if self[ox][oy][0] == 1:
                self[ox][oy][0] = 0
                self[mx][my][0] = 1
            else:
                self[ox][oy][1] = 0
                self[mx][my][1] = 1
        elif curPlayer == -1:
            if self[ox][oy][2] == 1:
                self[ox][oy][2] = 0
                self[mx][my][2] = 1
            else:
                self[ox][oy][3] = 0
                self[mx][my][3] = 1

        self[ox][oy][9] = 0
        self[mx][my][10] = 1

    def build(self, move):
        x, y = move
        square_height = self._get_square_height(move)

        self[x][y][square_height + 4] = 0
        self[x][y][square_height + 5] = 1

        selected_piece = self._get_selected_piece()

        if selected_piece is not None:
            sx, sy = selected_piece
            self[sx][sy][10] = 0

    def get_win_state(self, curPlayer):
        for player in [-1, 1]:
            piece_squares = self.get_piece_squares_for_player(player)
            for piece_square in piece_squares:
                x, y = piece_square
                if self[x][y][7] == 1:
                    return WinState(True, player)

        # no moves means loss
        if len(self.get_legal_moves(curPlayer)) == 0:
            return WinState(True, -curPlayer)

        # Game is not ended yet.
        return WinState(False, None)

    def get_legal_add_stone_moves(self, curPlayer):
        moves = set()

        if not self.has_placed_all_pieces():
            for x in range(self.n):
                for y in range(self.n):
                    if not self._is_piece_present((x, y)):
                        moves.update([(x,y)])

        return moves

    def get_legal_select_piece_moves(self, curPlayer):
        moves = set()

        for x in range(self.n):
            for y in range(self.n):
                if curPlayer == 1:
                    if self[x][y][0] == 1 or self[x][y][1] == 1:
                        moves.update([(x,y)])
                elif curPlayer == -1:
                    if self[x][y][2] == 1 or self[x][y][3] == 1:
                        moves.update([(x,y)])

        moves = self.filter_select_piece_by_valid_moves(curPlayer, moves)

        return moves

    def get_legal_movement_moves(self, curPlayer):
        selected_piece = self._get_selected_piece()

        return self._get_legal_movement_moves_for_piece(selected_piece)

    def _get_legal_movement_moves_for_piece(self, selectedPiece):
        moves = set()

        adjacent_squares = self._get_adjacent_squares(selectedPiece)

        origin_height = self._get_square_height(selectedPiece)
        moves = self.filter_moves_by_height(adjacent_squares, origin_height)
        moves = self.filter_moves_by_piece_presence(moves)

        return moves


    def get_legal_build_moves(self, curPlayer):
        selected_piece = self._get_selected_piece()

        moves = set()

        adjacent_squares = self._get_adjacent_squares(selected_piece)

        moves = self.filter_moves_by_piece_presence(adjacent_squares)
        moves = self.filter_moves_by_build_height(moves)

        return moves

    def get_legal_moves(self, player):
        selected_piece = self._get_selected_piece()

        if selected_piece is not None:
            sx, sy = selected_piece
            if self[sx][sy][9] == 1:
                return self.get_legal_movement_moves(player)
            elif self[sx][sy][10] == 1:
                return self.get_legal_build_moves(player)

        if self.has_placed_all_pieces():
            return self.get_legal_select_piece_moves(player)
        else:
            return self.get_legal_add_stone_moves(player)

    def _get_indices(self, player):
        switcher = {
            1: (0, 1),
            -1: (2,3)
        }

        return switcher.get(player, ())

    def has_placed_all_pieces(self):
        pieces_count = self._number_of_placed_pieces()
        return (pieces_count >= 4)

    def _number_of_placed_pieces(self):
        pieces_count = 0
        
        for x in range(self.n):
            for y in range(self.n):
                for index in range(4):
                    if self[x][y][index] == 1:
                        pieces_count += 1

        return pieces_count

    def get_piece_squares_for_player(self, player):
        piece_squares = []
        player_indicies = self._get_indices(player)
        for x in range(self.n):
            for y in range(self.n):
                for index in player_indicies:
                    if self[x][y][index] == 1:
                        piece_squares.append((x, y))
        return piece_squares

    def get_moves_for_square(self, square):
        """
        Returns all the legal moves that use the given square as a base.
        """

        adjacent_squares = self._get_adjacent_squares(square)

        origin_height = self._get_square_height(square)
        moves = self.filter_moves_by_height(adjacent_squares, origin_height)
        moves = self.filter_moves_by_piece_presence(moves)
        # moves = self.filter_moves_by_buildable_adjacent(moves)

        # return the generated move list
        return moves

    def filter_moves_by_height(self, moves, origin_height):
        filtered_moves = []
        for move in moves:
            move_height = self._get_square_height(move)
            if move_height == 4:
                continue
            
            if move_height > (origin_height + 1):
                continue

            filtered_moves.append(move)
        return filtered_moves

    def filter_moves_by_piece_presence(self, moves):
        filtered_moves = []
        for move in moves:
            if self._is_piece_present(move):
                continue
            filtered_moves.append(move)
        return filtered_moves

    def filter_moves_by_build_height(self, moves):
        filtered_moves = []
        for move in moves:
            move_height = self._get_square_height(move)
            if move_height == 4:
                continue

            filtered_moves.append(move)
        return filtered_moves

    def filter_select_piece_by_valid_moves(self, curPlayer, moves):
        filtered_moves = []
        for move in moves:
            legal_movement_moves = self._get_legal_movement_moves_for_piece(move)
            if len(legal_movement_moves) > 0:
                filtered_moves.append(move)

        return filtered_moves

    # def filter_moves_by_buildable_adjacent(self, moves, origin):

    def _get_adjacent_squares(self, square):
        directions = [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1)]

        x, y = square

        adjacent_squares = []

        for direction in directions:
            dx, dy = direction

            ax, ay = (x + dx, y + dy)

            if (ax < 0) or (ax >= self.n):
                continue

            if (ay < 0) or (ay >= self.n):
                continue

            adjacent_squares.append((ax,ay))

        return adjacent_squares

    def _get_square_height(self, square):
        x, y = square
        for index in range(4,9):
            if self[x][y][index] == 1:
                return index - 4

        return 0

    def _is_piece_present(self, square):
        x, y = square
        for index in range(4):
            if self[x][y][index] == 1:
                return True
        return False

    def _get_selected_piece(self):
        for x in range(self.n):
            for y in range(self.n):
                if self[x][y][9] == 1:
                    return (x,y)

                if self[x][y][10] == 1:
                    return [x,y]
        
        return None