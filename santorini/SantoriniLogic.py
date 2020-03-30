class Board():
    
    # list of all 8 directions on the board, as (x,y) offsets
    __directions = [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1)]

    def __init__(self, n=5):
        # Set up initial board configuration

        self.n = n
        # Create the empty board array
        self.pieces = [None]*self.n
        for i in range(self.n):
            self.pieces[i] = [[0,0,0,0, 0,0,0,0,0]]*self.n

    # add [][] indexer syntax to the Board
    def __getitem__(self, index): 
        return self.pieces[index]

    def get_legal_moves(self, color):
        # Returns all legal moves for the given color.
        # 1 for white, -1 for grey
        moves = set()

        player_indicies = self._get_indices(color)
        for y in range(self.n):
            for x in range(self.n):
                for index in player_indicies:
                    if self[x][y] == 1:
                        newmoves = self.get_moves_for_square(())

    def _get_indices(self, color):
        switcher = {
            1: (0, 1),
            -1: (2,3)
        }

        return switcher.get(color, ())

    def has_legal_moves(self, color):
        return len(self.get_legal_moves(color)) > 0