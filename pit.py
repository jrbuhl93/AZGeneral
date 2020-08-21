import Arena
from MCTS import MCTS

# from connect4.Connect4Game import Connect4Game
# from connect4.Connect4Players import *
# from connect4.tensorflow_resnet.NNet import NNetWrapper as NNet

from santorini.SantoriniGame import SantoriniGame
from santorini.SantoriniPlayers import *
from santorini.tensorflow.NNet import NNetWrapper as NNet

import numpy as np
from utils import *

human_vs_cpu = False

# g = Connect4Game()
g = SantoriniGame()

# all players
rp = RandomSantoriniPlayer(g).play
# hp = HumanConnect4Player(g).play
# oslap = OneStepLookaheadConnect4Player(g).play

hp = HumanSantoriniPlayer(g).play

# nnet players
# n1 = NNet(g)
# n1.load_checkpoint('./temp/', 'best')
# args1 = dotdict({'numMCTSSims': 120, 'cpuct': 1.25})
# mcts1 = MCTS(g, n1, args1, verbose=True)
# n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

# player1=n1p

player1 = rp

if human_vs_cpu:
    player2 = hp
else:
    # n2 = NNet(g)
    # n2.load_checkpoint('./temp/', 'best')
    # args2 = dotdict({'numMCTSSims': 120, 'cpuct': 1.25})
    # mcts2 = MCTS(g, n2, args2, verbose=True)
    # n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

    # player2 = n2p  # Player 2 is neural network if it's cpu vs cpu.

    # player2 = oslap
    player2 = rp

# Not Parallelized Arena
arena = Arena.Arena(player1, player2, g, display=SantoriniGame.display)
print(arena.playGames(2, verbose=True))


# Ray Arena

# rayArena = Arena.RayArena(11, 12)
# print(rayArena.playGames(100, verbose=True))