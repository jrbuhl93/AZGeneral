import Arena
from MCTS import MCTS

from connect4.Connect4Game import Connect4Game
from connect4.Connect4Players import *
from connect4.tensorflow_resnet.NNet import NNetWrapper as NNet

import numpy as np
from utils import *

human_vs_cpu = True

g = Connect4Game()

# all players
rp = RandomPlayer(g).play
hp = HumanConnect4Player(g).play
oslap = OneStepLookaheadConnect4Player(g).play

if human_vs_cpu:
    player2 = hp
else:
    n1 = NNet(g)
    n1.load_checkpoint('./temp/', 'best')
    args1 = dotdict({'numMCTSSims': 120, 'cpuct': 1.25})
    mcts1 = MCTS(g, n1, args1)
    n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

    # player1 = n1p  # Player 2 is neural network if it's cpu vs cpu.

    # player1 = oslap

# nnet players
n2 = NNet(g)
n2.load_checkpoint('./temp/', 'best')
args2 = dotdict({'numMCTSSims': 120, 'cpuct': 1.25})
mcts2 = MCTS(g, n2, args2, verbose=True)
n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

player1=n2p

# Not Parallelized Arena
arena = Arena.Arena(player1, player2, g, display=Connect4Game.display)
print(arena.playGames(2, verbose=True))


# Ray Arena

# rayArena = Arena.RayArena(11, 12)
# print(rayArena.playGames(100, verbose=True))