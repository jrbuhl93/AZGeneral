import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time

import ray
from MCTS import MCTS
from connect4.Connect4Game import Connect4Game
from connect4.Connect4Players import *
from connect4.tensorflow_v2.NNet import NNetWrapper as NNet
from utils import *


class RayArena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """
    def __init__(self, player_1_code=0, player_2_code=0):
        """
        Input:
            player 1,2: code for determining player function to instantiate
        """
        self.player_1_code = player_1_code
        self.player_2_code = player_2_code

    @ray.remote
    def playGame(self, player_1_code=0, player_2_code=0, verbose=False):
        """
        Executes one episode of a game.
        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """

        # Intialize Game
        game = Connect4Game()
        display = game.display
        
        mcts_args = dotdict({'numMCTSSims': 50, 'cpuct':1.0})

        if player_1_code == 0:
            player1 = RandomPlayer(game).play
        elif player_1_code == 1:
            player1 = OneStepLookaheadConnect4Player(game).play
        elif player_1_code == 10:
            n1 = NNet(game)
            n1.load_checkpoint('./temp/','best')
            mcts1 = MCTS(game, n1, mcts_args)
            player1 = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))
        elif player_1_code == 11:
            n1 = NNet(game)
            n1.load_checkpoint('./temp/','temp')
            mcts1 = MCTS(game, n1, mcts_args)
            player1 = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))
        elif player_1_code == 12:
            n1 = NNet(game)
            n1.load_checkpoint('./temp/','trained_temp')
            mcts1 = MCTS(game, n1, mcts_args)
            player1 = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))
        else:
            player1 = RandomPlayer(game).play

        if player_2_code == 0:
            player2 = RandomPlayer(game).play
        elif player_2_code == 1:
            player2 = OneStepLookaheadConnect4Player(game).play
        elif player_2_code == 10:
            n2 = NNet(game)
            n2.load_checkpoint('./temp/','best')
            mcts2 = MCTS(game, n2, mcts_args)
            player2 = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))
        elif player_2_code == 11:
            n2 = NNet(game)
            n2.load_checkpoint('./temp/','temp')
            mcts2 = MCTS(game, n2, mcts_args)
            player2 = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))
        elif player_2_code == 12:
            n2 = NNet(game)
            n2.load_checkpoint('./temp/','trained_temp')
            mcts2 = MCTS(game, n2, mcts_args)
            player2 = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))
        else:
            player2 = RandomPlayer(game).play

        

        players = [player2, None, player1]
        curPlayer = 1
        board = game.getInitBoard()
        it = 0
        while game.getGameEnded(board, curPlayer)==0:
            it+=1
            if verbose:
                assert(display)
                print("Turn ", str(it), "Player ", str(curPlayer))
                display(board)
            action = players[curPlayer+1](game.getCanonicalForm(board, curPlayer))

            valids = game.getValidMoves(game.getCanonicalForm(board, curPlayer),1)

            if valids[action]==0:
                print(action)
                assert valids[action] >0
            board, curPlayer = game.getNextState(board, curPlayer, action)
        if verbose:
            assert(display)
            print("Game over: Turn ", str(it), "Result ", str(game.getGameEnded(board, 1)))
            display(board)
        return curPlayer*game.getGameEnded(board, curPlayer)

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """
        eps_time = AverageMeter()
        bar = Bar('Arena.playGames', max=num)
        end = time.time()
        eps = 0
        maxeps = int(num)

        num = int(num/2)
        oneWon = 0
        twoWon = 0
        draws = 0

        if not ray.is_initialized():
            ray.init(num_cpus=5)

        gameObjects = [self.playGame.remote(self, player_1_code=self.player_1_code, player_2_code=self.player_2_code, verbose=verbose) for _ in range(num)]
        gameResults = ray.get(gameObjects)

        for gameResult in gameResults:
            if gameResult==1:
                oneWon+=1
            elif gameResult==-1:
                twoWon+=1
            else:
                draws+=1
            # bookkeeping + plot progress
            eps += 1
            eps_time.update(time.time() - end)
        end = time.time()
        bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps, maxeps=maxeps, et=eps_time.avg,
                                                                                                    total=bar.elapsed_td, eta=bar.eta_td)
        bar.next()
        
        gameObjects = [self.playGame.remote(self, player_1_code=self.player_2_code, player_2_code=self.player_1_code, verbose=verbose) for _ in range(num)]
        gameResults = ray.get(gameObjects)
        
        for gameResult in gameResults:
            if gameResult==-1:
                oneWon+=1                
            elif gameResult==1:
                twoWon+=1
            else:
                draws+=1
            # bookkeeping + plot progress
            eps += 1
            eps_time.update(time.time() - end)
        end = time.time()
        bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps, maxeps=maxeps, et=eps_time.avg,
                                                                                                    total=bar.elapsed_td, eta=bar.eta_td)
        bar.next()
            
        bar.finish()

        ray.shutdown()

        return oneWon, twoWon, draws