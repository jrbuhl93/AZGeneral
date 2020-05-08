from Coach import Coach
from connect4.Connect4Game import Connect4Game as Game
from utils import *

args = dotdict({
    'numIters': 30,
    'numEps': 120,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 10,        #
    'updateThreshold': 0.55,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 150,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 10,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 4,

    'checkpoint': './temp/',
    'checkpoint_filename': 'best',
    'load_model': True,
    'load_folder_file': ('./temp/','best'),
    'numItersForTrainExamplesHistory': 20,

})

if __name__ == "__main__":
    g = Game()
    # nnet = nn(g)

    # if args.load_model:
    #     nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()