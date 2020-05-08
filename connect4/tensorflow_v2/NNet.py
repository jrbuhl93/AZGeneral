import os
import time
import numpy as np
import sys
sys.path.append('../../')
from utils import *
from pytorch_classification.utils import Bar, AverageMeter
from NeuralNet import NeuralNet

import tensorflow as tf

from .Connect4NNet import Connect4NNet as cnnet

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'num_channels': 512,
})

class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = cnnet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            data_time = AverageMeter()
            batch_time = AverageMeter()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            l2_losses = AverageMeter()
            end = time.time()

            bar = Bar('Training Net', max=int(len(examples) / args.batch_size))
            batch_idx = 0

            while batch_idx < int(len(examples) / args.batch_size):
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = tf.convert_to_tensor(boards, dtype=tf.float32)
                target_pis = tf.Variable(np.array(pis))
                target_vs = tf.Variable(np.array(vs).astype(np.float32))
                # measure data loading time
                data_time.update(time.time() - end)

                # record loss
                pi_loss, v_loss, l2_loss = self.train_step(boards, target_pis, target_vs)
                board_count = tf.cast(tf.size(boards), dtype=tf.float32)
                pi_losses.update(pi_loss, board_count)
                v_losses.update(v_loss, board_count)
                l2_losses.update(l2_loss, board_count)
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                batch_idx += 1

                # plot progress
                bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss_pi: {lpi:.4f} | Loss_v: {lv:.3f} | Loss_l2 {ll2:.3f}'.format(
                            batch=batch_idx,
                            size=int(len(examples) / args.batch_size),
                            data=data_time.avg,
                            bt=batch_time.avg,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            lpi=pi_losses.avg,
                            lv=v_losses.avg,
                            ll2=l2_losses.avg,
                            )
                bar.next()
            bar.finish()           

    @tf.function
    def train_step(self, boards, target_pis, target_vs):
        with tf.GradientTape() as tape:
            # compute output
            out_pi, out_v = self.nnet(boards, training=True)
            l_pi = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(target_pis, out_pi)
            l_v = tf.keras.losses.MeanSquaredError()(target_vs, out_v)
            l_l2 = tf.add_n(self.nnet.losses)
            total_loss = l_pi + l_v + l_l2

        # compute gradient and do SGD step
        gradients = tape.gradient(total_loss, self.nnet.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.nnet.trainable_variables))

        return l_pi, l_v, l_l2

    @tf.function
    def predict(self, board):
        """
        board: np array with board
        """

        # timing
        start = time.time()

        # preparing input
        board = tf.reshape(board, shape=(1, self.board_x, self.board_y))
        pi, v = self.nnet(board, training=False)

        #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return tf.math.exp(pi)[0], v[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.nnet.save(filepath, save_format="tf")

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path {}".format(filepath))
        tf.keras.backend.clear_session()
        self.nnet = tf.keras.models.load_model(filepath)