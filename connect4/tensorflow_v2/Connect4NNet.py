import tensorflow as tf

from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Dense, Dropout
from tensorflow.keras import Model, regularizers

class Connect4NNet(tf.keras.Model):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        super(Connect4NNet, self).__init__()
        self.conv1 = Conv2D(args.num_channels, 3, strides=1, padding='same', kernel_regularizer=regularizers.l2(.0001))
        self.conv2 = Conv2D(args.num_channels, 3, strides=1, padding='same', kernel_regularizer=regularizers.l2(.0001))
        self.conv3 = Conv2D(args.num_channels, 3, strides=1, padding='valid', kernel_regularizer=regularizers.l2(.0001))
        self.conv4 = Conv2D(args.num_channels, 3, strides=1, padding='valid', kernel_regularizer=regularizers.l2(.0001))

        self.bn1 = BatchNormalization(axis=3)
        self.bn2 = BatchNormalization(axis=3)
        self.bn3 = BatchNormalization(axis=3)
        self.bn4 = BatchNormalization(axis=3)

        self.fc1 = Dense(1024, kernel_regularizer=regularizers.l2(.0001))
        self.fc_bn1 = BatchNormalization(axis=1)

        self.fc2 = Dense(512, kernel_regularizer=regularizers.l2(.0001))
        self.fc_bn2 = BatchNormalization(axis=1)

        self.fc3 = Dense(self.action_size, kernel_regularizer=regularizers.l2(.0001))

        self.fc4 = Dense(1, kernel_regularizer=regularizers.l2(.0001))

    def call(self, s, training=False):
        #                                                           s: batch_size x board_x x board_y
        s = tf.reshape(s, shape=(-1, self.board_x, self.board_y, 1)) # batch_size x 1 x board_x x board_y
        s = ReLU()(self.bn1(self.conv1(s)))                          # batch_size x num_channels x board_x x board_y
        s = ReLU()(self.bn2(self.conv2(s)))                          # batch_size x num_channels x board_x x board_y
        s = ReLU()(self.bn3(self.conv3(s)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = ReLU()(self.bn4(self.conv4(s)))                          # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = tf.reshape(s, shape=(-1, self.args.num_channels * (self.board_x - 4) * (self.board_y - 4)))

        s = ReLU()(self.fc_bn1(self.fc1(s)))  # batch_size x 1024
        # if training:
        #     s = Dropout(self.args.dropout)(s)
        s = ReLU()(self.fc_bn2(self.fc2(s)))  # batch_size x 512
        # if training:
        #     s = Dropout(self.args.dropout)(s)

        pi = self.fc3(s)                                                                         # batch_size x action_size
        v = self.fc4(s)                                                                          # batch_size x 1

        return tf.nn.log_softmax(pi, axis=1), tf.keras.activations.tanh(v)