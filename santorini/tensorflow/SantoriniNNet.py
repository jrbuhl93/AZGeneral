import tensorflow as tf

from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, ReLU, Dense, Concatenate
from tensorflow.keras import Model, regularizers

class SantoriniNNet(Model):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        super(SantoriniNNet, self).__init__()
        self.convInput = Conv2D(args.num_channels, 3, strides=1, padding='same', kernel_regularizer=regularizers.l2(.0001))
        self.bnInput = BatchNormalization(axis=3)

        self.residualBlock1 = ResidualBlock(args)
        self.residualBlock2 = ResidualBlock(args)
        self.residualBlock3 = ResidualBlock(args)
        self.residualBlock4 = ResidualBlock(args)
        self.residualBlock5 = ResidualBlock(args)
        self.residualBlock6 = ResidualBlock(args)

        self.convPolicy = Conv2D(2, 1, strides=1, padding='same', kernel_regularizer=regularizers.l2(.0001))
        self.bnPolicy = BatchNormalization(axis=3)
        self.fcPolicy = Dense(self.action_size, kernel_regularizer=regularizers.l2(.0001))

        self.convValue = Conv2D(1, 1, strides=1, padding='same', kernel_regularizer=regularizers.l2(.0001))
        self.bnValue = BatchNormalization(axis=3)
        self.fcValue1 = Dense(256, kernel_regularizer=regularizers.l2(.0001))
        self.fcValue2 = Dense(1, kernel_regularizer=regularizers.l2(.0001))
    
    def call(self, s, training=False):
        #   s: batch_size x board_x x board_y

        #   batch_size x board_x x board_y x 2
        s = tf.reshape(s, shape=(-1, self.board_x, self.board_y, 59))

        s = ReLU()(self.bnInput(self.convInput(s), training))

        s = self.residualBlock1(s, training=training)
        s = self.residualBlock2(s, training=training)
        s = self.residualBlock3(s, training=training)
        s = self.residualBlock4(s, training=training)
        s = self.residualBlock5(s, training=training)
        s = self.residualBlock6(s, training=training)

        pi = ReLU()(self.bnPolicy(self.convPolicy(s), training=training))
        pi = tf.reshape(pi, shape=(-1, 2 * (self.board_x) * (self.board_y)))
        pi = self.fcPolicy(pi)

        v = ReLU()(self.bnValue(self.convValue(s), training=training))
        v = tf.reshape(v, shape=(-1, (self.board_x) * (self.board_y)))
        v = self.fcValue2(ReLU()(self.fcValue1(v)))

        return tf.nn.log_softmax(pi, axis=1), tf.keras.activations.tanh(v)

class ResidualBlock(Layer):
    def __init__(self, args):
        # game params
        self.args = args

        super(ResidualBlock, self).__init__()
        self.conv1 = Conv2D(args.num_channels, 3, strides=1, padding='same', kernel_regularizer=regularizers.l2(.0001))
        self.bn1 = BatchNormalization(axis=3)

        self.conv2 = Conv2D(args.num_channels, 3, strides=1, padding='same', kernel_regularizer=regularizers.l2(.0001))
        self.bn2 = BatchNormalization(axis=3)

    def call(self, inputs, training=False):
        s = ReLU()(self.bn1(self.conv1(inputs), training=training))
        s = self.bn2(self.conv2((s), training=training))
        s += inputs
        s = ReLU()(s)

        return s