#encoding:utf-8
import tensorflow as tf
from ae.AutoEncoder import main_unsupervised
from ae.AutoEncoder import main_supervised
from inference.config import *
from inference.utils import prepare_sae_inputs,get_current_input,tf_dot,maxpool,conv
flags = tf.app.flags
FLAGS = flags.FLAGS

class LSTMModel(object):
    def __init__(self, inputs,training,regularizer):
        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE, forget_bias=1.0, state_is_tuple=True)
            # return tf.contrib.rnn.GRUCell(HIDDEN_SIZE)
        attn_cell = lstm_cell
        if training is not None:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(lstm_cell(), input_keep_prob=KEEP_PROB, output_keep_prob=KEEP_PROB)
        cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(LAYERS_NUM)], state_is_tuple=True)
        self.initial_state = cell.zero_state(BATCH_SIZE, tf.float32)
        outputs = []
        state = self.initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(TIME_SERIES_STEP):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                cell_output, state = cell(inputs[:, :, time_step], state)
                outputs.append(cell_output)
        output = tf.reshape(outputs[-1], [-1, HIDDEN_SIZE])
        with tf.variable_scope("fullyConnect"):
            weight = tf.get_variable("weight", [HIDDEN_SIZE, INPUT_SIZE*INPUT_SIZE])
            bias = tf.get_variable("bias", [INPUT_SIZE*INPUT_SIZE])
        if regularizer != None: tf.add_to_collection('losses', regularizer(weight))
        logits = tf.matmul(output, weight) + bias
        self.mat = tf.reshape(logits, [-1, INPUT_SIZE, INPUT_SIZE])

class SAEModel(object):
    def __init__(self,sess,td_utils,ph_set,input_data,regularizer):
        input_dim = input_data.get_shape().as_list()[1]

        self.ae = main_unsupervised(sess, td_utils, ph_set, input_data, input_dim)
        self.mat = main_supervised(self.ae,input_data,regularizer)

class TRAFFICModel(object):
    def __init__(self,sess,td_utils,regularizer,initializer,adj_mx):

        self.is_training = tf.placeholder(tf.bool)    #'traffic_model/Placeholder:0'
        self.traffic_flow_input_data_lstm = tf.placeholder(tf.float32, [None, INPUT_SIZE, TIME_SERIES_STEP])
        self.traffic_flow_input_data_sae = tf.placeholder(tf.float32, [None, TIME_REQUIRE, INPUT_SIZE*TIME_SERIES_STEP])#'traffic_model/Placeholder_2:0'
        self.targets = tf.placeholder(tf.float32, [None, INPUT_SIZE])    #'traffic_model/Placeholder_3:0'
        self.learning_rate =  tf.placeholder(tf.float32, [])

        ph_set = [self.traffic_flow_input_data_sae,self.targets,self.is_training]

        # initializer = tf.random_uniform_initializer(-0.05, 0.05)

        sdae_inputs = prepare_sae_inputs(self.traffic_flow_input_data_sae)
        lstm_inputs = tf.reshape(sdae_inputs[2],[-1,INPUT_SIZE,TIME_SERIES_STEP])

        with tf.variable_scope("traffic_model_month", reuse=None, initializer=initializer):
            model1 = SAEModel(sess,td_utils,ph_set,sdae_inputs[0],regularizer)
        with tf.variable_scope("traffic_model_week", reuse=None, initializer=initializer):
            model2 = SAEModel(sess,td_utils,ph_set,sdae_inputs[1],regularizer)
        with tf.variable_scope("traffic_model_cur", reuse=None, initializer=initializer):
            model3 = SAEModel(sess,td_utils,ph_set,sdae_inputs[2],regularizer)
        with tf.variable_scope("traffic_model_lstm"):
            model = LSTMModel(lstm_inputs, self.is_training, regularizer)

        w1 = tf.get_variable("w1",[1],initializer=initializer)
        w2 = tf.get_variable("w2",[1],initializer=initializer)
        w3 = tf.get_variable("w3",[1], initializer=initializer)
        w4 = tf.get_variable("w4",[1], initializer=initializer)
        w5 = tf.get_variable("w5",[1], initializer=initializer)
        nor_w = [w1,w2,w3,w4,w5]

        adj_mx = tf.reshape(adj_mx, [1, INPUT_SIZE, INPUT_SIZE])
        conv1 = conv("conv1", adj_mx, INPUT_SIZE, trainable=False)
        conv2 = conv("conv2", conv1, INPUT_SIZE, trainable=False)
        conv2 = tf.reshape(conv2, [1,1,INPUT_SIZE,INPUT_SIZE])
        pool1 = maxpool("pool1", conv2, trainable=False)
        pool1 = tf.reshape(pool1, [INPUT_SIZE, INPUT_SIZE])

        tempMat = tf.add(tf.multiply(model1.mat, nor_w[0]), tf.multiply(model2.mat, nor_w[1]))
        finalMat = tf.add(tf.multiply(model3.mat, nor_w[2]), tempMat)
        finalMat  = tf.add(tf.multiply(model.mat, nor_w[3]), finalMat)

        pool1 = tf.multiply(nor_w[4], pool1)

        finalMat = tf.multiply(pool1, finalMat)

        bias = tf.get_variable("bias", [INPUT_SIZE])
        cur_flow_input = get_current_input(self.traffic_flow_input_data_sae)
        convertM = tf.reshape(cur_flow_input, [BATCH_SIZE, 1, INPUT_SIZE])
        #finalMat = tf.reshape(finalMat, [BATCH_SIZE, -1])
        predict = tf_dot(convertM, finalMat) + bias
        #predict = tf.matmul(convertM, finalMat) + bias
        self.predict = tf.reshape(predict, [BATCH_SIZE, INPUT_SIZE])    #'traffic_model/Reshape_1:0'

        reg_loss = tf.losses.get_regularization_loss()
        self.loss = tf.reduce_mean(tf.square(self.predict - self.targets))
        self.train_loss = self.loss+reg_loss
        if self.is_training is None:
            return
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        # optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        self.train_op = optimizer.minimize(self.train_loss)
