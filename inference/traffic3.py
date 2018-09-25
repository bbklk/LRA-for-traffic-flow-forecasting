#encoding:utf-8
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),"..")))
import tensorflow as tf
import numpy as np
from inference.model import TRAFFICModel
from inference.datagenerator import BatchDataGenerator
from inference.utils import get_batch, get_batch_lstm,load_scaler,prepare_feed_data
from sklearn.metrics import mean_absolute_error
from inference.config import *
#os.environ['CUDA_VISIBLE_DEVICES']='7'


def main(_):
    flow_scaler, date_scaler = load_scaler()  # flow_scaler, wea_scaler
    train_generator = BatchDataGenerator([TRAIN_RECORD_FILE], BATCH_SIZE, shuffle=True)
    val_generator = BatchDataGenerator([VAL_RECORD_FILE], BATCH_SIZE, shuffle=False)
    test_generator = BatchDataGenerator([TEST_RECORD_FILE], BATCH_SIZE, shuffle=False)

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    initializer = tf.random_uniform_initializer(-0.05,0.05)
    # initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    td_utils = [train_generator,flow_scaler,date_scaler]
    test_batch_num = TEST_SAMPLE_NUMS_FIFTEEN // BATCH_SIZE
    adj_mx = np.loadtxt(CSV_PATH, delimiter=",")
    adj_mx = tf.convert_to_tensor(adj_mx, dtype=tf.float32)

    #with tf.Session(config=tf.ConfigProto(device_count={'GPU':0})) as sess:
    with tf.Session() as sess:
        coord = tf.train.Coordinator() # to manage the threads
        threads = tf.train.start_queue_runners(sess=sess, coord=coord) # with this, the input data is put in the memory
        with tf.variable_scope("traffic_model", reuse=None, initializer=initializer):
            train_model = TRAFFICModel(sess,td_utils, regularizer=regularizer,initializer=initializer,adj_mx=adj_mx)

        uninitialized_vars = []
        for var in tf.global_variables():
            try:
                sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninitialized_vars.append(var)
        init_new_vars_op = tf.variables_initializer(uninitialized_vars)
        sess.run(init_new_vars_op)
        saver = tf.train.Saver()
        # 以下待调试
        for i in range(TRAINING_STEPS):
            batch_data = get_batch(sess,train_generator,flow_scaler,date_scaler)
            # _, inputs_lstm, _ = get_batch_lstm(sess, train_generator, flow_scaler, date_scaler)
            date, traffic_input, targets = prepare_feed_data(batch_data)
            # date, traffic_input, targets = train_generator.next_batch_no_scale(sess)
            # 以下待调试
            loss,_ = sess.run([train_model.loss,train_model.train_op],feed_dict={train_model.traffic_flow_input_data_sae: traffic_input,
                                                                                 # train_model.traffic_flow_input_data_lstm: inputs_lstm,
                                                                                 train_model.targets: targets,
                                                                                 train_model.learning_rate:LEARNING_RATE,
                                                                                 train_model.is_training:True})
            if i % 10 == 0:
                #run summary
                print("After %d steps,train loss is %.3f" % (i,loss))
                # print("After %d steps,train loss_no_reg is %.3f" % (i,loss_no_reg))
            if i % 50 == 0:
                val_batch_data = get_batch(sess,val_generator,flow_scaler,date_scaler)
                # _, val_inputs, _ = get_batch_lstm(sess, val_generator, flow_scaler, date_scaler)
                val_date, val_traffic_input, val_targets = prepare_feed_data(val_batch_data)

                val_loss = sess.run(train_model.loss,feed_dict={train_model.traffic_flow_input_data_sae: val_traffic_input,
                                                                # train_model.traffic_flow_input_data_lstm: val_inputs,
                                                                train_model.targets: val_targets,
                                                                train_model.is_training:False})

                print("After %d steps,Validation loss is %.3f" % (i,val_loss))
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=i+1)

                loss_all = []
                for kk in range(test_batch_num):
                    test_batch_data = get_batch(sess, test_generator, flow_scaler, date_scaler)
                    # _, test_inputs, _ = get_batch_lstm(sess, test_generator, flow_scaler, date_scaler)
                    test_date, test_traffic_input, test_targets = prepare_feed_data(test_batch_data)
                    test_loss, test_predicts = sess.run([train_model.loss, train_model.predict],
                                                        feed_dict={train_model.traffic_flow_input_data_sae: test_traffic_input,
                                                                   # train_model.traffic_flow_input_data_lstm: test_inputs,
                                                                   train_model.targets: test_targets,
                                                                   train_model.is_training: False})

                    test_predicts_scale = flow_scaler.inverse_transform(test_predicts)
                    test_targets_scale = flow_scaler.inverse_transform(test_targets)

                    # test_predicts_scale = test_predicts
                    # test_targets_scale = test_targets
                    rmse_loss = np.sqrt(((test_predicts_scale - test_targets_scale) ** 2).mean())
                    mae_loss = mean_absolute_error(test_predicts_scale, test_targets_scale)
                    loss_all.append(mae_loss)

                model_test_loss = sum(loss_all) / len(loss_all)
                print('test loss is %.3f' % model_test_loss)

        coord.request_stop()
        coord.join(threads)


if __name__=='__main__':
    tf.app.run()
