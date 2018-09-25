import tensorflow as tf

MODEL_SAVE_PATH = 'model'
saver = tf.train.import_meta_graph(MODEL_SAVE_PATH + '/model.ckpt-32351.meta')
with tf.Session() as sess:
    saver.restore(sess, "model/model.ckpt-32351")
    graph = tf.get_default_graph()
    if tf.gfile.Exists("tmp/logs"):
        tf.gfile.DeleteRecursively("tmp/logs")
    summary_writer = tf.summary.FileWriter('tmp/logs', sess.graph)