import tensorflow as tf
import os
from text_classification.settings import *
from classification.methods.rnn.data_processing_rnn import batch_iter, sequence, process, process_text
import pandas as pd


class TextRnn(object):

    def __init__(self, pm):
        self.pm = pm
        self.input_x = tf.placeholder(tf.int32, shape=[None, pm.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, shape=[None, pm.num_classes], name='input_y')
        self.seq_length = tf.placeholder(tf.int32, shape=[None], name='sequen_length')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.rnn()

    def rnn(self):
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            embedding = tf.get_variable('embedding', shape=[self.pm.vocab_size, self.pm.embedding_dim],
                                        initializer=tf.constant_initializer(self.pm.pre_trianing))
            self.embedding_input = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope('cell'):
            cell = tf.nn.rnn_cell.LSTMCell(self.pm.hidden_dim)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

            cells = [cell for _ in range(self.pm.num_layers)]
            Cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

        with tf.name_scope('rnn'):
            # hidden一层 输入是[batch_size, seq_length, hidden_dim]
            # hidden二层 输入是[batch_size, seq_length, 2*hidden_dim]
            # 2*hidden_dim = embendding_dim + hidden_dim
            output, _ = tf.nn.dynamic_rnn(cell=Cell, inputs=self.embedding_input, sequence_length=self.seq_length,
                                          dtype=tf.float32)
            output = tf.reduce_sum(output, axis=1)
            # output:[batch_size, seq_length, hidden_dim]

        with tf.name_scope('dropout'):
            self.out_drop = tf.nn.dropout(output, keep_prob=self.keep_prob)

        with tf.name_scope('output'):
            w = tf.Variable(tf.truncated_normal([self.pm.hidden_dim, self.pm.num_classes], stddev=0.1), name='w')
            b = tf.Variable(tf.constant(0.1, shape=[self.pm.num_classes]), name='b')
            self.logits = tf.matmul(self.out_drop, w) + b
            self.predict = tf.argmax(tf.nn.softmax(self.logits), 1, name='predict')

        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.pm.learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))  # 计算变量梯度，得到梯度值,变量
            gradients, _ = tf.clip_by_global_norm(gradients, self.pm.clip)
            # 对g进行l2正则化计算，比较其与clip的值，如果l2后的值更大，让梯度*(clip/l2_g),得到新梯度
            self.optimizer = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)
            # global_step 自动+1

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(self.predict, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    def feed_data(self, x_batch, y_batch, seq_len, keep_prob):
        feed_dict = {self.input_x: x_batch,
                     self.input_y: y_batch,
                     self.seq_length: seq_len,
                     self.keep_prob: keep_prob}

        return feed_dict

    def evaluate(self, sess, x, y):
        batch_test = batch_iter(x, y, self.pm.batch_size)
        for x_batch, y_batch in batch_test:
            seq_len = sequence(x_batch)
            feet_dict = self.feed_data(x_batch, y_batch, seq_len, 1.0)
            loss, accuracy = sess.run([self.loss, self.accuracy], feed_dict=feet_dict)

        return loss, accuracy


def getImageUrl(dataid, pm):
    return "/static/csv2jpg/text_rnn/" + make_dir_string(dataid, pm) + "/train_info.png"


def make_dir_string(dataid, pm):
    '''创建文件夹的名称：数据集+参数的组合形式'''
    return ('dataid_' + str(dataid) +
            '_num_layers_' + str(pm.num_layers) + "_hidden_dim_" + str(pm.hidden_dim) +
            "_keep_prob_" + str(pm.keep_prob) + '_learning_rate_' + str(pm.learning_rate) + "_lr_decay_" + str(
                pm.lr_decay) + "_clip_" + str(pm.clip) + "_num_epochs_" + str(pm.num_epochs) +
            "_batch_size_" + str(pm.batch_size))


def save_info(file_path, train_info):
    dataFrame = pd.DataFrame(train_info)
    if not os.path.exists(file_path):
        with open(file_path, "a") as f:
            dataFrame.to_csv(f)
    else:
        with open(file_path, "a") as f:
            dataFrame.to_csv(f, header=False)


def train(model, pm, wordid, cat_to_id, dataid):
    tensorboard_dir = os.path.join(TENSORBOARD_DIR, 'text_rnn', make_dir_string(dataid, pm))
    save_dir = os.path.join(CHECKPOINTS, 'text_rnn', make_dir_string(dataid, pm))

    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'best_validation')

    tf.summary.scalar('loss', model.loss)
    tf.summary.scalar('accuracy', model.accuracy)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)
    saver = tf.train.Saver()
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    x_train, y_train = process(pm.train_filename, wordid, cat_to_id, max_length=250)
    x_test, y_test = process(pm.test_filename, wordid, cat_to_id, max_length=250)
    for epoch in range(pm.num_epochs):
        print('Epoch:', epoch + 1)
        num_batchs = int((len(x_train) - 1) / pm.batch_size) + 1
        batch_train = batch_iter(x_train, y_train, batch_size=pm.batch_size)

        # 保存信息为pandas
        train_info = {"global_step": [], "loss": [], "accuracy": []}  # 训练信息
        for x_batch, y_batch in batch_train:
            seq_len = sequence(x_batch)
            feed_dict = model.feed_data(x_batch, y_batch, seq_len, pm.keep_prob)
            _, global_step, _summary, train_loss, train_accuracy = session.run(
                [model.optimizer, model.global_step, merged_summary,
                 model.loss, model.accuracy], feed_dict=feed_dict)
            train_info["global_step"].append(global_step)
            train_info["loss"].append(train_loss)
            train_info["accuracy"].append(train_accuracy)
            if global_step % 100 == 0:
                test_loss, test_accuracy = model.evaluate(session, x_test, y_test)
                print('global_step:', global_step, 'train_loss:', train_loss, 'train_accuracy:', train_accuracy,
                      'test_loss:', test_loss, 'test_accuracy:', test_accuracy)

            if global_step % num_batchs == 0:
                print('Saving Model...')
                save_info(os.path.join(tensorboard_dir, "train_info.csv"), train_info)
                del train_info
                train_info = {"global_step": [], "loss": [], "accuracy": []}
                saver.save(session, save_path, global_step=global_step)

        pm.learning_rate *= pm.lr_decay


def val(model, pm, wordid, cat_to_id, data_id):
    pre_label = []
    label = []
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    save_path = tf.train.latest_checkpoint(os.path.join(CHECKPOINTS, 'text_rnn', make_dir_string(data_id,
                                                                                                 pm)))  # os.path.join(MEDIA_ROOT,'checkpoints','text_cnn',make_dir_string(data_id, pm))
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)

    val_x, val_y = process(pm.val_filename, wordid, cat_to_id, max_length=250)
    batch_val = batch_iter(val_x, val_y, batch_size=64)
    for x_batch, y_batch in batch_val:
        seq_len = sequence(x_batch)
        pre_lab = session.run(model.predict,
                              feed_dict={model.input_x: x_batch, model.seq_length: seq_len, model.keep_prob: 1.0})
        pre_label.extend(pre_lab)
        label.extend(y_batch)
    return pre_label, label


def val_text(model, text_data, pm, wordid, cat_to_id, data_id):
    pre_label = []  # 预测值

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    save_path = tf.train.latest_checkpoint(os.path.join(CHECKPOINTS, 'text_rnn', make_dir_string(data_id, pm)))  # os.path.join(MEDIA_ROOT,'checkpoints','text_cnn',make_dir_string(data_id, pm))
    saver = tf.train.Saver()
    flag = os.path.exists(save_path)
    saver.restore(sess=session, save_path=save_path)

    val_x = process_text(text_data, wordid, cat_to_id, max_length=250)
    seq_len = sequence(val_x)
    pre_lab = session.run(model.predict, feed_dict={model.input_x: val_x,model.seq_length: seq_len, model.keep_prob: 1.0})

    # 将预测结果展示
    return pre_lab[0]
