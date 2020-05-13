import tensorflow as tf
from classification.methods.cnn.data_processing import *
from classification.methods.cnn.Parameters import *
from text_classification.settings import CHECKPOINTS, TENSORBOARD_DIR
import pandas as pd


class TextCnn(object):

    def __init__(self, pm):
        # 需要往传pm
        self.pm = pm
        self.input_x = tf.placeholder(tf.int32, shape=[None, self.pm.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, shape=[None, self.pm.num_classes], name='input_y')
        self.keep_pro = tf.placeholder(tf.float32, name='drop_out')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.cnn()

    def cnn(self):
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.embedding = tf.get_variable("embeddings", shape=[self.pm.vocab_size, self.pm.embedding_size],
                                             initializer=tf.constant_initializer(self.pm.pre_trianing))
            embedding_input = tf.nn.embedding_lookup(self.embedding, self.input_x)
            self.embedding_expand = tf.expand_dims(embedding_input,
                                                   -1)  # [None,seq_length,embedding_size,1] [None,600,100,1]

        pooled_outputs = []
        for i, filter_size in enumerate(self.pm.kernel_size):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                filter_shape = [filter_size, self.pm.embedding_size, 1, self.pm.num_filters]  # [2,100,1,128]
                w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='w')  # 卷积核
                b = tf.Variable(tf.constant(0.1, shape=[self.pm.num_filters]), name='b')  # [128]
                conv = tf.nn.conv2d(self.embedding_expand, w, strides=[1, 1, 1, 1], padding='VALID',
                                    name='conv')  # [None,599,1,128]
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                pooled = tf.nn.max_pool(h, ksize=[1, self.pm.seq_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                                        padding='VALID', name='pool')  # 池化的参数很精妙   [None,1,1,128]
                pooled_outputs.append(pooled)

        num_filter_total = self.pm.num_filters * len(self.pm.kernel_size)  # 128 * 3
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filter_total])  # [None, 128 *3]

        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.keep_pro)

        with tf.name_scope('output'):
            w = tf.get_variable("w", shape=[num_filter_total, self.pm.num_classes],
                                initializer=tf.contrib.layers.xavier_initializer())

            b = tf.Variable(tf.constant(0.1, shape=[self.pm.num_classes]), name='b')

            self.scores = tf.nn.xw_plus_b(self.h_drop, w, b, name='scores')
            self.pro = tf.nn.softmax(self.scores)  # 最大为1，其余为0
            self.predicitions = tf.argmax(self.pro, 1, name='predictions')

        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)  # 对交叉熵取均值非常有必要

        with tf.name_scope('optimizer'):
            # 退化学习率 learning_rate = lr*(0.9**(global_step/10);staircase=True表示每decay_steps更新梯度
            # learning_rate = tf.train.exponential_decay(self.config.lr, global_step=self.global_step,
            # decay_steps=10, decay_rate=self.config.lr_decay, staircase=True)
            # optimizer = tf.train.AdadeltaOptimizer(learning_rate)
            # self.optimizer = optimizer.minimize(self.loss, global_step=self.global_step) #global_step 自动+1
            # no.2
            optimizer = tf.train.AdamOptimizer(self.pm.lr)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))  # 计算变量梯度，得到梯度值,变量
            gradients, _ = tf.clip_by_global_norm(gradients, self.pm.clip)
            # 对g进行l2正则化计算，比较其与clip的值，如果l2后的值更大，让梯度*(clip/l2_g),得到新梯度
            self.optimizer = optimizer.apply_gradients(zip(gradients, variables),
                                                       global_step=self.global_step)  # global_step 自动+1

        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predicitions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float32'), name='accuracy')

    def feed_data(self, x_batch, y_batch, keep_prob):
        feed_dict = {self.input_x: x_batch, self.input_y: y_batch, self.keep_pro: keep_prob}
        return feed_dict

    def evaluate(self, sess, x, y):
        batch_eva = batch_iter(x, y, self.pm.batch_size)
        for x_batch, y_batch in batch_eva:
            feed_dict = self.feed_data(x_batch, y_batch, 1.0)
            loss, accuracy = sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
        return loss, accuracy

def getImageUrl(dataid,pm):
    return "/static/csv2jpg/text_cnn/"+make_dir_string(dataid, pm)+"/train_info.png"

def make_dir_string(dataid, pm):
    '''创建文件夹的名称：数据集+参数的组合形式'''
    return ('dataid_' + str(dataid) + '_num_filters_' + str(pm.num_filters) + "_keep_prob_" + str(
        pm.keep_prob) + "_lr_" + str(pm.lr) + "_lr_decay_" + str(pm.lr_decay) + "_clip_" + str(
        pm.clip) + "_num_epochs_" + str(pm.num_epochs) + "_batch_size_" + str(pm.batch_size))


def save_info(file_path, train_info):
    dataFrame = pd.DataFrame(train_info)
    if not os.path.exists(file_path):
        with open(file_path, "a") as f:
            dataFrame.to_csv(f)
    else:
        with open(file_path, "a") as f:
            dataFrame.to_csv(f, header=False)


def train(model, pm, wordid, cat_to_id, dataid):
    '''model: 是cnn对象'''

    tensorboard_dir = os.path.join(TENSORBOARD_DIR, 'text_cnn', make_dir_string(dataid, pm))
    save_dir = os.path.join(CHECKPOINTS, 'text_cnn', make_dir_string(dataid, pm))
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, 'best_validation')  # 在这里构建

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.accuracy)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)
    saver = tf.train.Saver()
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print("Loading Training data...")
    x_train, y_train = process(pm.train_filename, wordid, cat_to_id, pm.seq_length)
    x_val, y_val = process(pm.val_filename, wordid, cat_to_id, pm.seq_length)
    for epoch in range(pm.num_epochs):
        print('Epoch:', epoch + 1)
        num_batchs = int((len(x_train) - 1) / pm.batch_size) + 1
        batch_train = batch_iter(x_train, y_train, pm.batch_size)

        # 保存信息为pandas
        train_info = {"global_step": [], "loss": [], "accuracy": []}  # 训练信息

        for x_batch, y_batch in batch_train:
            feed_dict = model.feed_data(x_batch, y_batch, pm.keep_prob)
            _, global_step, train_summary, train_loss, train_accuracy = session.run(
                [model.optimizer, model.global_step, merged_summary, model.loss, model.accuracy], feed_dict=feed_dict)
            train_info["global_step"].append(global_step)
            train_info["loss"].append(train_loss)
            train_info["accuracy"].append(train_accuracy)

            if global_step % 100 == 0:
                val_loss, val_accuracy = model.evaluate(session, x_val, y_val)
                print(global_step, train_loss, train_accuracy, val_loss, val_accuracy)

            if (global_step + 1) % num_batchs == 0:
                print("Saving model...")
                save_info(os.path.join(tensorboard_dir, "train_info.csv"), train_info)
                del train_info
                train_info = {"global_step": [], "loss": [], "accuracy": []}
                saver.save(session, save_path, global_step=global_step)

        pm.lr *= pm.lr_decay


def val(model, pm, wordid, cat_to_id, data_id):
    pre_label = []  # 预测值
    label = []  # 真实值
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    save_path = tf.train.latest_checkpoint(os.path.join(CHECKPOINTS, 'text_cnn', make_dir_string(data_id,
                                                                                                 pm)))  # os.path.join(MEDIA_ROOT,'checkpoints','text_cnn',make_dir_string(data_id, pm))
    saver = tf.train.Saver()
    flag = os.path.exists(save_path)
    saver.restore(sess=session, save_path=save_path)

    val_x, val_y = process(pm.val_filename, wordid, cat_to_id, max_length=600)
    batch_val = batch_iter(val_x, val_y, batch_size=64)
    for x_batch, y_batch in batch_val:
        pre_lab = session.run(model.predicitions, feed_dict={model.input_x: x_batch, model.keep_pro: 1.0})
        pre_label.extend(pre_lab)
        label.extend(y_batch)
    return pre_label, label




def val_text(model, text_data, pm, wordid, cat_to_id, data_id):
    pre_label = []  # 预测值

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    save_path = tf.train.latest_checkpoint(os.path.join(CHECKPOINTS, 'text_cnn', make_dir_string(data_id,
                                                                                                 pm)))  # os.path.join(MEDIA_ROOT,'checkpoints','text_cnn',make_dir_string(data_id, pm))
    saver = tf.train.Saver()
    flag = os.path.exists(save_path)
    saver.restore(sess=session, save_path=save_path)

    val_x = process_text(text_data, wordid, cat_to_id, max_length=600)

    pre_lab = session.run(model.predicitions, feed_dict={model.input_x: val_x, model.keep_pro: 1.0})

    # 将预测结果展示
    return pre_lab[0]
