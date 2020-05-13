import os
import tensorflow as tf
from classification.methods.rnn.Parameters_rnn import parameters as pm
from classification.methods.rnn.data_processing_rnn import read_category, get_wordid, get_word2vec, process, batch_iter, sequence
from classification.methods.rnn.rnn import RnnModel
from text_classification.settings import CHECKPOINTS,MEDIA_ROOT



def train(model, pm, wordid,cat_to_id,dataid):
    tensorboard_dir = os.path.join(MEDIA_ROOT, 'tensorboard', 'text_rnn', make_dir_string(dataid, pm))
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
        print('Epoch:', epoch+1)
        num_batchs = int((len(x_train) - 1) / pm.batch_size) + 1
        batch_train = batch_iter(x_train, y_train, batch_size=pm.batch_size)
        for x_batch, y_batch in batch_train:
            seq_len = sequence(x_batch)
            feed_dict = model.feed_data(x_batch, y_batch, seq_len, pm.keep_prob)
            _, global_step, _summary, train_loss, train_accuracy = session.run([model.optimizer, model.global_step, merged_summary,
                                                                                model.loss, model.accuracy],feed_dict=feed_dict)
            if global_step % 100 == 0:
                test_loss, test_accuracy = model.evaluate(session, x_test, y_test)
                print('global_step:', global_step, 'train_loss:', train_loss, 'train_accuracy:', train_accuracy,
                      'test_loss:', test_loss, 'test_accuracy:', test_accuracy)

            if global_step % num_batchs == 0:
                print('Saving Model...')
                saver.save(session, save_path, global_step=global_step)

        pm.learning_rate *= pm.lr_decay

def val(model,pm,wordid,cat_to_id,data_id):
    pre_label = []  # 预测值
    label = [] # 真实值
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    save_path = tf.train.latest_checkpoint(os.path.join(CHECKPOINTS,'text_rnn',make_dir_string(data_id, pm)))   #os.path.join(MEDIA_ROOT,'checkpoints','text_cnn',make_dir_string(data_id, pm))
    saver = tf.train.Saver()
    flag = os.path.exists(save_path)
    saver.restore(sess=session, save_path=save_path)

    val_x, val_y = process(pm.val_filename, wordid, cat_to_id, max_length=600)
    batch_val = batch_iter(val_x, val_y, batch_size=64)
    for x_batch, y_batch in batch_val:
        pre_lab = session.run(model.predicitions, feed_dict={model.input_x: x_batch,
                                                             model.keep_pro: 1.0})
        pre_label.extend(pre_lab)
        label.extend(y_batch)
    return pre_label, label
if __name__ == '__main__':

    pm = pm
    filenames = [pm.train_filename, pm.test_filename, pm.val_filename]
    categories, cat_to_id = read_category()
    wordid = get_wordid(pm.vocab_filename)
    pm.vocab_size = len(wordid)
    pm.pre_trianing = get_word2vec(pm.vector_word_npz)

    model = RnnModel()

    train()
