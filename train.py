import time
from load_data import *
from model import *
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


# 训练模型
def training():
    N_CLASSES = 2
    IMG_SIZE = 208
    BATCH_SIZE = 50
    CAPACITY = 200
    MAX_STEP = 30000
    LEARNING_RATE = 1e-4

    # 测试图片读取
    image_dir = r'train/'
    pre_logs = 'logs_1'
    logs_dir = 'logs_2'

    sess = tf.Session()



    train_list = get_all_files(image_dir, True)
    image_train_batch, label_train_batch = get_batch(train_list, IMG_SIZE, BATCH_SIZE, CAPACITY, True)
    train_logits = inference(image_train_batch, N_CLASSES)
    train_loss = losses(train_logits, label_train_batch)
    train_acc = evaluation(train_logits, label_train_batch)

    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(train_loss)

    var_list = tf.trainable_variables()
    paras_count = tf.reduce_sum([tf.reduce_prod(v.shape) for v in var_list])
    print('parameter number:%d' % sess.run(paras_count), end='\n\n')
    saver = tf.train.Saver()
    print('\nloading checkpoint...')
    ckpt = tf.train.get_checkpoint_state(pre_logs)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('loading sucecessful，global_step = %s\n' % global_step)
    else:
        print('no checkpoint')
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    s_t = time.time()
    try:
        for step in range(MAX_STEP):
            if coord.should_stop():
                break

            _, loss, acc = sess.run([train_op, train_loss, train_acc])

            if step % 100 == 0:
                runtime = time.time() - s_t
                print('Step: %6d, loss: %.8f, accuracy: %.2f%%, time:%.2fs, time left: %.2fhours'
                      % (step, loss, acc * 100, runtime, (MAX_STEP - step) * runtime / 360000))
                s_t = time.time()

            if step % 1000 == 0 or step == MAX_STEP - 1:  #
                checkpoint_path = os.path.join(logs_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        coord.request_stop()

    coord.join(threads=threads)
    sess.close()


def eval():
    N_CLASSES = 2
    IMG_SIZE = 208
    BATCH_SIZE = 1
    CAPACITY = 50
    MAX_STEP = 2000

    test_dir = 'test/'
    logs_dir = 'logs_1/'

    sess = tf.Session()

    train_list = get_all_files(test_dir, is_random=False)
    image_train_batch, label_train_batch = get_batch(train_list, IMG_SIZE, BATCH_SIZE, CAPACITY, False)
    train_logits = inference(image_train_batch, N_CLASSES)
    train_logits = tf.nn.softmax(train_logits)  #


    saver = tf.train.Saver()
    print('\nloading checkpoint')
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('loading successful，global_step = %s\n' % global_step)
    else:
        print('no checkpoint')
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        label_test = []
        for step in range(MAX_STEP):
            if coord.should_stop():
                break

            image, prediction = sess.run([image_train_batch, train_logits])
            max_index = np.argmax(prediction)
            label = []
            if max_index == 0:
                label_test.append('Cat')
            else:
                label_test.append('Dog')

    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        data = pd.DataFrame(label_test)
        data.to_csv("data.csv")
        coord.request_stop()

    coord.join(threads=threads)
    sess.close()


if __name__ == '__main__':
    training()
