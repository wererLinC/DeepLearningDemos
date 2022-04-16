import argparse
import os
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
import facenet
import mobilenet_v1

# GPU设置
tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
# 设置第二块可见
tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
# 设置内存自增长
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# 自定义学习率
class CustomizedSchedule(
    keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomizedSchedule, self).__init__()

        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** (-1.5))

        arg3 = tf.math.rsqrt(self.d_model)

        return arg3 * tf.math.minimum(arg1, arg2)


def main(args):
    # 导进我们的主干网络
    network = mobilenet_v1.mobilenet_v1(args.embedding_size)
    # 加载进数据
    dataset = facenet.get_dataset(args.data_dir)
    # 定义自适应学习率
    learning_rate = CustomizedSchedule(128)
    # 定义优化器
    optimizer = keras.optimizers.Adam(learning_rate,
                                      beta_1=0.9,
                                      beta_2=0.98,
                                      epsilon=1e-9)
    train_loss = keras.metrics.Mean(name='train_loss')

    ####### chickpoint ##############
    checkpoint_dir = args.chickpoint_base_dir
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     model=network)
    # 如果保存模型的文件不存在就创建
    if not os.path.exists(checkpoint_prefix):
        os.mkdir(checkpoint_prefix)
    if tf.train.latest_checkpoint(checkpoint_dir):
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    ######### chickpoit ##############

    @tf.function
    def train_step(x):
        with tf.GradientTape() as tape:
            # batch_img_data 经过我们主干特征提取网络得到embeddings
            embeddings = network(x, training=True)
            # 进行 l2 正则化
            embeddings = tf.nn.l2_normalize(embeddings, 1, 1e-10, name='embeddings')
            # 传入 triplet_loss 计算三元组损失
            loss = facenet.triplet_loss(embeddings, args.embedding_size, args.alpha)
        gradients = tape.gradient(loss, network.trainable_variables)
        optimizer.apply_gradients(zip(gradients, network.trainable_variables))
        train_loss(loss)

    for epoch in range(args.epoch_size):
        start = time.time()
        # 把累加值清空
        train_loss.reset_states()
        # 每次重新生成新三元组
        triplets, n_triplets = facenet.select_triplets(dataset)
        print("Selecting suitable triplets for Training")
        print("(n_triplets, triplet_per_batch) = ({}, {})".format(n_triplets, args.triplet_per_batch))
        start_trilpet_index = 0
        for i in range(len(triplets) // args.triplet_per_batch):
            batch_img_data, batch_img_num = facenet.produce_train_data(triplets, args.triplet_per_batch, start_trilpet_index)
            start_trilpet_index += args.triplet_per_batch
            # 把数据类型转化为 float32
            batch_img_data = tf.cast(batch_img_data, tf.float32)
            # 归一化一下
            batch_img_data /= 255.0
            # 喂进去神经网络
            train_step(batch_img_data)
            # 打印我们所关心的值
            print('Epoch [{}/{}] Batch [{}/{}]'.format(
                    epoch + 1, args.epoch_size,  (i+1),
                (len(triplets) // args.triplet_per_batch)))
        print('Epoch {} '.format(epoch + 1))
        print('Time take for 1 epoch: {} secs\n'.format(time.time() - start))
        # 保存模型一下
        if (epoch+1) % 50 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        if (epoch+1) % 50 == 0:
            signatures = network.call.get_concrete_function(
                tf.TensorSpec(shape=[None, 160, 160, 3], dtype=tf.float32, name="inp"))
            tf.saved_model.save(network, args.models_base_dir, signatures)
            # 每训练 100 步 我们来验证一下我们模型的准确性
            print("模型已经保存成功")

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_base_dir', type=str,
                        help='模型存放的路径.', default='E:\\zhangshiWorkSpace\\westudy\\facenet-tf2\\src\\models')
    parser.add_argument('--chickpoint_base_dir', type=str,
                        help='模型存放的路径.', default='E:\\zhangshiWorkSpace\\westudy\\facenet-tf2\\src\\chickpoint')
    parser.add_argument('--data_dir', type=str,
                        help='训练数据集的位置.', default='E:\\zhangshiWorkSpace\\FaceRegition\\faceV5_160')
    parser.add_argument('--triplet_per_batch', type=int,
                        help='每个批次选用几个三元组进行训练.', default=100)
    parser.add_argument('--epoch_size', type=int,
                        help='算法总共迭代几次.', default=50)
    parser.add_argument('--alpha', type=float,
                        help='损失函数中的 alpha.', default=0.2)
    parser.add_argument('--embedding_size', type=int,
                        help='经过神经网络输出的特征维度.', default=128)

    # 这几个参数是我们在使用 lfw 做验证集时会用到的。暂时先不同管
    parser.add_argument('--lfw_pairs', type=str,
                        help='The file containing the pairs to use for validation.', default='data/pairs.txt')
    parser.add_argument('--lfw_dir', type=str,
                        help='Path to the data directory containing aligned face patches.', default='')
    parser.add_argument('--lfw_nrof_folds', type=int,
                        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    return parser.parse_args(argv)


if __name__ == '__main__':
    # sys.argv[0]) 是我们文件的路径，从第1个开始才是我们真正传进去的参数
    main(parse_arguments(sys.argv[1:]))
