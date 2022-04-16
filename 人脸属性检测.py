import os
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
import My_Resnet
tf.config.experimental_run_functions_eagerly(True)

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


expected_features = {
    "image": tf.io.FixedLenFeature([], dtype=tf.string), # 变长的数据
    "Eyeglasses": tf.io.FixedLenFeature([1], dtype=tf.int64), # 定长的数据
    "Male": tf.io.FixedLenFeature([1], dtype=tf.int64), # 定长的数据
    "Young": tf.io.FixedLenFeature([1], dtype=tf.int64), # 定长的数据
    "Smiling": tf.io.FixedLenFeature([1], dtype=tf.int64)  # 定长的数据
}
def parse_example(serialized_example):
    example = tf.io.parse_example(serialized_example,
                                         expected_features)
    return example["image"], (example["Eyeglasses"], \
           example["Male"], example["Young"], example["Smiling"])
def read_batch_data(filename_fullpath, batch_size,prefetch_num, num_parallel_calls=5):
    dataset = tf.data.Dataset.list_files(filename_fullpath)
    dataset = dataset.interleave(
        lambda filename: tf.data.TFRecordDataset(
            filename_fullpath),
            cycle_length=1)
    # 数据集的个数为8w+
    dataset.shuffle(80000)
    dataset = dataset.map(parse_example,
                        num_parallel_calls=num_parallel_calls)
    # drop_remainder=True不够一个批次，丢弃掉
    batch_dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(prefetch_num)
    return batch_dataset

# 计算四个任务的损失函数
def cus_loss(labels, logits):
    # tensorflow2 rise Bug: https://github.com/tensorflow/tensorflow/issues/16345
    Eyeglasses_loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=labels[0], logits=logits[0])
    Male_loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=labels[1], logits=logits[1])
    Young_loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=labels[2], logits=logits[2])
    Smiling_loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=labels[3], logits=logits[3])
    return Eyeglasses_loss+Male_loss+Young_loss+Smiling_loss


def main():
    # 导进我们的主干网络
    network = My_Resnet.My_Resnet()
    # 定义自适应学习率
    learning_rate = CustomizedSchedule(128)
    # 定义优化器
    optimizer = keras.optimizers.Adam(learning_rate,
                                      beta_1=0.9,
                                      beta_2=0.98,
                                      epsilon=1e-9)
    train_loss = keras.metrics.Mean(name='train_loss')

    ####### chickpoint ##############
    checkpoint_dir = "./chickpoint"
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     model=network)
    # 如果保存模型的文件不存在就创建
    if not os.path.exists(checkpoint_prefix):
        os.mkdir(checkpoint_prefix)
    if tf.train.latest_checkpoint(checkpoint_dir):
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    ######### chickpoit end ##############

    @tf.function
    def train_step(batch_img, batch_labels):
        with tf.GradientTape() as tape:
            # batch_img_data 经过我们主干特征提取网络得到embeddings
            batch_logits = network(batch_img)
            # 计算回归损失函数
            loss = cus_loss(batch_labels, batch_logits)
        gradients = tape.gradient(loss, network.trainable_variables)
        optimizer.apply_gradients(zip(gradients, network.trainable_variables))
        train_loss(loss)
    # 循环10次
    for epoch in range(10):
        start = time.time()
        # 把累加值清空
        train_loss.reset_states()
        # 生成我们的批训练数据集
        filename_fullpath = "../dataset/train.tfrecords"
        batch_size = 50
        print("开始读取数据")
        batch_dataset = read_batch_data(filename_fullpath, batch_size=batch_size,
                                        prefetch_num=batch_size)
        print("数据读取完了")
        batch_ = 0
        for batch_img, batch_label in batch_dataset:
            batch_img = tf.io.decode_raw(batch_img, tf.uint8)
            batch_img = tf.cast(tf.reshape(batch_img,
                                      (batch_size, 224, 224, 3)), tf.float32)
            # 喂进去神经网络
            train_step(batch_img, batch_label)
            # 打印我们所关心的值
            batch_ += 1
            print('Batch [{}]  Loss {:.10f}'.format(batch_, train_loss.result()))
        print('Epoch [{}/{}]  Loss {:.10f}'.format(
                    epoch + 1, 10, train_loss.result()))
        print('Time take for 1 epoch: {} secs\n'.format(time.time() - start))
        # 每5次保存一下chickpoint和model
        if (epoch+1) % 5 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
            # 保存模型一下
            signatures = network.call.get_concrete_function(
                tf.TensorSpec(shape=[None, 224, 224, 3], dtype=tf.float32, name="inp"))
            tf.saved_model.save(network, './models', signatures)
            print("模型已经保存成功")
if __name__=="__main__":
    main()

