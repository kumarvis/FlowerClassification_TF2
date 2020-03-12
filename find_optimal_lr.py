import os, math
import tensorflow as tf
import project_params as pp
from prepare_tf_dataset_format import get_dataset_ready
import matplotlib.pyplot as plt
from prepare_model import get_custom_model
from LRFinder import LRFinder

def show_batch(image_batch, label_batch, batch_sz):
    N = math.floor(math.sqrt(batch_sz))
    SqN = N * N
    plt.figure(figsize=(10, 10))
    for n in range(SqN):
        ax = plt.subplot(N, N, n+1)
        plt.imshow(image_batch[n])
        plt.title(label_batch[n])
        plt.axis('off')
    plt.show()

Base_Path = pp.Base_Path
batch_sz = pp.exp_batch_sz

Train_Path = os.path.join(Base_Path, 'train')
train_ds, no_train_images = get_dataset_ready(Train_Path)

show_batch_flag = False
if show_batch_flag == True:
    image_batch, label_batch = next(iter(train_ds))
    show_batch(image_batch.numpy(), label_batch.numpy(), batch_sz)

custom_model = get_custom_model()

min_lr, max_lr = 0.00001, 1
# Compile:
optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

lr_finder = LRFinder(custom_model, optimizer, loss_fn, train_ds)
#lr_finder.range_test(min_lr, max_lr)
#lr_finder.dump()
lr_finder.plot_smooth_curve('optimal_lr.log')

print('----> EXPERIMENT FINISHED <----')

