import os, math
import tensorflow as tf
import project_params as pp
import argparse
from prepare_tf_dataset_format import get_dataset_ready
import matplotlib.pyplot as plt
from prepare_model import get_custom_model
from tf_rectified_adam import RectifiedAdam


def show_batch(image_batch, label_batch, batch_sz):
    N = math.floor(math.sqrt(batch_sz))
    SqN = N * N
    plt.figure(figsize=(10,10))
    for n in range(SqN):
        ax = plt.subplot(N, N , n+1)
        plt.imshow(image_batch[n])
        plt.title(label_batch[n])
        plt.axis('off')
    plt.show()

Base_Path = pp.Base_Path
batch_sz = pp.exp_batch_sz

Train_Path = os.path.join(Base_Path, 'train')
Validation_Path = os.path.join(Base_Path, 'validation')

train_ds, no_train_images = get_dataset_ready(Train_Path)
validation_ds, no_validation_images = get_dataset_ready(Validation_Path)

show_batch_flag = False
if show_batch_flag == True:
    image_batch, label_batch = next(iter(train_ds))
    show_batch(image_batch.numpy(), label_batch.numpy(), batch_sz)

custom_model = get_custom_model()

# Compile:
lr = pp.exp_learning_rate
optimizer = tf.keras.optimizers.Adam(lr)
#optimizer = RectifiedAdam(lr=lr)

custom_model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy(name='acc')
        #tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5_acc')
    ])

train_steps_per_epoch = math.ceil(no_train_images / batch_sz)
val_steps_per_epoch = math.ceil(no_validation_images / batch_sz)
num_epochs = pp.exp_max_epochs

##Callbacks
from custom_callbacks import callbacks
# Train Start:
history_freeze = custom_model.fit(
    train_ds, epochs=num_epochs, steps_per_epoch=train_steps_per_epoch,
    validation_data=validation_ds, validation_steps=val_steps_per_epoch,
    verbose=1, callbacks=callbacks)

print('----> EXPERIMENT FINISHED <----')
