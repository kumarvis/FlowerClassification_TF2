import os
import tensorflow as tf
import project_params as pp

model_dir = pp.exp_checkpoint_path
csv_logger_path = os.path.join(model_dir, 'training_log.csv')
custom_period = 3
custom_patience = 5

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=custom_patience, monitor='val_acc',
                                     restore_best_weights=True),
    tf.keras.callbacks.TensorBoard(log_dir=model_dir, histogram_freq=0, write_graph=True),
    tf.keras.callbacks.CSVLogger(csv_logger_path),
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(model_dir, 'weights-epoch{epoch:02d}-loss{val_loss:.2f}.h5'), monitor='val_accuracy', verbose=1,
        period=custom_period, mode='max')
]