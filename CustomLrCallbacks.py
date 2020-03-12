import tensorflow as tf
import matplotlib.pyplot as plt
import logging

class MyCustomCallback(tf.keras.callbacks.Callback):

    def __init__(self, lr_step):
        super().__init__(self)
        self.lr_step = lr_step
        self.list_lr = []
        self.list_loss = []
        log_batch_end = logging.getLogger("batch-end-logger")

    def on_batch_end(self, batch, logs=None):
        smoothing = 0.05
        loss = logs['loss']
        if len(self.list_loss) == 0:
            curr_loss = loss
        else:
            curr_loss = smoothing * loss + (1 - smoothing) * self.list_loss[-1]
        self.list_loss.append(curr_loss)

        curr_lr = self.model.optimizer.lr.read_value()
        self.list_lr.append(curr_lr)
        new_lr = curr_lr * self.lr_step
        self.model.optimizer.lr.assign(new_lr)

    def on_train_end(self, logs=None):
        print("Training: \033[91mend\033[0m.")
        min_loss = min(self.list_loss)
        min_loss_index = self.list_loss.index(min_loss)
        min_loss_lr = self.list_lr[min_loss_index]
        recommended_learning_rate = min_loss_lr / 10

        plt.xlabel("lr_rate")
        plt.ylabel("loss")
        plt.xscale("log")
        plt.plot(self.list_lr, self.list_loss)
        plt.scatter(min_loss_lr, min_loss, s=20, c='red')
        plt.savefig('loss_lr_rate.png')

        logging.basicConfig(filename='optimal_lr.log', filemode='w', format='%(asctime)s - %(message)s',
                            level=logging.INFO)
        logging.info('min loss = %f', min_loss)
        logging.info('recommended_learning_rate = %f', recommended_learning_rate)




