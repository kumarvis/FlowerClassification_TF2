from matplotlib import pyplot as plt

def plot_hist_data(history):
    ## Plot Accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model-accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('model-accuracy.png')
    plt.clf()

    ## Plot Error
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model-loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('model-loss.png')