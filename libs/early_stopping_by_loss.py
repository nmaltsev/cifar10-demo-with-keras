from keras.callbacks import Callback

# https://towardsdatascience.com/resuming-a-training-process-with-keras-3e93152ee11a

class EarlyStoppingByLoss(Callback):
    def __init__(self, threshold=0.03):
        super(Callback, self).__init__()
        self.threshold = threshold

    # def on_batch_end(self, batch, logs={}):
    def on_epoch_end(self, epoch, logs={}):
        print('\nEE ', logs.get('loss'), ' ',  self.threshold)
        print(logs)
        if logs.get('loss') <= self.threshold:
                self.model.stop_training = True
