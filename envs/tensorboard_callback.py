
from tensorboardX import SummaryWriter
from rl.callbacks import Callback

class TensorBoard(Callback):
    def __init__(self):
        self.step = 0
        self.writer = SummaryWriter()
        self.reward = 0

    def on_step_end(self, step, logs={}):
        self.reward += logs['reward']
        self.writer.add_scalar("reward", self.reward, self.step)
        self.writer.add_scalar("loss", logs['metrics'][0], self.step)

        self.step += 1

    def on_episode_end(self, episode, logs):
        self.reward = 0

    def on_train_end(self, logs):
        self.writer.close()
