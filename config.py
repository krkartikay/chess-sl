# Configuration for experiments

from experiment import Config

BATCH_SIZE = Config(
    name="BATCH_SIZE",
    default=128,
    values=[64, 128, 256, 512]
)

NUM_EPOCHS = Config(
    name="NUM_EPOCHS",
    default=100,
    dev=10
)

LEARNING_RATE = Config(
    name="LEARNING_RATE",
    default=0.1,
    values=[0.01, 0.03, 0.1, 0.3, 1, 3, 10]
)
