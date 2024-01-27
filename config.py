# Configuration for experiments

from experiment import Config

OPTIMIZER = Config(
    name="OPTIMIZER",
    default="ADAM",
    values=['SGD', 'ADAM']
)

NUM_TRAINING_EXAMPLES = Config(
    name="NUM_TRAINING_EXAMPLES",
    default=30000,
    dev=10000,
    values=[10000, 30000, 100000]
)

BATCH_SIZE = Config(
    name="BATCH_SIZE",
    default=128,
    values=[8, 64, 128, 256, 512]
)

NUM_EPOCHS = Config(
    name="NUM_EPOCHS",
    default=100,
    dev=20,
    values=[10, 20, 50, 100]
)

LEARNING_RATE = Config(
    name="LEARNING_RATE",
    default=1,
    values=[1e-1, 1, 1e1, 1e2]
)
