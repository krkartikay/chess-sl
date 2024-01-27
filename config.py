# Configuration for experiments

from experiment import Config

OPTIMIZER = Config(
    name="OPTIMIZER",
    default="ADAM",
    values=['SGD', 'ADAM']
)

DROPOUT_ENABLED = Config(
    name="DROPOUT_ENABLED",
    values=[False, True]
)

N_HIDDEN = Config(
    name = "N_HIDDEN",
    default=4096,
    dev=1024,
    values=[1024,4096],
)

N_BLOCKS = Config(
    name = "N_BLOCKS",
    default=8,
    dev=2,
    values=[0,1,4,8]
)

N_CHANNELS = Config(
    name="N_CHANNELS",
    default=16,
    dev=16,
    values=[4,16,64,128]
)

FILTER_SIZE = Config(
    name="FILTER_SIZE",
    default=3
)

NUM_TRAINING_EXAMPLES = Config(
    name="NUM_TRAINING_EXAMPLES",
    default=10000,
    dev=10000,
    values=[10000, 30000, 100000]
)

BATCH_SIZE = Config(
    name="BATCH_SIZE",
    default=128,
    values=[64, 128, 256]
)

NUM_EPOCHS = Config(
    name="NUM_EPOCHS",
    default=20,
    dev=20,
    values=[20, 50]
)

LEARNING_RATE = Config(
    name="LEARNING_RATE",
    default=3,
    values=[1, 3]
)
