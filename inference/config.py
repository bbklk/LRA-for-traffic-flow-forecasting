TRAINING_STEPS = 40000
NUM_EPOCH = 2
LAYERS_NUM = 2
REGULARIZATION_RATE = 0.5
KEEP_PROB = 1
BATCH_SIZE = 256
HIDDEN_SIZE = 200
INPUT_SIZE = 249
TIME_SERIES_STEP = 6
LEARNING_RATE = 0.0001
MAX_GRAD_NORM = 5
TIME_REQUIRE = 3

# data path
TRAIN_RECORD_FILE = '../input/train.tfrecords'
VAL_RECORD_FILE = '../input/validation.tfrecords'
TEST_RECORD_FILE = '../input/test.tfrecords'
CSV_PATH = '../input/adj_mx_' + str(INPUT_SIZE) + '.csv'

# sample numbers
TEST_SAMPLE_NUMS_FIFTEEN = 2973
TRAIN_SAMPLE_NUMS_FIFTEEN = 23439
VAL_SAMPLE_NUMS_FIFTEEN = 5930

# scaler path
FLOW_SCALER_PATH = '../input/flow_scaler.pkl'
DATE_SCALER_PATH = '../input/date_scaler.pkl'

# model path
MODEL_NAME = 'model.ckpt'
MODEL_SAVE_PATH = '../model'
RT_MODEL_PATH = '../rt_model'

