import os
import torch
import torchvision
import logging
from utils import color
from tabulate import tabulate

def logger_config(log_path):
    loggerr = logging.getLogger()
    loggerr.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    loggerr.addHandler(handler)
    loggerr.addHandler(console)
    return loggerr

SEED = 40

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ['PYTHONHASHSEED'] = str(SEED)

##########################################################################
# Log Directories
##########################################################################
tensorboard = False
tensorboard_folder = './logs/tensorboard'
log = True
logging_folder = './logs'

if log:
    logging_log = logging_folder
    if not os.path.isdir(logging_log):
        os.makedirs(logging_log)
    logger = logger_config(log_path = logging_log + '/training_log.log')
    logger.info(f'Logging Directory: {logging_log}')   
##########################################################################

LEARNING_RATE = 0.001
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE    = 64
NUM_EPOCHS    = 30
NUM_WORKERS   = 4
IMAGE_HEIGHT  = 224
IMAGE_WIDTH   = 224
PIN_MEMORY    = True

LOAD_MODEL = True
CONTINUE   = True

TEACHER = False

SAVE_MODEL = True
POLY_LR    = True
DOWNLOAD   = False

task_ids = ['1','2','3','4','5']
task_table = tabulate(
                    tabular_data=[
                        ['YCD'          , 1],
                        ['YCDLW'        , 2],
                        ['OxfordIIITPet', 3],
                        ['Stanford40'   , 4],
                        ['MIT-67'       , 5]],
                    headers=['Task Name', 'ID'],
                    tablefmt="fancy_grid"
                    ) 

print(task_table)
task_id = input('Enter Task ID:  ')
assert (task_id in task_ids),'ID is Incorrect.'
task_id = int(task_id)

if task_id==1:
    NUM_CLASS = 2
    TASK_NAME = 'YCD'

if task_id==2:
    NUM_CLASS = 5
    TASK_NAME = 'YCDLW'

if task_id==3:
    NUM_CLASS = 37
    TASK_NAME = 'OxfordIIITPet'

if task_id==4:
    NUM_CLASS = 40
    TASK_NAME = 'Stanford40'

if task_id==5:
    NUM_CLASS = 3
    TASK_NAME = 'MIT-67'

model_ids = ['1','2','3','4']
model_table = tabulate(
                    tabular_data=[
                        ['ConvNext'    , 1],
                        ['MVIT'        , 2],
                        ['VIT'         , 3],
                        ['DinoV2'      , 4]],
                    headers=['Model Name', 'ID'],
                    tablefmt="fancy_grid"
                    )

print(model_table)
model_id = input('Enter Model ID:  ')
assert (model_id in model_ids),'ID is Incorrect.'
model_id = int(model_id)

if model_id==1:
    MODEL_NAME = 'ConvNext'

elif model_id==2:
    MODEL_NAME = 'MVIT'

elif model_id==3:
    MODEL_NAME = 'VIT'

elif model_id==4:
    MODEL_NAME = 'DinoV2'

CKPT_NAME = MODEL_NAME + '_' + TASK_NAME

table = tabulate(
    tabular_data=[
        ['Learning Rate', LEARNING_RATE],
        ['Num Classes', NUM_CLASS],
        ['Device', DEVICE],
        ['Batch Size', BATCH_SIZE],
        ['POLY_LR', POLY_LR],
        ['Num Epochs', NUM_EPOCHS],
        ['Num Workers', NUM_WORKERS],
        ['Image Height', IMAGE_HEIGHT],
        ['Image Width', IMAGE_WIDTH],
        ['Pin Memory', PIN_MEMORY],
        ['Load Model', LOAD_MODEL],
        ['Save Model', SAVE_MODEL],
        ['Download Dataset', DOWNLOAD],
        ['Model Name', MODEL_NAME],
        ['Seed', SEED],
        ['Task Name', TASK_NAME],
        # ['GPU', torch.cuda.get_device_name(0)],
        ['Torch', torch.__version__],
        ['Torchvision', torchvision.__version__],
        ['Checkpoint Name', CKPT_NAME]],
    headers=['Hyperparameter', 'Value'],
    tablefmt="fancy_grid"
    )

logger.info(table)













