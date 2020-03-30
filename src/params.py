import os
import seaborn as sns
import warnings

sns.set_style('white')
warnings.filterwarnings("ignore")

DATA_PATH = '../input/all-dogs/'
ANNOTATION_PATH = '../input/Annotation'
CP_PATH = '../output/'

IMAGES = [DATA_PATH + p for p in os.listdir(DATA_PATH)]

seed = 2019
IMG_SIZE = 64
NUM_CLASSES = 120