from config import global_config
from config_utils import merge_configs, get_local_pipeline
from utils import STAGES, final_report
from snorkel_pipeline import SnorkelPipeline
from babble_pipeline import BabblePipeline
from image_pipeline import ImagePipeline