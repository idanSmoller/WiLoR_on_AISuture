import glob
from PIL import Image
import os
from moviepy.editor import VideoFileClip
import numpy as np
from pathlib import Path
from time import time
from tqdm import tqdm, trange

INPUT_DIR = "/strg/E/shared-data/AIxSuture"
OUTPUT_DIR = "/strg/E/shared-data/AIxSuture_wilor_output"
# INPUT_DIR = "../rps_input"
# OUTPUT_DIR = "../rps_output"
FRAMES_DIR = OUTPUT_DIR + "/frames"
OBJECTS_DIR = OUTPUT_DIR + "/objects"
MOVEMENT_DIR = OUTPUT_DIR + "/movement"
MOTION_DIR = OUTPUT_DIR + "/motion"
LOCATION_DIR = OUTPUT_DIR + "/location"
COMBINATION_DIR = OUTPUT_DIR + "/combined_frames"
LOGGER_PATH = "progress_log.txt"

FINISHED_EXTRACTING_MSG = "Finished extracting frames for {}"
FINISHED_PREDICTING_MSG = "Finished running the model on {}'s frames"
FINISHED_PROCESSING_MSG = "Finished processing {} into motion and location"

FPS = 30
FRAME_LIMIT = None
FOCAL_LENGTH = 16.8
KEEP_ALL = ["A31H"]
