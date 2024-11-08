import glob
from PIL import Image
import os
from moviepy.editor import VideoFileClip
import numpy as np
from pathlib import Path
from time import time
from tqdm import tqdm, trange
import torch
import torch.multiprocessing as mp

NUM_OF_PROCESSES = 5

INPUT_DIR = "/strg/E/shared-data/AIxSuture/videos"
OUTPUT_DIR = "/strg/E/shared-data/AIxSuture_wilor_output"
# INPUT_DIR = "../rps_input"
# OUTPUT_DIR = "../rps_output"
FRAMES_DIR = OUTPUT_DIR + "/frames"
OBJECTS_DIR = OUTPUT_DIR + "/objects"
MOVEMENT_DIR = OUTPUT_DIR + "/movement"
MOTION_DIR = OUTPUT_DIR + "/motion"
LOCATION_DIR = OUTPUT_DIR + "/location"
COMBINATION_DIR = OUTPUT_DIR + "/combined_frames"
CONFIDENCES_DIR = OUTPUT_DIR + "/confs"
LOGGER_PATH = "progress_log.txt"
HAND_FACES_PATH = "hand_faces.txt"

RIGHT = 0
LEFT = 1
NUM_OF_VERTICES = 778

FINISHED_EXTRACTING_MSG = "Finished extracting frames for {}\n"
FINISHED_PREDICTING_MSG = "Finished running the model on {}'s frames\n"
FINISHED_PROCESSING_MSG = "Finished processing {} into motion and location\n"

FPS = 30
FRAME_LIMIT = None
FOCAL_LENGTH = 16.8
