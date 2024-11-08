import sys

import processor
import wilor_code
from includes import *


def is_logged(msg):
    if not os.path.exists(LOGGER_PATH):
        return False

    with open(LOGGER_PATH) as log_file:
        lines = log_file.readlines()
        return msg in lines


def write_into_log(msg):
    with open(LOGGER_PATH, "a") as log_file:
        log_file.write(msg)


def process_video(video):
    name = Path(video).stem
    print(f"Starting to process {name}")

    if not is_logged(FINISHED_EXTRACTING_MSG.format(name)):
        print("Extracting frames...")
        processor.extract_frames(os.path.join(INPUT_DIR, video),
                       os.path.join(FRAMES_DIR, name),
                       fps=FPS,
                       frame_limit=FRAME_LIMIT)
        write_into_log(FINISHED_EXTRACTING_MSG.format(name))

    if not is_logged(FINISHED_PREDICTING_MSG.format(name)):
        print("Calling WiLoR...")
        wilor_code.predict_on_folder(input_folder=os.path.join(FRAMES_DIR, name),
                          output_folder=os.path.join(OBJECTS_DIR, name),
                          focal_length=FOCAL_LENGTH)
        write_into_log(FINISHED_PREDICTING_MSG.format(name))

    if not is_logged(FINISHED_PROCESSING_MSG.format(name)):
        print("Saving location tensor...")
        processor.build_location_tensor(os.path.join(OBJECTS_DIR, name),
                              LOCATION_DIR, name)

        # print("Calculating movement vectors...")
        # processor.calculate_movement_vectors(os.path.join(OBJECTS_DIR, name),
        #                            os.path.join(MOVEMENT_DIR, name))
        #
        # print("Building motion tensor...")
        # processor.build_video_motion_tensor(os.path.join(MOVEMENT_DIR, name),
        #                           MOTION_DIR, name)
        # write_into_log(FINISHED_PROCESSING_MSG.format(name))

    print(f"Done with {name}")
    os.system(f"rm -r {os.path.join(FRAMES_DIR, name)}")
    os.system(f"rm -r {os.path.join(OBJECTS_DIR, name)}")
    os.system(f"rm -r {os.path.join(MOVEMENT_DIR, name)}")


def main():
    videos = sorted(os.listdir(INPUT_DIR))

    with mp.Pool(NUM_OF_PROCESSES) as pool:
        results = [pool.apply_async(process_video, args=(os.path.join(INPUT_DIR, video),)) for video in videos]

        for result in results:
            print(result.get())

if __name__ == "__main__":
    main()