import obj_helper
from includes import *


def extract_frames(video_path, output_folder, fps, frame_limit):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the video clip
    clip = VideoFileClip(video_path)
    num_of_frames = int(clip.duration * clip.fps)

    # Iterate over each frame and save it as an image
    for i, frame in enumerate(tqdm(clip.iter_frames(fps=fps), total=num_of_frames), start=1):
        if frame_limit and i > frame_limit:
            break
        # Construct the file name for the frame
        filename = os.path.join(output_folder, f"frame-{i:04d}.jpg")
        # Save the frame as an image
        clip = Image.fromarray(frame)
        clip.save(filename)

    # Close the clip
    clip.close()


def load_obj_file(file_path):
    """Load vertices from an OBJ file."""
    vertices = []
    with open(file_path+'.obj', 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.split()
                x, y, z = map(float, parts[1:4])
                vertices.append((x, y, z))
    return vertices


def reconstruct_hands(location_tensor_path, output_folder):
    with open(HAND_FACES_PATH) as file:
        faces = file.readlines()
    faces_right = [face.split() for face in faces]
    faces_left = [[x[0], x[2], x[1]] for x in faces_right]

    location_tensor = np.load(location_tensor_path)
    for i, location_vec in enumerate(tqdm(location_tensor)):
        vertices_right = location_vec[:NUM_OF_VERTICES]
        vertices_left = location_vec[NUM_OF_VERTICES:]

        obj_helper.write_obj(os.path.join(output_folder, f"frame-{i:04}_{RIGHT}"), vertices_right, faces_right)
        obj_helper.write_obj(os.path.join(output_folder, f"frame-{i:04}_{LEFT}"), vertices_left, faces_left)


def build_location_tensor(source_folder, output_folder, name):
    """Normalize the location of the hands in the frames and stack them into a location tensor"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = [Path(file).stem for file in glob.glob(f'{source_folder}/frame-*.obj')]

    # Group files by frame number
    frames = {}
    for file in files:
        frame_number = file.split('_')[0].split('-')[1]
        hand_index = file.split('_')[1].split('.')[0]

        # Skip faulty frames
        if any(f"frame-{frame_number}_2" in f for f in files):
            continue

        if frame_number not in frames:
            frames[frame_number] = {}
        frames[frame_number][hand_index] = os.path.join(source_folder, file)

    # Sort frame numbers
    sorted_frames = sorted(frames.keys())

    location_tensor = []
    # Calculate vectors for consecutive frames
    for i in trange(len(sorted_frames)):
        # Process right hand first (index '1'), then left hand (index '0')
        vectors = []
        for hand_index in ['1', '0']:
            if hand_index in frames[sorted_frames[i]]:
                vertices = load_obj_file(frames[sorted_frames[i]][hand_index])
            else:
                k = 1
                filled_in = False

                while not filled_in:
                    if i + k >= len(sorted_frames) and i - k < 0:
                        raise(Exception(f"the {'right' if hand_index == 0 else 'left'} hand is never detected!"))

                    if i + k < len(sorted_frames) and hand_index in frames[sorted_frames[i + k]]:
                        vertices = load_obj_file(frames[sorted_frames[i + k]][hand_index])
                        filled_in = True
                    if i - k >= 0 and hand_index in frames[sorted_frames[i - k]]:
                        vertices = load_obj_file(frames[sorted_frames[i - k]][hand_index])
                        filled_in = True

                    k += 1

            vectors += vertices

        location_tensor.append(vectors)

    location_tensor_np = np.array(location_tensor)
    np.save(f"{output_folder}/{name}", location_tensor_np)


def calculate_movement_vectors(source_folder, output_folder):
    """Calculate movement vectors between consecutive frames."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = [Path(file).stem for file in glob.glob(f'{source_folder}/frame-*.obj')]

    # Group files by frame number
    frames = {}
    for file in files:
        frame_number = file.split('_')[0].split('-')[1]
        hand_index = file.split('_')[1].split('.')[0]

        # Skip faulty frames
        if any(f"frame-{frame_number}_2" in f for f in files):
            continue

        if frame_number not in frames:
            frames[frame_number] = {}
        frames[frame_number][hand_index] = os.path.join(source_folder, file)

    # Sort frame numbers
    sorted_frames = sorted(frames.keys())

    # Calculate vectors for consecutive frames
    for i in trange(len(sorted_frames) - 1):

        # Process right hand first (index '1'), then left hand (index '0')
        vectors = []
        for hand_index in ['1', '0']:
            if hand_index in frames[sorted_frames[i]]:
                vertices_frame1 = load_obj_file(frames[sorted_frames[i]][hand_index])
            else:
                k = 1
                filled_in = False

                while not filled_in:
                    if hand_index in frames[sorted_frames[i + k]]:
                        vertices_frame1 = load_obj_file(frames[sorted_frames[i + k]][hand_index])
                        filled_in = True
                    if hand_index in frames[sorted_frames[i - k]]:
                        vertices_frame1 = load_obj_file(frames[sorted_frames[i - k]][hand_index])
                        filled_in = True

                    k += 1

            if hand_index in frames[sorted_frames[i + 1]]:
                vertices_frame2 = load_obj_file(frames[sorted_frames[i + 1]][hand_index])
            else:
                k = 1
                filled_in = False

                while not filled_in:
                    if hand_index in frames[sorted_frames[i + 1 + k]]:
                        vertices_frame2 = load_obj_file(frames[sorted_frames[i + 1 + k]][hand_index])
                        filled_in = True
                    if hand_index in frames[sorted_frames[i + 1 - k]]:
                        vertices_frame2 = load_obj_file(frames[sorted_frames[i + 1 - k]][hand_index])
                        filled_in = True

                    k += 1

            # Calculate movement vectors
            for v1, v2 in zip(vertices_frame1, vertices_frame2):
                dx = v2[0] - v1[0]
                dy = v2[1] - v1[1]
                dz = v2[2] - v1[2]
                vectors.append((dx, dy, dz))

        output_file = Path(f"{output_folder}/{sorted_frames[i]}_{sorted_frames[i + 1]}.txt")
        with open(output_file, 'w') as f:
            for vector in vectors:
                f.write(f"{vector[0]} {vector[1]} {vector[2]}\n")


def build_video_motion_tensor(movement_path, output_folder, name):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frames = []
    for file_name in tqdm(os.listdir(movement_path)):
        if file_name.endswith('.txt'):
            file_path = os.path.join(movement_path, file_name)
            with open(file_path, 'r') as file:
                points = [[float(num) for num in line.split()] for line in file]
                frames.append(points)
    motion_tensor = np.array(frames)
    np.save(f'{output_folder}/{name}.npy', np.array(motion_tensor))
