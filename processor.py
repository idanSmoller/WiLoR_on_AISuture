from includes import *


def extract_frames(video_path, output_directory, fps, frame_limit):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Load the video clip
    clip = VideoFileClip(video_path)

    # Iterate over each frame and save it as an image
    for i, frame in enumerate(clip.iter_frames(fps=fps), start=1):
        if frame_limit and i > frame_limit:
            break
        # Construct the file name for the frame
        filename = os.path.join(output_directory, f"frame-{i:04d}.jpg")
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


def build_location_tensor(source_folder, output_directory, name):
    """Normalize the location of the hands in the frames and stack them into a location tensor"""
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

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
    for i in range(len(sorted_frames)):
        # Process right hand first (index '1'), then left hand (index '0')
        vectors = []
        for hand_index in ['1', '0']:
            if hand_index in frames[sorted_frames[i]]:
                vertices = load_obj_file(frames[sorted_frames[i]][hand_index])
            else:
                k = 1
                filled_in = False

                while not filled_in:
                    if hand_index in frames[sorted_frames[i + k]]:
                        vertices = load_obj_file(frames[sorted_frames[i + k]][hand_index])
                        filled_in = True
                    if hand_index in frames[sorted_frames[i - k]]:
                        vertices = load_obj_file(frames[sorted_frames[i - k]][hand_index])
                        filled_in = True

                    k += 1

            vectors += vertices

        location_tensor.append(vectors)

    location_tensor_np = np.array(location_tensor)
    np.save(f"{output_directory}/{name}", location_tensor_np)


def calculate_movement_vectors(source_folder, output_directory):
    """Calculate movement vectors between consecutive frames."""
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

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
    for i in range(len(sorted_frames) - 1):

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

        output_file = Path(f"{output_directory}/{sorted_frames[i]}_{sorted_frames[i + 1]}.txt")
        with open(output_file, 'w') as f:
            for vector in vectors:
                f.write(f"{vector[0]} {vector[1]} {vector[2]}\n")


def build_video_motion_tensor(movement_path, output_directory, name):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    frames = []
    for file_name in os.listdir(movement_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(movement_path, file_name)
            with open(file_path, 'r') as file:
                points = [[float(num) for num in line.split()] for line in file]
                frames.append(points)
    motion_tensor = np.array(frames)
    np.save(f'{output_directory}/{name}.npy', np.array(motion_tensor))