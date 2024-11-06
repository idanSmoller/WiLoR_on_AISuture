import obj_helper
from includes import *


def combine_obj_files(file1, file2, output_file):
    vertices1, faces1 = obj_helper.read_obj(file1)
    vertices2, faces2 = obj_helper.read_obj(file2)

    offset = len(vertices1)

    adjusted_faces2 = []
    for face in faces2:
        adjusted_faces2.append([x + offset for x in face])

    obj_helper.write_obj(output_file, vertices1 + vertices2, faces1 + adjusted_faces2)


def combine(input_folder, output_folder):
    num_of_frames = int(sorted(os.listdir(input_folder))[-1][-10:-6])

    for i in trange(num_of_frames):
        combine_obj_files(os.path.join(input_folder, f"frame-{i+1:04}_0.obj"),
                          os.path.join(input_folder, f"frame-{i+1:04}_0.obj"),
                          os.path.join(output_folder, f"frame-{i+1:04}.obj"))