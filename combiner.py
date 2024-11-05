from includes import *


def read_obj(file_path):
    vertices = []
    faces = []

    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith('v '):  # Vertex line
                    parts = line.split()
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])
                    vertices.append(f"v {x} {y} {z}\n")
                elif line.startswith('f '):  # Face line
                    faces.append(line)
    return vertices, faces


def combine_obj_files(file1, file2, output_file):
    vertices1, faces1 = read_obj(file1)
    vertices2, faces2 = read_obj(file2)

    offset = len(vertices1)

    adjusted_faces2 = []
    for face in faces2:
        parts = face.split()
        adjusted_face = 'f ' + ' '.join(
            str(int(vertex.split('/')[0]) + offset) if '/' not in vertex else
            str(int(vertex.split('/')[0]) + offset) + '/' + '/'.join(vertex.split('/')[1:])
            for vertex in parts[1:]
        ) + '\n'
        adjusted_faces2.append(adjusted_face)

    with open(output_file, 'w') as out_file:
        out_file.writelines(vertices1)
        out_file.writelines(vertices2)
        out_file.writelines(faces1)
        out_file.writelines(adjusted_faces2)


def combine(input_folder, output_folder):
    num_of_frames = int(sorted(os.listdir(input_folder))[-1][-10:-6])

    for i in trange(num_of_frames):
        combine_obj_files(os.path.join(input_folder, f"frame-{i+1:04}_0.obj"),
                          os.path.join(input_folder, f"frame-{i+1:04}_0.obj"),
                          os.path.join(output_folder, f"frame-{i+1:04}.obj"))