from includes import *


def read_obj(file_path):
    vertices = []
    faces = []

    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith('v '):
                    vertices.append([float(x) for x in line.split()])
                elif line.startswith('f '):
                    faces.append([float(x) for x in line.split()])
    return vertices, faces


def write_obj(output_file, vertices, faces):
    with open(output_file, "w") as file:
        for v in vertices:
            file.write(f"v {str(v[0])} {str(v[1])} {str(v[2])}")
        for f in faces:
            file.write(f"f {str(f[0])} {str(f[1])} {str(f[2])}")