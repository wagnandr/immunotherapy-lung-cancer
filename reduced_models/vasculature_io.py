import os
import pywavefront as pywf
import numpy as np
import dolfin as df

default_path_simple =  'data/Segmentation_Domains/simplified_blood_system_reduced.obj'

default_path_complex = 'data/Segmentation_Domains/complex_blood_system_reduced.obj'

default_path_vanilla = 'data/Segmentation_Domains/Vasculature.obj'


def read_default_simple(convert_mm_to_m=False):
    path_to_file_dir = os.path.dirname(os.path.realpath(__file__))
    return read(
        path=os.path.join(path_to_file_dir, '..', default_path_simple),
        convert_mm_to_m=convert_mm_to_m)


def read_default_complex(convert_mm_to_m=False):
    path_to_file_dir = os.path.dirname(os.path.realpath(__file__))
    return read(
        path=os.path.join(path_to_file_dir, '..', default_path_complex),
        convert_mm_to_m=convert_mm_to_m)


def read_default_vanilla(convert_mm_to_m=False):
    path_to_file_dir = os.path.dirname(os.path.realpath(__file__))
    return read(
        path=os.path.join(path_to_file_dir, '..', default_path_vanilla),
        convert_mm_to_m=convert_mm_to_m)


def read(path, convert_mm_to_m=False):
    mesh = pywf.Wavefront(path, collect_faces=True)
    assert len(mesh.mesh_list) == 1, 'supports only meshes containing one mesh list' 
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.mesh_list[0].faces)
    faces = faces.reshape((1,-1))
    vasculature = np.squeeze(vertices[faces])
    if convert_mm_to_m:
        vasculature[:] /= 1000
    return vasculature



def get_boundary_points(vertices):
    return np.array([[vertices[:,i].min(), vertices[:,i].max()] for i in range(3)]).transpose()


def get_bounding_cube(vertices, N):
    points = get_boundary_points(vertices) 
    intervals = points[1] - points[0]
    min_interval = intervals.min()
    normalized_intervals = intervals / min_interval
    num_intervals = np.round(normalized_intervals * N).astype(np.int32)
    dolfin_p0 = df.Point(points[0,0], points[0,1], points[0,2])
    dolfin_p1 = df.Point(points[1,0], points[1,1], points[1,2])
    return df.BoxMesh(dolfin_p0, dolfin_p1, num_intervals[0], num_intervals[1], num_intervals[2])


if __name__ == '__main__':
    vertices = read_default_simple()

    print(f'shape {vertices.shape}')

    mesh = get_bounding_cube(vertices, 4)

    for i in range(3):
        print(f'range {i}: {vertices[:,i].min()}, {vertices[:,i].max()}')
