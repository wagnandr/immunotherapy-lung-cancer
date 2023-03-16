import time
import numpy as np
import dolfin as df
import petsc4py as p4py
from typing import List, Tuple

code_cpp = """

#include <dolfin.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

void integrate_2d_surface_mat(
        const dolfin::FunctionSpace & V,
        const std::vector< Eigen::Vector3d > & triangle_list,
        const std::vector< double > & surface_value_list,
        const std::vector< double > & boundary_box,
        const size_t N,
        std::vector< size_t >& row_indices,
        std::vector< size_t >& col_indices,
        std::vector< double >& mat_values
)
{
    // vertex 2 dof map
    std::vector<dolfin::la_index> vertex2dof = V.dofmap()->entity_dofs(*V.mesh(), 0);


    const double x_min = boundary_box[0];
    const double y_min = boundary_box[1];
    const double z_min = boundary_box[2];

    const double x_max = boundary_box[3];
    const double y_max = boundary_box[4];
    const double z_max = boundary_box[5];

    // local to global map
    std::vector<size_t> local_to_global;
    V.dofmap()->tabulate_local_to_global_dofs(local_to_global);

    std::shared_ptr< dolfin::BoundingBoxTree > tree = V.mesh()->bounding_box_tree();

    std::shared_ptr< const dolfin::FiniteElement > dolfin_element = V.element();

    std::vector< double > coordinate_dofs;
    std::vector< Eigen::Vector3d > points;
    std::vector< double > values( 4, 0. );

    const size_t num_triangles = triangle_list.size() / 3;
    dolfin::Cell cell(*V.mesh(), 0);
    for (size_t triangle_index = 0; triangle_index < num_triangles; triangle_index += 1)
    {
        const Eigen::Vector3d& p0 = triangle_list[triangle_index * 3 + 0];
        const Eigen::Vector3d& p1 = triangle_list[triangle_index * 3 + 1];
        const Eigen::Vector3d& p2 = triangle_list[triangle_index * 3 + 2];

        Eigen::Vector3d d01 = p1 - p0;
        Eigen::Vector3d d02 = p2 - p0;

        const double area = 0.5 * d01.cross(d02).norm();

        const double len01 = d01.norm();
        const double len02 = d02.norm();

        d01 /= len01;
        d02 /= len02;

        const double h01 = len01 / N;
        const double h02 = len02 / N;

        size_t num_points = ((N+1) * (N+2)) / 2;

        for (size_t i = 0; i < N+1; i += 1)
        {
            for (size_t j = 0; j < N+1-i; j += 1)
            {
                Eigen::Vector3d point = p0 + double(h01 * i) * d01 + double(h02 * j) * d02;

                const bool in_x = (point[0] >= x_min-1e-12 && point[0] <= x_max+1e-12);
                const bool in_y = (point[1] >= y_min-1e-12 && point[1] <= y_max+1e-12);
                const bool in_z = (point[2] >= z_min-1e-12 && point[2] <= z_max+1e-12);

                if ( !(in_x && in_y && in_z) )
                    continue;
                
                dolfin::Point dolfin_point(point[0], point[1], point[2]);
                if (!cell.contains(dolfin_point))
                {
                    // auto result_tree = tree->compute_closest_entity(dolfin_point);
                    // const size_t cell_id = result_tree.first;
                    // const double distance = result_tree.second;

                    // if (distance > 1e-14)
                    //     continue;

                    const unsigned int cell_id = tree->compute_first_entity_collision(dolfin_point);
                    if (cell_id == std::numeric_limits< unsigned int >::max())
                        continue;

                    cell = dolfin::Cell (*V.mesh(), cell_id);
                }

                if (cell.is_ghost())
                    continue;

                cell.get_coordinate_dofs(coordinate_dofs);

                const size_t orientation = cell.orientation();

                const size_t num_values = cell.num_entities(0);
                values.resize( num_values );
                dolfin_element->evaluate_basis_all(
                        values.data(),
                        point.data(),
                        coordinate_dofs.data(),
                        orientation);

                const unsigned int * vertex_ids = cell.entities(0);

                for (size_t i = 0; i < cell.num_entities(0); i += 1)
                {
                    for (size_t j = 0; j < cell.num_entities(0); j += 1)
                    {
                        row_indices.push_back( local_to_global[vertex2dof[vertex_ids[i]]] );
                        col_indices.push_back( local_to_global[vertex2dof[vertex_ids[j]]] );
                        mat_values.push_back( values[i] * values[j] * area * surface_value_list[triangle_index] / num_points );
                    }
                }
            }
        }
    }
}

void integrate_2d_surface(
        const dolfin::FunctionSpace & V,
        const std::vector< Eigen::Vector3d > & triangle_list,
        const std::vector< double > & surface_value_list,
        const std::vector< double > & boundary_box,
        const size_t N,
        std::vector< size_t >& rhs_indices,
        std::vector< double >& rhs_values
)
{
    // vertex 2 dof map
    std::vector<dolfin::la_index> vertex2dof = V.dofmap()->entity_dofs(*V.mesh(), 0);


    const double x_min = boundary_box[0];
    const double y_min = boundary_box[1];
    const double z_min = boundary_box[2];

    const double x_max = boundary_box[3];
    const double y_max = boundary_box[4];
    const double z_max = boundary_box[5];

    // local to global map
    std::vector<size_t> local_to_global;
    V.dofmap()->tabulate_local_to_global_dofs(local_to_global);

    std::shared_ptr< dolfin::BoundingBoxTree > tree = V.mesh()->bounding_box_tree();

    std::shared_ptr< const dolfin::FiniteElement > dolfin_element = V.element();

    std::vector< double > coordinate_dofs;
    std::vector< Eigen::Vector3d > points;
    std::vector< double > values( 4, 0. );

    const size_t num_triangles = triangle_list.size() / 3;
    dolfin::Cell cell(*V.mesh(), 0);
    for (size_t triangle_index = 0; triangle_index < num_triangles; triangle_index += 1)
    {
        const Eigen::Vector3d& p0 = triangle_list[triangle_index * 3 + 0];
        const Eigen::Vector3d& p1 = triangle_list[triangle_index * 3 + 1];
        const Eigen::Vector3d& p2 = triangle_list[triangle_index * 3 + 2];

        Eigen::Vector3d d01 = p1 - p0;
        Eigen::Vector3d d02 = p2 - p0;

        const double area = 0.5 * d01.cross(d02).norm();

        const double len01 = d01.norm();
        const double len02 = d02.norm();

        d01 /= len01;
        d02 /= len02;

        const double h01 = len01 / N;
        const double h02 = len02 / N;

        size_t num_points = ((N+1) * (N+2)) / 2;

        for (size_t i = 0; i < N+1; i += 1)
        {
            for (size_t j = 0; j < N+1-i; j += 1)
            {
                Eigen::Vector3d point = p0 + double(h01 * i) * d01 + double(h02 * j) * d02;

                const bool in_x = (point[0] >= x_min-1e-12 && point[0] <= x_max+1e-12);
                const bool in_y = (point[1] >= y_min-1e-12 && point[1] <= y_max+1e-12);
                const bool in_z = (point[2] >= z_min-1e-12 && point[2] <= z_max+1e-12);

                if ( !(in_x && in_y && in_z) )
                    continue;
                
                dolfin::Point dolfin_point(point[0], point[1], point[2]);
                if (!cell.contains(dolfin_point))
                {
                    // auto result_tree = tree->compute_closest_entity(dolfin_point);
                    // const size_t cell_id = result_tree.first;
                    // const double distance = result_tree.second;

                    // if (distance > 1e-14)
                    //     continue;

                    const unsigned int cell_id = tree->compute_first_entity_collision(dolfin_point);
                    if (cell_id == std::numeric_limits< unsigned int >::max())
                        continue;

                    cell = dolfin::Cell (*V.mesh(), cell_id);
                }

                if (cell.is_ghost())
                    continue;

                cell.get_coordinate_dofs(coordinate_dofs);

                const size_t orientation = cell.orientation();

                const size_t num_values = cell.num_entities(0);
                values.resize( num_values );
                dolfin_element->evaluate_basis_all(
                        values.data(),
                        point.data(),
                        coordinate_dofs.data(),
                        orientation);

                const unsigned int * vertex_ids = cell.entities(0);

                for (size_t i = 0; i < cell.num_entities(0); i += 1)
                {
                    rhs_indices.push_back( local_to_global[vertex2dof[vertex_ids[i]]] );
                    rhs_values.push_back( values[i] * area * surface_value_list[triangle_index] / num_points );
                }
            }
        }
    }
}

struct Result
{
    std::vector< size_t > indices;
    std::vector< double > values;
};

struct ResultMatrix
{
    std::vector< size_t > row_indices;
    std::vector< size_t > col_indices;
    std::vector< double > values;
};

Result integrate_2d_surface_converter(
        const dolfin::FunctionSpace & V,
        // const std::vector< std::array< double, 3 >>& triangle_list,
        const std::vector< double >& triangle_list,
        const std::vector< double >& surface_value_list,
        const std::vector< double > & boundary_box,
        const size_t N)
{
    std::vector< Eigen::Vector3d > triangle_list_converted;
    for (int i = 0; i < triangle_list.size() / 3; i += 1)
    {
        triangle_list_converted.emplace_back(
            triangle_list[3*i+0],
            triangle_list[3*i+1],
            triangle_list[3*i+2]);
        // p[0], p[1], p[2]);
    }

    Result result;

    integrate_2d_surface( V, triangle_list_converted, surface_value_list, boundary_box, N, result.indices, result.values );

    return result;
}

ResultMatrix integrate_2d_surface_mat_converter(
        const dolfin::FunctionSpace & V,
        const std::vector< std::array< double, 3 >>& triangle_list,
        const std::vector< double >& surface_value_list,
        const std::vector< double > & boundary_box,
        const size_t N)
{
    std::vector< Eigen::Vector3d > triangle_list_converted;
    for (auto& p: triangle_list)
    {
        triangle_list_converted.emplace_back(p[0], p[1], p[2]);
    }

    ResultMatrix result;

    integrate_2d_surface_mat( V, triangle_list_converted, surface_value_list, boundary_box, N, result.row_indices, result.col_indices, result.values );

    return result;
}

PYBIND11_MODULE(SIGNATURE, m)
{
    pybind11::class_<Result>(m, "Result")
        .def_readwrite("indices", &Result::indices)
        .def_readwrite("values", &Result::values);

    pybind11::class_<ResultMatrix>(m, "ResultMatrix")
        .def_readwrite("row_indices", &ResultMatrix::row_indices)
        .def_readwrite("col_indices", &ResultMatrix::col_indices)
        .def_readwrite("values", &ResultMatrix::values);

    m.def("integrate_2d_surface", &integrate_2d_surface_converter, "integrates the 2d surface");
    m.def("integrate_2d_surface", &integrate_2d_surface_converter, "integrates the 2d surface");
    m.def("integrate_2d_surface_mat", &integrate_2d_surface_mat_converter, "integrates the 2d surface");
}

"""

compiled = df.compile_cpp_code(code_cpp)


def integrate_2d_surface_python(
    V: df.FunctionSpace,
    triangle_list: List[List[np.array]],
    surface_values: List[float],
    N: int
) -> Tuple[List[np.array], List[np.array]]:

    vertex2dof: np.ndarray[np.int32] = np.array(
        V.dofmap().entity_dofs(V.mesh(), 0), dtype=np.int32)
    local_to_global = np.array(
        V.dofmap().tabulate_local_to_global_dofs(), dtype=np.int32)

    tree = V.mesh().bounding_box_tree()

    dolfin_element = V.dolfin_element()

    indices_list: List[int] = []
    values_list: List[float] = []

    for triangle, surface_value in zip(triangle_list, surface_values):
        p0, p1, p2 = triangle

        d01 = p1 - p0
        d02 = p2 - p0

        area = 0.5 * np.linalg.norm(np.cross(d01, d02))

        len_01 = np.linalg.norm(d01)
        len_02 = np.linalg.norm(d02)

        d01 /= len_01
        d02 /= len_02

        h01 = len_01 / N
        h02 = len_02 / N

        points = [p0 + d01 * h01 * i + d02 * h02 *
                  j for i in range(N+1) for j in range(N+1) if i + j <= N]

        num_points = len(points)

        cell: df.Cell = None
        for point in points:
            point = df.Point(point)
            if cell and cell.contains(point):
                # print('reuse')
                pass
            else:
                # print('no reuse')
                cell_id, distance = tree.compute_closest_entity(point)
                cell = df.Cell(V.mesh(), cell_id)
                assert distance <= 1e-14

            if cell.is_ghost():
                continue

            coordinate_dofs = cell.get_coordinate_dofs()
            cell_orientation = cell.orientation()
            values = dolfin_element.evaluate_basis_all(
                point.array(), coordinate_dofs, cell_orientation)

            v0, v1, v2, v3 = df.vertices(cell)

            indices_local_vertices = np.array(
                [v0.index(), v1.index(), v2.index(), v3.index()], dtype=np.int32)
            indices = local_to_global[vertex2dof[indices_local_vertices]]

            indices_list.append(indices)
            values_list.append(values * area * surface_value / num_points)

    return indices_list, values_list


def integrate_2d_surface(
    V: df.FunctionSpace,
    triangle_list: np.array,
    surface_values: np.array,
    N: int
) -> Tuple[List[np.array], List[np.array]]:
    surface_values = surface_values.tolist()
    triangle_list = triangle_list.tolist()
    triangle_list = [c for p in triangle_list for c in p]
    boundary_box = [
        V.mesh().coordinates()[:,0].min(),
        V.mesh().coordinates()[:,1].min(),
        V.mesh().coordinates()[:,2].min(),
        V.mesh().coordinates()[:,0].max(),
        V.mesh().coordinates()[:,1].max(),
        V.mesh().coordinates()[:,2].max()
    ]
    V.mesh().bounding_box_tree().build(V.mesh())
    result = compiled.integrate_2d_surface(V._cpp_object, triangle_list, surface_values, boundary_box, N)
    indices_list = np.array(result.indices, dtype=np.int32)
    values_list = np.array(result.values)
    return indices_list, values_list


def integrate_2d_surface_mat(
    V: df.FunctionSpace,
    triangle_list: np.array,
    surface_values: np.array,
    N: int
) -> Tuple[List[np.array], List[np.array]]:
    surface_values = surface_values.tolist()
    triangle_list = triangle_list.tolist()
    triangle_list = [np.array(p) for p in triangle_list]
    boundary_box = [
        V.mesh().coordinates()[:,0].min(),
        V.mesh().coordinates()[:,1].min(),
        V.mesh().coordinates()[:,2].min(),
        V.mesh().coordinates()[:,0].max(),
        V.mesh().coordinates()[:,1].max(),
        V.mesh().coordinates()[:,2].max()
    ]
    V.mesh().bounding_box_tree().build(V.mesh())
    result = compiled.integrate_2d_surface_mat(V._cpp_object, triangle_list, surface_values, boundary_box, N)
    row_indices = np.array(result.row_indices, dtype=np.int32)
    col_indices = np.array(result.col_indices, dtype=np.int32)
    values_list = np.array(result.values)
    return row_indices, col_indices, values_list


if __name__ == '__main__':
    N = 32
    num_test_triangles = 0
    M = 8

    mesh = df.BoxMesh(df.Point(0, 0, 0), df.Point(1, 1, 1), N, N, N)
    V = df.FunctionSpace(mesh, 'P', 1)

    triangle_list = [
        np.array([0.2, 0., 0.]),
        np.array([0.2, 1, 0.]),
        np.array([0.2, 0., 1.]),
        np.array([0.8, 0., 0.]),
        np.array([0.8, 0.5, 0.]),
        np.array([0.8, 0., 0.5]),
    ]

    surface_list = [1., 1.]

    for i in range(num_test_triangles):
        triangle_list.append(np.array([0.8, 0., 0.]))
        triangle_list.append(np.array([0.8, 0.005, 0.]))
        triangle_list.append(np.array([0.8, 0., 0.005]))
        surface_list.append(1.)

    V.mesh().bounding_box_tree().build(V.mesh())

    tic = time.perf_counter()
    result = compiled.integrate_2d_surface(
        V._cpp_object,
        triangle_list,
        surface_list,
        M
    )
    toc = time.perf_counter()
    print(f"needed {toc - tic:0.4f} seconds")

    alpha = df.Constant(0.1)
    beta = 1

    f = df.Function(V, name='rhs')
    f_vec = f.vector().vec()

    triangle_list = [
        [np.array([0.2, 0., 0.]),
         np.array([0.2, 1., 0.]),
         np.array([0.2, 0., 1.])],
        [np.array([0.8, 0., 0.]),
         np.array([0.8, 0.5, 0.]),
         np.array([0.8, 0., 0.5])],
    ]

    surface_values = [beta, beta]

    for i in range(num_test_triangles):
        triangle_list.append(
            [np.array([0.8, 0., 0.]),
             np.array([0.8, 0.005, 0.]),
             np.array([0.8, 0., 0.005])])
        surface_values.append(1.)

    tic = time.perf_counter()
    indices_list, values_list = integrate_2d_surface_python(
        V, triangle_list, surface_values, M)
    toc = time.perf_counter()
    print(f"needed {toc - tic:0.4f} seconds")

    indices_list_py = np.array([x for d in indices_list for x in d.tolist()])
    values_list_py = np.array([x for d in values_list for x in d.tolist()])

    indices_list_cpp = np.array(result.indices, dtype=np.int32)
    values_list_cpp = np.array(result.values)

    #print(
    #    f'difference indices {np.linalg.norm(indices_list_py - indices_list_cpp)}')
    #print(
    #    f'difference values  {np.linalg.norm(values_list_py - values_list_cpp)}')

    indices_list = indices_list_cpp
    values_list = values_list_cpp

    f_vec.setValues(indices_list, values_list, p4py.PETSc.InsertMode.ADD)

    print(f'sum = {np.sum(f.vector()[:])}')

    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    a = df.inner(df.grad(u), df.grad(v)) * df.dx + alpha * u * v * df.dx
    bcs = []

    A = df.assemble(a)

    for bc in bcs:
        bc.apply(A)
        bc.apply(f.vector())

    u = df.Function(V, name='solution')

    solver = df.LUSolver()

    solver.solve(A, u.vector(), f.vector())

    file = df.File('output/solution.pvd')
    file << u
