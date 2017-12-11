#include <iostream>
#include <cstdio>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <numpy/ndarrayobject.h>
#include <limits>
#include "./python_wrapper.h"

namespace bp = boost::python;
namespace ei = Eigen;
namespace bpn = boost::python::numpy;

typedef ei::Matrix<float, 3, 3> Matrix3f;
typedef ei::Matrix<float, 3, 1> Vector3f;
class Array3D {
    size_t width, height;
    std::vector<uint32_t> bin_count;
    std::vector< std::vector<float> > bin_pos;
  public:
    Array3D(size_t x, size_t y, size_t z):
      width(x), height(y), bin_count(x*y*z, 0), bin_pos(x*y*z, std::vector <float>(3,0))
    {}
    uint32_t get_count(size_t x, size_t y, size_t z)
    {
        return bin_count.at(x + y * width + z * width * height);
    }
    std::vector<float> get_pos(uint32_t x_bin, uint32_t y_bin,uint32_t  z_bin)
    {
        return bin_pos.at(x_bin + y_bin * width + z_bin * width * height);
    }
    void add(uint32_t x_bin, uint32_t  y_bin, uint32_t z_bin, float x, float y, float z)
    {
        bin_count.at(x_bin + y_bin * width + z_bin * width * height)
                = bin_count.at(x_bin + y_bin * width + z_bin * width * height)
                + 1;
        bin_pos.at(x_bin + y_bin * width + z_bin * width * height).at(0)
                = bin_pos.at(x_bin + y_bin * width + z_bin * width * height).at(0)
                + x;
        bin_pos.at(x_bin + y_bin * width + z_bin * width * height).at(1)
                = bin_pos.at(x_bin + y_bin * width + z_bin * width * height).at(1)
                + y;
        bin_pos.at(x_bin + y_bin * width + z_bin * width * height).at(2)
                = bin_pos.at(x_bin + y_bin * width + z_bin * width * height).at(2)
                + z;
        return;
    }
};

PyObject * prune(const bpn::ndarray & xyz ,float voxel_size)
{
    std::cout << "=============" << std::endl;
    std::cout << "===pruning===" << std::endl;
    std::cout << "=============" << std::endl;
    std::size_t n_ver = boost::python::len(xyz);
    const float * xyz_data = reinterpret_cast<float*>(xyz.get_data());
    //---find min max of xyz----
    float x_max = std::numeric_limits<float>::min(), x_min = std::numeric_limits<float>::max();
    float y_max = std::numeric_limits<float>::min(), y_min = std::numeric_limits<float>::max();
    float z_max = std::numeric_limits<float>::min(), z_min = std::numeric_limits<float>::max();
    #pragma omp parallel for reduction(max : x_max, y_max, z_max), reduction(min : x_min, y_min, z_min)
    for (std::size_t i_ver = 0; i_ver < n_ver; i_ver ++)
    {
        if (x_max < xyz_data[i_ver])
            x_max = xyz_data[i_ver];
        if (y_max < xyz_data[i_ver + n_ver])
            y_max = xyz_data[i_ver + n_ver];
        if (z_max < xyz_data[i_ver + 2 * n_ver])
            z_max = xyz_data[i_ver + 2 * n_ver];
        if (x_min > xyz_data[i_ver])
            x_min = xyz_data[i_ver];
        if (y_min > xyz_data[i_ver + n_ver])
            y_min = xyz_data[i_ver + n_ver];
        if (z_min > xyz_data[i_ver + 2 * n_ver])
            z_min = xyz_data[i_ver + 2 * n_ver];
    }
    //---accumulate points----
    //std::cout << x_min << " " << x_max << std::endl;
    //std::cout << y_min << " " << y_max << std::endl;
    //std::cout << z_min << " " << z_max << std::endl;
    uint32_t n_bin_x = std::ceil((x_max - x_min) / voxel_size);
    uint32_t n_bin_y = std::ceil((y_max - y_min) / voxel_size);
    uint32_t n_bin_z = std::ceil((z_max - z_min) / voxel_size);
    std::cout << "Voxelization into " << n_bin_x << " x " << n_bin_y << " x " << n_bin_z << " = " << n_bin_x*n_bin_y*n_bin_z << std::endl;
    Array3D voxels(n_bin_x, n_bin_y, n_bin_z);
    #pragma omp parallel for schedule(static)
    for (std::size_t i_ver = 0; i_ver < n_ver; i_ver ++)
    {
        uint32_t bin_x = std::floor((xyz_data[i_ver] - x_min) / voxel_size);
        uint32_t bin_y = std::floor((xyz_data[i_ver + n_ver] - y_min) / voxel_size);
        uint32_t bin_z = std::floor((xyz_data[i_ver + 2 * n_ver] - z_min) / voxel_size);
        voxels.add(bin_x, bin_y, bin_z, xyz_data[i_ver], xyz_data[i_ver + n_ver], xyz_data[i_ver + 2 * n_ver]);
    }
    //---compute pruned graph----
    uint32_t current_index = 0;
    std::vector< std::vector< float > > pruned_xyz;
    pruned_xyz.reserve(n_bin_x * n_bin_y * n_bin_z);
    for (std::size_t i_bin_x = 0; i_bin_x < n_bin_x; i_bin_x ++)
    {
        for (std::size_t i_bin_y = 0; i_bin_y < n_bin_y; i_bin_y ++)
        {
            for (std::size_t i_bin_z = 0; i_bin_z < n_bin_z; i_bin_z ++)
            {
                uint32_t count = voxels.get_count(i_bin_x, i_bin_y, i_bin_z);
                //std::cout << i_bin_x << " " << i_bin_y << " " << i_bin_z << " : " << count << std::endl;
                if (count)
                {
                    std::vector<float> pos = voxels.get_pos(i_bin_x, i_bin_y, i_bin_z);
                    pos.at(0) = pos.at(0) / count;
                    pos.at(1) = pos.at(1) / count;
                    pos.at(2) = pos.at(2) / count;
                    pruned_xyz.push_back(pos);
                }
            }
        }
    }
    std::cout << "Reduced from " << n_ver << " to " << pruned_xyz.size() << "points ("
              << std::ceil(10000 * pruned_xyz.size() / n_ver)/100 << "%)" << std::endl;
    return VecvecToArray<float>::convert(pruned_xyz);
}


PyObject * compute_geof(const bpn::ndarray & xyz ,const bpn::ndarray & target, int k_nn)
{
    std::size_t n_ver = boost::python::len(xyz);
    std::cout <<n_ver << std::endl;
    std::vector< std::vector< float > > geof(n_ver, std::vector< float >(4,0));
    const uint32_t * target_data = reinterpret_cast<uint32_t*>(target.get_data());
    const float * xyz_data = reinterpret_cast<float*>(xyz.get_data());
    std::size_t s_ver = 0;
    #pragma omp parallel for schedule(static)
    for (std::size_t i_ver = 0; i_ver < n_ver; i_ver++)
    {
        //--- compute 3d covariance matrix of neighborhood ---
        ei::MatrixXf position(k_nn+1,3);
        std::size_t i_edg = k_nn * i_ver;
        std::size_t ind_nei;
        position(0,0) = xyz_data[i_ver];
        position(0,1) = xyz_data[i_ver + n_ver];
        position(0,2) = xyz_data[i_ver + 2 * n_ver];
        for (std::size_t i_nei = 0; i_nei < k_nn; i_nei++)
        {
                //add the neighbors to the position matrix
            ind_nei = target_data[i_edg];
            position(i_nei+1,0) = xyz_data[ind_nei];
            position(i_nei+1,1) = xyz_data[ind_nei + n_ver];
            position(i_nei+1,2) = xyz_data[ind_nei + 2 * n_ver];
            i_edg++;
        }
        // compute the covariance matrix
        ei::MatrixXf centered_position = position.rowwise() - position.colwise().mean();
        ei::Matrix3f cov = (centered_position.adjoint() * centered_position) / float(k_nn + 1);
        ei::EigenSolver<Matrix3f> es(cov);
        //--- compute the eigen values and vectors---
        std::vector<float> ev = {es.eigenvalues()[0].real(),es.eigenvalues()[1].real(),es.eigenvalues()[2].real()};
        std::vector<int> indices(3);
        std::size_t n(0);
        std::generate(std::begin(indices), std::end(indices), [&]{ return n++; });
        std::sort(std::begin(indices),std::end(indices),
                       [&](int i1, int i2) { return ev[i1] > ev[i2]; } );
        std::vector<float> lambda = {(std::max(ev[indices[0]],0.f)),
                                    (std::max(ev[indices[1]],0.f)),
                                    (std::max(ev[indices[2]],0.f))};
        std::vector<float> v1 = {es.eigenvectors().col(indices[0])(0).real()
                               , es.eigenvectors().col(indices[0])(1).real()
                               , es.eigenvectors().col(indices[0])(2).real()};
        std::vector<float> v2 = {es.eigenvectors().col(indices[1])(0).real()
                               , es.eigenvectors().col(indices[1])(1).real()
                               , es.eigenvectors().col(indices[1])(2).real()};
        std::vector<float> v3 = {es.eigenvectors().col(indices[2])(0).real()
                               , es.eigenvectors().col(indices[2])(1).real()
                               , es.eigenvectors().col(indices[2])(2).real()};

        //--- compute the geometric features---
        float linearity  = (sqrtf(lambda[0]) - sqrtf(lambda[1])) / sqrtf(lambda[0]);
        float planarity  = (sqrtf(lambda[1]) - sqrtf(lambda[2])) / sqrtf(lambda[0]);
        float scattering =  sqrtf(lambda[2]) / sqrtf(lambda[0]);
        std::vector<float> unary_vector =
            {lambda[0] * fabsf(v1[0]) + lambda[1] * fabsf(v2[0]) + lambda[2] * fabsf(v3[0])
            ,lambda[0] * fabsf(v1[1]) + lambda[1] * fabsf(v2[1]) + lambda[2] * fabsf(v3[1])
            ,lambda[0] * fabsf(v1[2]) + lambda[1] * fabsf(v2[2]) + lambda[2] * fabsf(v3[2])};
        float norm = sqrt(unary_vector[0] * unary_vector[0] + unary_vector[1] * unary_vector[1]
                        + unary_vector[2] * unary_vector[2]);
        float verticality = unary_vector[2] / norm;
        //---fill the geof vector---
        geof[i_ver][0] = linearity;
        geof[i_ver][1] = planarity;
        geof[i_ver][2] = scattering;
        geof[i_ver][3] = verticality;
        //---progression---
        s_ver++;
        if (s_ver % 10000 == 0)
        {
            std::cout << s_ver << "% done          \r" << std::flush;
            std::cout << ceil(s_ver*100/n_ver) << "% done          \r" << std::flush;
        }
    }
    std::cout <<  std::endl;
    return VecvecToArray<float>::convert(geof);
}

using namespace boost::python;
BOOST_PYTHON_MODULE(libply_c)
{
    _import_array();
    //bp::to_python_converter<std::vector<float, std::allocator<float> >, VecToArray<float> >();
    bp::to_python_converter<std::vector<std::vector<float>, std::allocator<std::vector<float> > >, VecvecToArray<float> >();
   /*bp::to_python_converter<std::vector<float, std::allocator<float> >, VecToList<float> >();
    */
    Py_Initialize();
    boost::python::numpy::initialize();
    def("compute_geof", compute_geof);
    def("prune", prune);
    //Py_Finalize();
}
