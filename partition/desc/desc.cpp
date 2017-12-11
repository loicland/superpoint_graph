#include <boost/python.hpp>
#include <iostream>
#include <cstdio>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
using namespace boost::python;
using namespace Eigen;

template<class T>
struct VecToList
{
    static PyObject* convert(const std::vector<T>& vec)
    {
        boost::python::list* l = new boost::python::list();
        for(size_t i = 0; i < vec.size(); i++) {
            l->append(vec[i]);
        }

        return l->ptr();
    }
};

template<class T>
struct VecvecToList
{
    static PyObject* convert(const std::vector< std::vector<T> > & vec)
    {
        boost::python::list* l1 = new boost::python::list();
        for(size_t i = 0; i < vec.size(); i++)
        {
            boost::python::list* l2 = new boost::python::list();
            for(size_t j = 0; j < vec[i].size(); j++)
            {
                l2->append(vec[i][j]);
            }
            l1->append((l2, l2[0]));
        }
        return l1->ptr();
    }
};


typedef Matrix<float, 3, 3> Matrix3f;
typedef Matrix<float, 3, 1> Vector3f;






PyObject* compute_descriptors(boost::python::list xyz, boost::python::list target, std::size_t k_nn_min
          , std::size_t k_nn_max, std::size_t k_nn_step){

    std::cout << "computing descriptors" << std::endl;
    std::size_t n_ver = boost::python::len(xyz);
    std::size_t n_edg = boost::python::len(target);
    std::vector< std::vector< float > > desc(n_ver, std::vector< float >(4));
    std::vector< std::vector< float > > xyz_vec(n_ver, std::vector< float >(3));
    std::vector< int > target_vec(n_edg);
    //convert to vectors for random access necessary for //ization
    for (std::size_t i_ver = 0; i_ver < n_ver; i_ver++)
    {
        xyz_vec[i_ver][0] = extract<float>(xyz[i_ver][0]);
        xyz_vec[i_ver][1] = extract<float>(xyz[i_ver][1]);
        xyz_vec[i_ver][2] = extract<float>(xyz[i_ver][2]);
    }
    for (std::size_t i_edg = 0; i_edg < n_edg; i_edg++)
    {
        target_vec[i_edg] = extract<int>(target[i_edg][0]);
    }
    std::size_t s_ver = 0;
    std::string message = "";
    std::string reverse_message = "";
    #pragma omp parallel for schedule(static)
    for (std::size_t i_ver = 0; i_ver < n_ver; i_ver++)
    {
        float entropy_opt = 1000000;
        EigenSolver<Matrix3f> es_opt;
        //std::ssize_t k_opt;
        //--- select the covariance matrix with the lowest eigentropy ---
        for (std::size_t i_k_nn = k_nn_min; i_k_nn <= k_nn_max;  i_k_nn += k_nn_step)
        {
            //--- compute 3d covariance matrix of neighborhood ---
            MatrixXf position(i_k_nn+1,3);
            //MatrixXf position(i_k_nn,3);
            std::size_t i_edg = k_nn_max * i_ver;
            std::size_t ind_nei;
            position(0,0) = xyz_vec[i_ver][0];
            position(0,1) = xyz_vec[i_ver][1];
            position(0,2) = xyz_vec[i_ver][2];
            for (std::size_t i_nei = 0; i_nei < i_k_nn; i_nei++)
            {
                //add the neighbors to the position matrix
                ind_nei = target_vec[i_edg];
                //position(i_nei,0) = xyz_vec[ind_nei][0];
                //position(i_nei,1) = xyz_vec[ind_nei][1];
                //position(i_nei,2) = xyz_vec[ind_nei][2];
                position(i_nei+1,0) = xyz_vec[ind_nei][0];
                position(i_nei+1,1) = xyz_vec[ind_nei][1];
                position(i_nei+1,2) = xyz_vec[ind_nei][2];
                i_edg++;
            }
            // compute the covariance matrix
            MatrixXf centered_position = position.rowwise() - position.colwise().mean();
            Matrix3f cov = (centered_position.adjoint() * centered_position) / float(i_k_nn + 1);
            EigenSolver<Matrix3f> es(cov);
            std::vector<float> ev = {sqrt(std::max(es.eigenvalues()[0].real(), 0.f))
                                    ,sqrt(std::max(es.eigenvalues()[1].real() ,0.f))
                                    ,sqrt(std::max(es.eigenvalues()[2].real(), 0.f))};
            // compute the eigentropy
            float norm = ev[0] + ev[1] + ev[2];
            ev[0] = ev[0] / norm;
            ev[1] = ev[1] / norm;
            ev[2] = ev[2] / norm;
            float eigentropy = -log(pow(ev[0], ev[0])) - log(pow(ev[1], ev[1])) - log(pow(ev[2], ev[2]));
            //std::cout << i_k_nn << " = " << eigentropy << " : " << ev[0] << " , " << ev[1] << " , " <<ev[2] << std::endl;
            if (eigentropy < entropy_opt)
            {
                entropy_opt = eigentropy;
                es_opt = es;
                //k_opt = i_k_nn;
            }
        }
        //std::cout << "===> " << i_ver << " = " << k_opt << " " << entropy_opt << std::endl;
        //--- compute the eigen values and vectors---
        std::vector<float> ev = {es_opt.eigenvalues()[0].real(),es_opt.eigenvalues()[1].real(),es_opt.eigenvalues()[2].real()};
        std::vector<int> indices(3);
        std::size_t n(0);
        std::generate(std::begin(indices), std::end(indices), [&]{ return n++; });
        std::sort(std::begin(indices),std::end(indices),
                       [&](int i1, int i2) { return ev[i1] > ev[i2]; } );
        std::vector<float> lambda = {(std::max(ev[indices[0]],0.f)),
                                    (std::max(ev[indices[1]],0.f)),
                                    (std::max(ev[indices[2]],0.f))};
        std::vector<float> v1 = {es_opt.eigenvectors().col(indices[0])(0).real()
                               , es_opt.eigenvectors().col(indices[0])(1).real()
                               , es_opt.eigenvectors().col(indices[0])(2).real()};
        std::vector<float> v2 = {es_opt.eigenvectors().col(indices[1])(0).real()
                               , es_opt.eigenvectors().col(indices[1])(1).real()
                               , es_opt.eigenvectors().col(indices[1])(2).real()};
        std::vector<float> v3 = {es_opt.eigenvectors().col(indices[2])(0).real()
                               , es_opt.eigenvectors().col(indices[2])(1).real()
                               , es_opt.eigenvectors().col(indices[2])(2).real()};

        //--- compute the desc---
        float linearity  = (sqrt(lambda[0]) - sqrt(lambda[1])) / sqrt(lambda[0]);
        float planarity  = (sqrt(lambda[1]) - sqrt(lambda[2])) / sqrt(lambda[0]);
        float scattering =  sqrt(lambda[2]) / sqrt(lambda[0]);
        std::vector<float> unary_vector =
            {lambda[0] * fabsf(v1[0]) + lambda[1] * fabsf(v2[0]) + lambda[2] * fabsf(v3[0])
            ,lambda[0] * fabsf(v1[1]) + lambda[1] * fabsf(v2[1]) + lambda[2] * fabsf(v3[1])
            ,lambda[0] * fabsf(v1[2]) + lambda[1] * fabsf(v2[2]) + lambda[2] * fabsf(v3[2])};
        float norm = sqrt(unary_vector[0] * unary_vector[0] + unary_vector[1] * unary_vector[1]
                        + unary_vector[2] * unary_vector[2]);
        float verticality = unary_vector[2] / norm;
        //---fill the descriptor vector---
        desc[i_ver][0] = linearity;
        desc[i_ver][1] = planarity;
        desc[i_ver][2] = scattering;
        desc[i_ver][3] = verticality;
        //---progression---
        s_ver++;
        if (s_ver % 10000 == 0)
        {
            std::cout << s_ver << "% done          \r" << std::flush;
            std::cout << ceil(s_ver*100/n_ver) << "% done          \r" << std::flush;
        }

    }
    std::cout <<  std::endl;
    return VecvecToList<float>::convert(desc);
}

BOOST_PYTHON_MODULE(libdesc)
{
    to_python_converter<std::vector<float, std::allocator<float> >, VecToList<float> >();
    to_python_converter<std::vector<std::vector<float>, std::allocator<std::vector<float> > >, VecvecToList<float> >();
    def("compute_descriptors", compute_descriptors);
}
