#include <boost/python.hpp>
#include <iostream>
#include <cstdio>
#include <vector>
#include <../include/API.h>


using namespace boost::python;

typedef std::pair< std::vector< std::vector<int> >, std::vector<int> > Custom_pair;

struct PairToList
{
    static PyObject* convert(const Custom_pair & c_pair)
    {
        boost::python::list* l1 = new boost::python::list();
        for(size_t i = 0; i < c_pair.first.size(); i++)
        {
            boost::python::list* l2 = new boost::python::list();
            for(size_t j = 0; j < c_pair.first.at(i).size(); j++)
            {
                l2->append((uint32_t) c_pair.first.at(i).at(j));
            }
            l1->append((l2, l2[0]));
        }
        boost::python::list* l3 = new boost::python::list();
        for(size_t i = 0; i < c_pair.second.size(); i++) {
            l3->append((uint32_t) c_pair.second[i]);
        }
        boost::python::list* l4 = new boost::python::list();
        l4->append((l1, l1[0]));
        l4->append((l3, l3[0]));
        return l4->ptr();
    }
};



PyObject * cutpursuit(boost::python::list obs, boost::python::list source, boost::python::list target,  float lambda)
{
    std::size_t n_ver = boost::python::len(obs);
    std::size_t n_edg = boost::python::len(source);
    std::size_t n_obs = boost::python::len(obs[0]);
    std::vector< std::vector< float > > obs_vec(n_ver, std::vector< float >(n_obs));
    std::vector< float > node_weight(n_ver,1.);
    std::vector< int > source_vec(n_edg);
    std::vector< int > target_vec(n_edg);
    std::vector< float > edge_weight(n_edg,1.);
    //convert to vectors for random access necessary for //ization
    for (std::size_t i_ver = 0; i_ver < n_ver; i_ver++)
    {
        for (std::size_t i_obs = 0; i_obs < n_obs; i_obs++)
        {
            if (std::isnan(extract<float>(obs[i_ver][i_obs])))
            {
                throw "NaN values in the observation";
            }
            obs_vec[i_ver][i_obs] = extract<float>(obs[i_ver][i_obs]);
        }
    }
    for (std::size_t i_edg = 0; i_edg < n_edg; i_edg++)
    {
       target_vec[i_edg] = extract<int>(target[i_edg][0]);
       source_vec[i_edg] = extract<int>(source[i_edg][0]);
    }
    std::vector< std::vector<float> > solution(n_ver, std::vector<float>(n_obs));
    std::vector<int> in_component(n_ver,0);
    std::vector< std::vector<int> > components(1,std::vector<int>(1,0));
    CP::cut_pursuit(n_ver, n_edg, n_obs, obs_vec, source_vec, target_vec, edge_weight, node_weight
              , solution, in_component, components, lambda, 1.f, 2.f, 2.f);
    return PairToList::convert(Custom_pair(components, in_component));

}


BOOST_PYTHON_MODULE(libcp)
{
    //to_python_converter<std::vector<int, std::allocator<int> >, VecToList<int> >();
    //to_python_converter<std::vector<std::vector<int>, std::allocator<std::vector<int> > >, VecvecToList<int> >();
    //to_python_converter<PyObject *, make_pair<PyObject *, PyObject *> >();
    to_python_converter<Custom_pair, PairToList >();
    def("cutpursuit", cutpursuit);
}

