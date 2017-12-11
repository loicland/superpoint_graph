// std::vector to python strcutures wrappers
// Loic Landrieu Dec. 2017 
// note: for some reason, the numpy array wrappers needs the following line in the BOOST_PYTHON_MODULE:
// _import_array();
#include <vector>
#include <numpy/ndarrayobject.h>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
using namespace boost::python;

template<class T>
struct VecToArray
{//convert a vector<T> to a numpy array
 //add to BOOST_PYTHON_MODULE (for float vectors):
 //boost::python::to_python_converter<std::vector<float, std::allocator<float> >, VecToArray<float> >();
 //use: VecToArray<T>::convert(yourvector)
    static PyObject * convert(std::vector<T> vec) {
    npy_intp dims = vec.size();
	PyObject * obj; 
	if (typeid(T) == typeid(int))
	{
    	obj = PyArray_SimpleNew(1, &dims, NPY_INT);
	} else if (typeid(T) == typeid(float))
	{
    	obj = PyArray_SimpleNew(1, &dims, NPY_FLOAT32);
	} else if (typeid(T) == typeid(double))
	{
    	obj = PyArray_SimpleNew(1, &dims, NPY_DOUBLE);
	} else 
	{
		throw("unknown type");
	}
    void * arr_data = PyArray_DATA((PyArrayObject*)obj);
    memcpy(arr_data, &vec[0], dims * sizeof(T));
    return obj;
    }
};

template<class T>
struct VecvecToArray
{//convert a vector< vector<T> > to a numpy 2d array
 //add to BOOST_PYTHON_MODULE (for float vectors):
 // bp::to_python_converter<std::vector<std::vector<float>, std::allocator<std::vector<float> > >, VecvecToArray >();
    static PyObject * convert(std::vector< std::vector<T> > vecvec)
    {
        npy_intp dims[2];
        dims[0] = vecvec.size();
        dims[1] = vecvec[0].size();
        PyObject * obj; 
		if (typeid(T) == typeid(int))
		{
			obj = PyArray_SimpleNew(2, dims, NPY_INT);
		} else if (typeid(T) == typeid(float))
		{
			obj = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
		} else if (typeid(T) == typeid(double))
		{
			obj = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
		} else 
		{
			throw("unknown type");
		}
        void * arr_data = PyArray_DATA((PyArrayObject*)obj);
        std::size_t cell_size = sizeof(float);
        for (std::size_t i = 0; i < dims[0]; i++)
        {
            memcpy(arr_data + i * dims[1] * cell_size, &(vecvec[i][0]), dims[1] * cell_size);
        }
		std::cout << "done" << std::endl;
        return obj;
    }
};


template<class T>
struct VecToList
{//convert a vector<T> to a list
 //add to BOOST_PYTHON_MODULE (for float vectors):
 // bp::to_python_converter<std::vector<float, std::allocator<float> >, VecToList<float> >();
    static PyObject* convert(const std::vector<T>& vec)
    {
        boost::python::list* pylist = new boost::python::list();
        for(size_t i = 0; i < vec.size(); i++) {
            pylist->append(vec[i]);
        }

        return pylist->ptr();
    }
};

template<class T>
struct VecvecToList
{//convert a vector< vector<T> > to a list
 //add to BOOST_PYTHON_MODULE (for float vectors):
	static PyObject* convert(const std::vector< std::vector<T> > & vecvec)
    {
        std::cout << "converting out" << std::endl;
        boost::python::list* pylistlist = new boost::python::list();
        for(size_t i = 0; i < vecvec.size(); i++)
        {
            boost::python::list* pylist = new boost::python::list();
            for(size_t j = 0; j < vecvec[i].size(); j++)
            {
                pylist->append(vecvec[i][j]);
            }
            pylistlist->append((pylist, pylist[0]));
        }
        return pylistlist->ptr();
    }
};
