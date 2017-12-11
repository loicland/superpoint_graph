#pragma once
#include <omp.h>
#include "CutPursuit_L2.h"
#include "CutPursuit_Linear.h"
#include "CutPursuit_KL.h"
//**********************************************************************************
//*******************************L0-CUT PURSUIT*************************************
//**********************************************************************************
//Greedy graph cut based algorithm to solve the generalized minimal
//partition problem
//
//Cut Pursuit: fast algorithms to learn piecewise constant functions on
//general weighted graphs, Loic Landrieu and Guillaume Obozinski,2016.
//
//Produce a piecewise constant approximation of signal $y$ structured
//by the graph G=(V,e,mu,w) with mu the node weight and w the edgeweight:
//argmin \sum_{i \IN V}{mu_i * phi(x_I, y_I)}
//+ \sum_{(i,j) \IN E}{w_{i,j} 1(x_I != x_J)}
//
//phi(X,Y) the fidelity function (3 are implemented)
//(x != y) the function equal to 1 if x!=y and 0 else
//
// LOIC LANDRIEU 2017
//
//=======================SYNTAX===================================================
//---------------REGULARIZATION---------------------------------------------------
//C style inputs
//void cut_pursuit(const int n_nodes, const int n_edges, const int nObs
//          ,const T * observation, const int * Eu, const int * Ev
//          ,const T * edgeWeight, const T * nodeWeight
//          ,T * solution,  const T lambda, const T mode, const T speed
//          , const float verbose)
//C++ style input
//void cut_pursuit(const int n_nodes, const int n_edges, const int nObs
//          , std::vector< std::vector<T> > & observation
//          , const std::vector<int> & Eu, const std::vector<int> & Ev
//          ,const std::vector<T> & edgeWeight, const std::vector<T> & nodeWeight
//          ,std::vector< std::vector<T> > & solution,  const T lambda, const T mode, const T speed
//          , const float verbose)
// when D = 1
//void cut_pursuit(const int n_nodes, const int n_edges, const int nObs
//          , std::vector<T> & observation
//          , const std::vector<int> & Eu, const std::vector<int> & Ev
//          ,const std::vector<T> & edgeWeight, const std::vector<T> & nodeWeight
//          ,std::vector<T> & solution,  const T lambda, const T mode, const T speed
//         , const float verbose)

//-----INPUT-----
// 1x1 int n_nodes = number of nodes
// 1x1 int n_edges = number of edges
// 1x1 int nObs   = dimension of data on each node
// NxD float observation : the observed signal
// Ex1 int Eu, Ev: the origin and destination of each node
// Ex1 float  edgeWeight: the edge weight
// Nx1 float  nodeWeight: the node weight
// 1x1 float lambda : the regularization strength
// 1x1 float mode : the fidelity function
//      0 : linear (for simplex bound data)
//      1 : quadratic (default)
//   0<a<1: KL with a smoothing (for simplex bound data)
// 1x1 float speed : parametrization impacting performance
//      0 : slow but precise
//      1 : recommended (default)
//      2 : fast but approximated (no backward step)
//      3 : ludicrous - for prototyping (no backward step)
// 1x1 bool verose : verbosity
//      0 : silent
//      1 : recommended (default)
//      2 : chatty
//-----OUTPUT-----
// Nx1 float  solution: piecewise constant approximation
// Nx1 int inComponent: for each node, in which component it belongs
// n_node_redx1 cell components : for each component, list of the nodes
// 1x1 n_node_red : number of components
// 1x1 int n_edges_red : number of edges in reduced graph
// n_edges_redx1 int Eu_red, Ev_red : source and target of reduced edges
// n_edges_redx1 float edgeWeight_red: weights of reduced edges
// n_node_redx1  float nodeWeight_red: weights of reduced nodes
//---------------SEGMENTATION--------------------------------------------------
//for the segmentation, the functions has a few extra argumens allowing to
//record the structrue of the reduced graph
//C++ style input
//void cut_pursuit(const int n_nodes, const int n_edges, const int nObs
//          , std::vector< std::vector<T> > & observation
//          , const std::vector<int> & Eu, const std::vector<int> & Ev
//          ,const std::vector<T> & edgeWeight, const std::vector<T> & nodeWeight
//          ,std::vector< std::vector<T> > & solution,
//          , const std::vector<int> & in_component
//	    , std::vector< std::vector<int> > & components
//          , int & n_nodes_red, int & n_edges_red
//          , std::vector<int> & Eu_red, std::vector<int> & Ev_red
//          , std::vector<T> & edgeWeight_red, std::vector<T> & nodeWeight_red
//  	    , const T lambda, const T mode, const T speed
//          , const float verbose)
//-----EXTRA INPUT-----
// Nx1 int inComponent: for each node, in which component it belongs
// 1x1 n_node_red : number of components
// 1x1 int n_edges_red : number of edges in reduced graph
// n_node_redx1 cell components : for each component, list of the nodes
// n_edges_redx1 int Eu_red, Ev_red : source and target of reduced edges
// n_edges_redx1 float edgeWeight_red: weights of reduced edges
// n_node_redx1  float nodeWeight_red: weights of reduced nodes


namespace CP {
//===========================================================================
//=====================    CREATE_CP      ===================================
//===========================================================================

template<typename T>
CP::CutPursuit<T> * create_CP(const T mode, const float verbose)
{
    CP::CutPursuit<float> * cp = NULL;
    fidelityType fidelity = L2;
    if (mode == 0)
    {
        if (verbose > 0)
        {
            std::cout << " WITH LINEAR FIDELITY" << std::endl;
        }
        fidelity = linear;
        cp = new CP::CutPursuit_Linear<float>();
     }
     else if (mode == 1)
     {
        if (verbose > 0)
        {
            std::cout << " WITH L2 FIDELITY" << std::endl;
        }
        fidelity = L2;
        cp = new CP::CutPursuit_L2<float>();
     }
     else if (mode > 0 && mode < 1)
     {
        if (verbose > 0)
        {
            std::cout << " WITH KULLBACK-LEIBLER FIDELITY SMOOTHING : "
                      << mode << std::endl;
        }
        fidelity = KL;
        cp = new CP::CutPursuit_KL<float>();
        cp->parameter.smoothing = mode;
     }
     else
     {
        std::cout << " UNKNOWN MODE, SWICTHING TO L2 FIDELITY"
                << std::endl;
        fidelity = L2;
        cp = new CP::CutPursuit_L2<float>();
     }
     cp->parameter.fidelity = fidelity;
     cp->parameter.verbose  = verbose;
     return cp;
}
//===========================================================================
//=====================  cut_pursuit  C-style  ==============================
//===========================================================================
template<typename T>
void cut_pursuit(const int n_nodes, const int n_edges, const int nObs
          ,const T * observation, const int * Eu, const int * Ev
          ,const T * edgeWeight, const T * nodeWeight
          ,T * solution,  const T lambda, const T mode, const T speed
          , const float verbose)
{   //C-style interface
    std::srand (1);
    if (verbose > 0)
    {
        std::cout << "L0-CUT PURSUIT";
    }
    //--------parameterization---------------------------------------------
    CP::CutPursuit<T> * cp = create_CP(mode, verbose);
    set_speed(cp, speed, verbose);
    set_up_CP(cp, n_nodes, n_edges, nObs, observation, Eu, Ev
             ,edgeWeight, nodeWeight);
    cp->parameter.reg_strenth = lambda;
    //-------run the optimization------------------------------------------
    cp->run();
    //------------write the solution-----------------------------
    VertexAttributeMap<T> vertex_attribute_map = boost::get(
            boost::vertex_bundle, cp->main_graph);
    std::size_t ind_sol = 0;	
    VertexIterator<T> ite_nod = boost::vertices(cp->main_graph).first;
    for(int ind_nod = 0; ind_nod < n_nodes; ind_nod++ )
    {        
        for(int i_dim=0; i_dim < nObs; i_dim++)
        {
            solution[ind_sol] = vertex_attribute_map[*ite_nod].value[i_dim];
            ind_sol++;
        }
        ite_nod++;
   }
    delete cp;
    return;
}

//===========================================================================
//=====================  cut_pursuit  C++-style  ============================
//===========================================================================
template<typename T>
void cut_pursuit(const int n_nodes, const int n_edges, const int nObs
          , std::vector< std::vector<T> > & observation
          , const std::vector<int> & Eu, const std::vector<int> & Ev
          , const std::vector<T> & edgeWeight, const std::vector<T> & nodeWeight
          , std::vector< std::vector<T> > & solution,  const T lambda, const T mode, const T speed
          , const float verbose)
{   //C-style ++ interface
    std::srand (1);
    if (verbose > 0)
    {
        std::cout << "L0-CUT PURSUIT";
    }
    //--------parameterization---------------------------------------------
    CP::CutPursuit<T> * cp = create_CP(mode, verbose);
    set_speed(cp, speed, verbose);
    set_up_CP(cp, n_nodes, n_edges, nObs, observation, Eu, Ev
             ,edgeWeight, nodeWeight);
    cp->parameter.reg_strenth = lambda;
    //-------run the optimization------------------------------------------
    cp->run();
    //------------write the solution-----------------------------
    VertexAttributeMap<T> vertex_attribute_map = boost::get(
            boost::vertex_bundle, cp->main_graph);
    VertexIterator<T> ite_nod = boost::vertices(cp->main_graph).first;
    for(int ind_nod = 0; ind_nod < n_nodes; ind_nod++ )
    {        
        for(int ind_dim=0; ind_dim < nObs; ind_dim++)
        {
            solution[ind_nod][ind_dim] = vertex_attribute_map[*ite_nod].value[ind_dim];
        }
        ite_nod++;
   }
    delete cp;
    return;
}
//===========================================================================
//=====================  cut_pursuit  C++-style  ============================
//===========================================================================
template<typename T>
void cut_pursuit(const int n_nodes, const int n_edges, const int nObs
          , std::vector<T>& observation
          , const std::vector<int> & Eu, const std::vector<int> & Ev
          , const std::vector<T> & edgeWeight, const std::vector<T> & nodeWeight
          , std::vector<T> & solution,  const T lambda, const T mode, const T speed
          , const float verbose)
{   //C-style ++ interface
    std::srand (1);
    if (verbose > 0)
    {
        std::cout << "L0-CUT PURSUIT";
    }
    //--------parameterization---------------------------------------------
    CP::CutPursuit<T> * cp = create_CP(mode, verbose);
    set_speed(cp, speed, verbose);
    set_up_CP(cp, n_nodes, n_edges, nObs, observation, Eu, Ev
             ,edgeWeight, nodeWeight);
    cp->parameter.reg_strenth = lambda;
    //-------run the optimization------------------------------------------
    cp->run();
    //------------write the solution-----------------------------
    VertexAttributeMap<T> vertex_attribute_map = boost::get(
            boost::vertex_bundle, cp->main_graph);
    VertexIterator<T> ite_nod = boost::vertices(cp->main_graph).first;
    for(int ind_nod = 0; ind_nod < n_nodes; ind_nod++ )
    {        
	solution[ind_nod] = vertex_attribute_map[*ite_nod].value[0];
        ite_nod++;
    }
    delete cp;
    return;
}

//===========================================================================
//=====================  cut_pursuit segmentation C++-style  ================
//===========================================================================
template<typename T>
void cut_pursuit(const int n_nodes, const int n_edges, const int nObs
          , std::vector< std::vector<T> > & observation
          , const std::vector<int> & Eu, const std::vector<int> & Ev
          , const std::vector<T> & edgeWeight, const std::vector<T> & nodeWeight
          , std::vector< std::vector<T> > & solution
	  , std::vector<int> & in_component, std::vector< std::vector<int> > & components
          , int & n_nodes_red, int & n_edges_red
          , std::vector<int> & Eu_red, std::vector<int> & Ev_red
          , std::vector<T> & edgeWeight_red, std::vector<T> & nodeWeight_red
  	  , const T lambda, const T mode, const T speed
          , const float verbose)
{   //C-style ++ interface
    std::srand (1);
    if (verbose > 0)
    {
        std::cout << "L0-CUT PURSUIT";
    }
    //--------parameterization---------------------------------------------
    CP::CutPursuit<T> * cp = create_CP(mode, verbose);

    set_speed(cp, speed, verbose);
    set_up_CP(cp, n_nodes, n_edges, nObs, observation, Eu, Ev
             ,edgeWeight, nodeWeight);
    cp->parameter.reg_strenth = lambda;
    //-------run the optimization------------------------------------------
    cp->run();
    cp->compute_reduced_graph();
    //------------resize the vectors-----------------------------
    n_nodes_red = boost::num_vertices(cp->reduced_graph);
    n_edges_red = boost::num_edges(cp->reduced_graph);
    in_component.resize(n_nodes);
    components.resize(n_nodes_red);
    Eu_red.resize(n_edges_red);
    Ev_red.resize(n_edges_red);
    edgeWeight_red.resize(n_edges_red);
    nodeWeight_red.resize(n_nodes_red);
    //------------write the solution-----------------------------
    VertexAttributeMap<T> vertex_attribute_map = boost::get(
            boost::vertex_bundle, cp->main_graph);
    VertexIterator<T> ite_nod = boost::vertices(cp->main_graph).first;
    for(int ind_nod = 0; ind_nod < n_nodes; ind_nod++ )
    {        
        for(int ind_dim=0; ind_dim < nObs; ind_dim++)
        {
            solution[ind_nod][ind_dim] = vertex_attribute_map[*ite_nod].value[ind_dim];
        }
        ite_nod++;
    }
  
    //------------fill the components-----------------------------
    VertexIndexMap<T> vertex_index_map = get(boost::vertex_index, cp->main_graph);
    for(int ind_nod_red = 0; ind_nod_red < n_nodes_red; ind_nod_red++ )
    {
	std::size_t component_size = cp->components[ind_nod_red].size();
        components[ind_nod_red] = std::vector<int>(component_size, 0);
	for(std::size_t ind_nod = 0; ind_nod < component_size; ind_nod++ )
    	{
	    components[ind_nod_red][ind_nod] = vertex_index_map(cp->components[ind_nod_red][ind_nod]);
	}	
    }
    //------------write the reduced graph-----------------------------
    VertexAttributeMap<T> vertex_attribute_map_red = boost::get(
            boost::vertex_bundle, cp->reduced_graph);
    EdgeAttributeMap<T> edge_attribute_map_red = boost::get(
            boost::edge_bundle, cp->reduced_graph);
    VertexIndexMap<T> vertex_index_map_red = get(boost::vertex_index, cp->reduced_graph);
    ite_nod = boost::vertices(cp->main_graph).first;
    for(int ind_nod = 0; ind_nod < n_nodes; ind_nod++ )
    {
        in_component[ind_nod] = vertex_attribute_map[*ite_nod].in_component;
        ite_nod++;
    }
    ite_nod = boost::vertices(cp->reduced_graph).first;
    for(int ind_nod_red = 0; ind_nod_red < n_nodes_red; ind_nod_red++ )
    {
	nodeWeight_red[ind_nod_red] = vertex_attribute_map_red[*ite_nod].weight;
        ite_nod++;
    }
    EdgeIterator<T> ite_edg = boost::edges(cp->reduced_graph).first;
    for(int ind_edg = 0; ind_edg < n_edges_red; ind_edg++ )
    {    
	edgeWeight_red[ind_edg] = edge_attribute_map_red[*ite_edg].weight;
	Eu_red[ind_edg] = vertex_index_map_red(boost::source(*ite_edg, cp->reduced_graph));
	Ev_red[ind_edg] = vertex_index_map_red(boost::target(*ite_edg, cp->reduced_graph));        
        ite_edg++;
    }
    delete cp;
    return;
}


//===========================================================================
//=====================  cut_pursuit segmentation light C++-style  ================
//===========================================================================
template<typename T>
void cut_pursuit(const int n_nodes, const int n_edges, const int nObs
          , std::vector< std::vector<T> > & observation
          , const std::vector<int> & Eu, const std::vector<int> & Ev
          , const std::vector<T> & edgeWeight, const std::vector<T> & nodeWeight
          , std::vector< std::vector<T> > & solution
	  , std::vector<int> & in_component, std::vector< std::vector<int> > & components
  	  , const T lambda, const T mode, const T speed
          , const float verbose)
{   //C-style ++ interface
    std::srand (1);
    if (verbose > 0)
    {
        std::cout << "L0-CUT PURSUIT";
    }
    //--------parameterization---------------------------------------------
    CP::CutPursuit<T> * cp = create_CP(mode, verbose);

    set_speed(cp, speed, verbose);
    set_up_CP(cp, n_nodes, n_edges, nObs, observation, Eu, Ev
             ,edgeWeight, nodeWeight);
    cp->parameter.reg_strenth = lambda;
    //-------run the optimization------------------------------------------
    cp->run();
    cp->compute_reduced_graph();
    //------------resize the vectors-----------------------------
    std::size_t n_nodes_red = boost::num_vertices(cp->reduced_graph);
    in_component.resize(n_nodes);
    components.resize(n_nodes_red);
    //------------write the solution-----------------------------
    VertexAttributeMap<T> vertex_attribute_map = boost::get(
            boost::vertex_bundle, cp->main_graph);
    VertexIterator<T> ite_nod = boost::vertices(cp->main_graph).first;
    for(int ind_nod = 0; ind_nod < n_nodes; ind_nod++ )
    {        
        for(int ind_dim=0; ind_dim < nObs; ind_dim++)
        {
            solution[ind_nod][ind_dim] = vertex_attribute_map[*ite_nod].value[ind_dim];
        }
        ite_nod++;
    }
    //------------fill the components-----------------------------
    VertexIndexMap<T> vertex_index_map = get(boost::vertex_index, cp->main_graph);
    for(int ind_nod_red = 0; ind_nod_red < n_nodes_red; ind_nod_red++ )
    {
		std::size_t component_size = cp->components[ind_nod_red].size();
		    components[ind_nod_red] = std::vector<int>(component_size, 0);
		for(std::size_t ind_nod = 0; ind_nod < component_size; ind_nod++ )
		{
			components[ind_nod_red][ind_nod] = vertex_index_map(cp->components[ind_nod_red][ind_nod]);
		}	
    }
    ite_nod = boost::vertices(cp->main_graph).first;
    for(int ind_nod = 0; ind_nod < n_nodes; ind_nod++ )
    {
        in_component[ind_nod] = vertex_attribute_map[*ite_nod].in_component;
        ite_nod++;
    }
    delete cp;
    return;
}

//===========================================================================
//=====================  cut_pursuit segmentation C++-style  ================
//===========================================================================
template<typename T>
void cut_pursuit(const int n_nodes, const int n_edges, const int nObs
          , std::vector< std::vector<T> > & observation
          , const std::vector<int> & Eu, const std::vector<int> & Ev
          , const std::vector<T> & edgeWeight, const std::vector<T> & nodeWeight
          , std::vector< std::vector<T> > & solution
	  , std::vector<int> & in_component, std::vector< std::vector<int> > & components
          , std::vector< std::vector<int> > & borders
          , int & n_nodes_red, int & n_edges_red
          , std::vector<int> & Eu_red, std::vector<int> & Ev_red
          , std::vector<T> & edgeWeight_red, std::vector<T> & nodeWeight_red
  	  , const T lambda, const T mode, const T speed
          , const float verbose)
{   //C-style ++ interface
    std::srand (1);
    if (verbose > 0)
    {
        std::cout << "L0-CUT PURSUIT";
    }
    //--------parameterization---------------------------------------------
    CP::CutPursuit<T> * cp = create_CP(mode, verbose);

    set_speed(cp, speed, verbose);
    set_up_CP(cp, n_nodes, n_edges, nObs, observation, Eu, Ev
             ,edgeWeight, nodeWeight);
    cp->parameter.reg_strenth = lambda;
    //-------run the optimization------------------------------------------
    cp->run();
    cp->compute_reduced_graph();
    //------------resize the vectors-----------------------------
    n_nodes_red = boost::num_vertices(cp->reduced_graph);
    n_edges_red = boost::num_edges(cp->reduced_graph);
    in_component.resize(n_nodes);
    components.resize(n_nodes_red);
    Eu_red.resize(n_edges_red);
    Ev_red.resize(n_edges_red);
    edgeWeight_red.resize(n_edges_red);
    nodeWeight_red.resize(n_nodes_red);
    borders.resize(n_edges_red);
    //------------write the solution-----------------------------
    VertexAttributeMap<T> vertex_attribute_map = boost::get(
            boost::vertex_bundle, cp->main_graph);
    VertexIterator<T> ite_nod = boost::vertices(cp->main_graph).first;
    for(int ind_nod = 0; ind_nod < n_nodes; ind_nod++ )
    {        
        for(int ind_dim=0; ind_dim < nObs; ind_dim++)
        {
            solution[ind_nod][ind_dim] = vertex_attribute_map[*ite_nod].value[ind_dim];
        }
        ite_nod++;
    }
  
    //------------fill the components-----------------------------
    VertexIndexMap<T> vertex_index_map = get(boost::vertex_index, cp->main_graph);
    for(int ind_nod_red = 0; ind_nod_red < n_nodes_red; ind_nod_red++ )
    {
	std::size_t component_size = cp->components[ind_nod_red].size();
        components[ind_nod_red] = std::vector<int>(component_size, 0);
	for(std::size_t ind_nod = 0; ind_nod < component_size; ind_nod++ )
    	{
	    components[ind_nod_red][ind_nod] = vertex_index_map(cp->components[ind_nod_red][ind_nod]);
	}	
    }
    //------------write the reduced graph-----------------------------
    VertexAttributeMap<T> vertex_attribute_map_red = boost::get(
            boost::vertex_bundle, cp->reduced_graph);
    EdgeAttributeMap<T> edge_attribute_map_red = boost::get(
            boost::edge_bundle, cp->reduced_graph);
    VertexIndexMap<T> vertex_index_map_red = get(boost::vertex_index, cp->reduced_graph);
    ite_nod = boost::vertices(cp->main_graph).first;
    for(int ind_nod = 0; ind_nod < n_nodes; ind_nod++ )
    {
        in_component[ind_nod] = vertex_attribute_map[*ite_nod].in_component;
        ite_nod++;
    }
    ite_nod = boost::vertices(cp->reduced_graph).first;
    for(int ind_nod_red = 0; ind_nod_red < n_nodes_red; ind_nod_red++ )
    {
	nodeWeight_red[ind_nod_red] = vertex_attribute_map_red[*ite_nod].weight;
        ite_nod++;
    }
    EdgeIterator<T> ite_edg_red = boost::edges(cp->reduced_graph).first;
    for(int ind_edg_red = 0; ind_edg_red < n_edges_red; ind_edg_red++ )
    {    
	edgeWeight_red[ind_edg_red] = edge_attribute_map_red[*ite_edg_red].weight;
	Eu_red[ind_edg_red] = vertex_index_map_red(boost::source(*ite_edg_red, cp->reduced_graph));
	Ev_red[ind_edg_red] = vertex_index_map_red(boost::target(*ite_edg_red, cp->reduced_graph)); 
	//the order in which the edges are scanned with the iterator doesn not necessarily follow their index
	std::size_t  ind_edg_red_in_graph = edge_attribute_map_red[*ite_edg_red].index;
        for(int ind_edg = 0; ind_edg < cp->borders[ind_edg_red_in_graph].size(); ind_edg++ )
    	{
	    //we have to divide the index by 2 to find the corresponding index of input Eu Ev, because of the doubling of edges
	    borders[ind_edg_red].push_back(edge_attribute_map_red[cp->borders[ind_edg_red_in_graph][ind_edg]].index/2);	  
	}	   
        ite_edg_red++;
    }
    delete cp;
    return;
}

//===========================================================================
//=====================  cut_pursuit segmentation C++-style D = 1============
//===========================================================================
template<typename T>
void cut_pursuit(const int n_nodes, const int n_edges, const int nObs
          , std::vector<T> & observation
          , const std::vector<int> & Eu, const std::vector<int> & Ev
          , const std::vector<T> & edgeWeight, const std::vector<T> & nodeWeight
          , std::vector<T> & solution
	  , std::vector<int> & in_component, std::vector< std::vector<int> > & components
          , int & n_nodes_red, int & n_edges_red
          , std::vector<int> & Eu_red, std::vector<int> & Ev_red
          , std::vector<T> & edgeWeight_red, std::vector<T> & nodeWeight_red
  	  , const T lambda, const T mode, const T speed
          , const float verbose)
{   //C-style ++ interface
    std::srand (1);
    if (verbose > 0)
    {
        std::cout << "L0-CUT PURSUIT";
    }
    //--------parameterization---------------------------------------------
    CP::CutPursuit<T> * cp = create_CP(mode, verbose);

    set_speed(cp, speed, verbose);
    set_up_CP(cp, n_nodes, n_edges, nObs, observation, Eu, Ev
             ,edgeWeight, nodeWeight);
    cp->parameter.reg_strenth = lambda;
    //-------run the optimization------------------------------------------
    cp->run();
    cp->compute_reduced_graph();
    //------------resize the vectors-----------------------------
    n_nodes_red = boost::num_vertices(cp->reduced_graph);
    n_edges_red = boost::num_edges(cp->reduced_graph);
    in_component.resize(n_nodes);
    components.resize(n_nodes_red);
    Eu_red.resize(n_edges_red);
    Ev_red.resize(n_edges_red);
    edgeWeight_red.resize(n_edges_red);
    nodeWeight_red.resize(n_nodes_red);
    //------------write the solution-----------------------------
    VertexAttributeMap<T> vertex_attribute_map = boost::get(
            boost::vertex_bundle, cp->main_graph);
    VertexIterator<T> ite_nod = boost::vertices(cp->main_graph).first;
    for(int ind_nod = 0; ind_nod < n_nodes; ind_nod++ )
    {        

        solution[ind_nod] = vertex_attribute_map[*ite_nod].value[0];
        ite_nod++;
    }
  
    //------------fill the components-----------------------------
    VertexIndexMap<T> vertex_index_map = get(boost::vertex_index, cp->main_graph);
    for(int ind_nod_red = 0; ind_nod_red < n_nodes_red; ind_nod_red++ )
    {
	std::size_t component_size = cp->components[ind_nod_red].size();
        components[ind_nod_red] = std::vector<int>(component_size, 0);
	for(std::size_t ind_nod = 0; ind_nod < component_size; ind_nod++ )
    	{
	    components[ind_nod_red][ind_nod] = vertex_index_map(cp->components[ind_nod_red][ind_nod]);
	}	
    }
    //------------write the reduced graph-----------------------------
    VertexAttributeMap<T> vertex_attribute_map_red = boost::get(
            boost::vertex_bundle, cp->reduced_graph);
    EdgeAttributeMap<T> edge_attribute_map_red = boost::get(
            boost::edge_bundle, cp->reduced_graph);
    VertexIndexMap<T> vertex_index_map_red = get(boost::vertex_index, cp->reduced_graph);
    ite_nod = boost::vertices(cp->main_graph).first;
    for(int ind_nod = 0; ind_nod < n_nodes; ind_nod++ )
    {
        in_component[ind_nod] = vertex_attribute_map[*ite_nod].in_component;
        ite_nod++;
    }
    ite_nod = boost::vertices(cp->reduced_graph).first;
    for(int ind_nod_red = 0; ind_nod_red < n_nodes_red; ind_nod_red++ )
    {
	nodeWeight_red[ind_nod_red] = vertex_attribute_map_red[*ite_nod].weight;
        ite_nod++;
    }
    EdgeIterator<T> ite_edg_red = boost::edges(cp->reduced_graph).first;
    for(int ind_edg_red = 0; ind_edg_red < n_edges_red; ind_edg_red++ )
    {    
	edgeWeight_red[ind_edg_red] = edge_attribute_map_red[*ite_edg_red].weight;
	Eu_red[ind_edg_red] = vertex_index_map_red(boost::source(*ite_edg_red, cp->reduced_graph));
	Ev_red[ind_edg_red] = vertex_index_map_red(boost::target(*ite_edg_red, cp->reduced_graph)); 
	//the order in which the edges are scanned with the iterator doesn not necessarily follow their index
        ite_edg_red++;
    }
    delete cp;
    return;
}
//===========================================================================
//=====================  cut_pursuit_segmentation_full C++-style D = 1============
//===========================================================================
template<typename T>
void cut_pursuit(const int n_nodes, const int n_edges, const int nObs
          , std::vector<T> & observation
          , const std::vector<int> & Eu, const std::vector<int> & Ev
          , const std::vector<T> & edgeWeight, const std::vector<T> & nodeWeight
          , std::vector<T> & solution
	  , std::vector<int> & in_component, std::vector< std::vector<int> > & components
	  , std::vector< std::vector<int> > & borders
          , int & n_nodes_red, int & n_edges_red
          , std::vector<int> & Eu_red, std::vector<int> & Ev_red
          , std::vector<T> & edgeWeight_red, std::vector<T> & nodeWeight_red
  	  , const T lambda, const T mode, const T speed
          , const float verbose)
{   //C-style ++ interface
    std::srand (1);
    if (verbose > 0)
    {
        std::cout << "L0-CUT PURSUIT";
    }
    //--------parameterization---------------------------------------------
    CP::CutPursuit<T> * cp = create_CP(mode, verbose);

    set_speed(cp, speed, verbose);
    set_up_CP(cp, n_nodes, n_edges, nObs, observation, Eu, Ev
             ,edgeWeight, nodeWeight);
    cp->parameter.reg_strenth = lambda;
    //-------run the optimization------------------------------------------
    cp->run();
    cp->compute_reduced_graph();
    //------------resize the vectors-----------------------------
    n_nodes_red = boost::num_vertices(cp->reduced_graph);
    n_edges_red = boost::num_edges(cp->reduced_graph);
    in_component.resize(n_nodes);
    components.resize(n_nodes_red);
    borders.resize(n_edges_red);
    Eu_red.resize(n_edges_red);
    Ev_red.resize(n_edges_red);
    edgeWeight_red.resize(n_edges_red);
    nodeWeight_red.resize(n_nodes_red);
    //------------write the solution-----------------------------
    VertexAttributeMap<T> vertex_attribute_map = boost::get(
            boost::vertex_bundle, cp->main_graph);
    VertexIterator<T> ite_nod = boost::vertices(cp->main_graph).first;
    for(int ind_nod = 0; ind_nod < n_nodes; ind_nod++ )
    {        
        solution[ind_nod] = vertex_attribute_map[*ite_nod].value[0];
        ite_nod++;
    }
    //------------fill the components-----------------------------
    VertexIndexMap<T> vertex_index_map = get(boost::vertex_index, cp->main_graph);
    for(int ind_nod_red = 0; ind_nod_red < n_nodes_red; ind_nod_red++ )
    {
	std::size_t component_size = cp->components[ind_nod_red].size();
        components[ind_nod_red] = std::vector<int>(component_size, 0);
	for(std::size_t ind_nod = 0; ind_nod < component_size; ind_nod++ )
    	{
	    components[ind_nod_red][ind_nod] = vertex_index_map(cp->components[ind_nod_red][ind_nod]);
	}	
    }

    //------------write the reduced graph-----------------------------
    VertexAttributeMap<T> vertex_attribute_map_red = boost::get(
            boost::vertex_bundle, cp->reduced_graph);
    EdgeAttributeMap<T> edge_attribute_map_red = boost::get(
            boost::edge_bundle, cp->reduced_graph);
    VertexIndexMap<T> vertex_index_map_red = get(boost::vertex_index, cp->reduced_graph);
    ite_nod = boost::vertices(cp->main_graph).first;
    for(int ind_nod = 0; ind_nod < n_nodes; ind_nod++ )
    {
        in_component[ind_nod] = vertex_attribute_map[*ite_nod].in_component;
        ite_nod++;
    }
    ite_nod = boost::vertices(cp->reduced_graph).first;
    for(int ind_nod_red = 0; ind_nod_red < n_nodes_red; ind_nod_red++ )
    {
	nodeWeight_red[ind_nod_red] = vertex_attribute_map_red[*ite_nod].weight;
        ite_nod++;
    }
    EdgeIterator<T> ite_edg_red = boost::edges(cp->reduced_graph).first;
    for(int ind_edg_red = 0; ind_edg_red < n_edges_red; ind_edg_red++ )
    {   
	edgeWeight_red[ind_edg_red] = edge_attribute_map_red[*ite_edg_red].weight;
	Eu_red[ind_edg_red] = vertex_index_map_red(boost::source(*ite_edg_red, cp->reduced_graph));
	Ev_red[ind_edg_red] = vertex_index_map_red(boost::target(*ite_edg_red, cp->reduced_graph));  
        for(int ind_edg = 0; ind_edg < cp->borders.size(); ind_edg++ )
    	{
	    borders[ind_edg_red].push_back(edge_attribute_map_red[cp->borders[ind_edg_red]].index);	  
	}	   
        ite_edg_red++;
    }
    delete cp;
    return;
}

//===========================================================================
//=====================     SET_UP_CP C style   =============================
//===========================================================================
template<typename T>
void set_up_CP(CP::CutPursuit<T> * cp, const int n_nodes, const int n_edges, const int nObs
               ,const T * observation, const int * Eu, const int * Ev
               ,const T * edgeWeight, const T * nodeWeight)
{
    cp->main_graph = Graph<T>(n_nodes);
    cp->dim = nObs;
    //--------fill the vertices--------------------------------------------
    VertexAttributeMap<T> vertex_attribute_map = boost::get(
            boost::vertex_bundle, cp->main_graph);
    VertexIterator<T> ite_nod = boost::vertices(cp->main_graph).first;
    //the node attributes used to fill each node
    std::size_t ind_obs = 0;
    for(int ind_nod = 0; ind_nod < n_nodes; ind_nod++ )
    {
        VertexAttribute<T> v_attribute (nObs);
        for(int i_dim=0; i_dim < nObs; i_dim++)
        { //fill the observation of v_attribute
            v_attribute.observation[i_dim] = observation[ind_obs];
            ind_obs++;
        }//and its weight
        v_attribute.weight = nodeWeight[ind_nod];
        //set the attributes of the current node
        vertex_attribute_map[*ite_nod++] = v_attribute;
    }
    //--------build the edges-----------------------------------------------
    EdgeAttributeMap<T> edge_attribute_map = boost::get(boost::edge_bundle
            , cp->main_graph);
	std::size_t true_ind_edg = 0; //this index count the number of edges ACTUALLY added
    for( int ind_edg = 0; ind_edg < n_edges; ind_edg++ )
    {   //add edges in each direction
        if (addDoubledge(cp->main_graph, boost::vertex(Eu[ind_edg]
                    , cp->main_graph), boost::vertex(Ev[ind_edg]
                    , cp->main_graph), edgeWeight[ind_edg],2 * ind_edg
                    , edge_attribute_map))
		{
			true_ind_edg += 2;	
		}
    }
}

//===========================================================================
//=====================     SET_UP_CP C++ style  ============================
//===========================================================================
template<typename T>
void set_up_CP(CP::CutPursuit<T> * cp, const int n_nodes, const int n_edges, const int nObs
               ,const std::vector< std::vector<T>> observation, const std::vector<int> Eu, const std::vector<int> Ev
               ,const std::vector<T> edgeWeight, const std::vector<T> nodeWeight)
{
    cp->main_graph = Graph<T>(n_nodes);
    cp->dim = nObs;
    //--------fill the vertices--------------------------------------------
    VertexAttributeMap<T> vertex_attribute_map = boost::get(
            boost::vertex_bundle, cp->main_graph);
    VertexIterator<T> ite_nod = boost::vertices(cp->main_graph).first;
    //the node attributes used to fill each node
    for(int ind_nod = 0; ind_nod < n_nodes; ind_nod++ )
    {
        VertexAttribute<T> v_attribute (nObs);
        for(int i_dim=0; i_dim < nObs; i_dim++)
        { //fill the observation of v_attribute
            v_attribute.observation[i_dim] = observation[ind_nod][i_dim];
        }//and its weight
        v_attribute.weight = nodeWeight[ind_nod];
        //set the attributes of the current node
        vertex_attribute_map[*ite_nod++] = v_attribute;
    }
    //--------build the edges-----------------------------------------------
    EdgeAttributeMap<T> edge_attribute_map = boost::get(boost::edge_bundle
            , cp->main_graph);
    std::size_t true_ind_edg = 0; //this index count the number of edges ACTUALLY added
    for( int ind_edg = 0; ind_edg < n_edges; ind_edg++ )
    {   //add edges in each direction	
        if (addDoubledge(cp->main_graph, boost::vertex(Eu[ind_edg]
                    , cp->main_graph), boost::vertex(Ev[ind_edg]
                    , cp->main_graph), edgeWeight[ind_edg], true_ind_edg
                    , edge_attribute_map))
		{
			true_ind_edg += 2;
		}

    }
}
//===========================================================================
//=====================     SET_UP_CP C++ style D = 1========================
//===========================================================================
template<typename T>
void set_up_CP(CP::CutPursuit<T> * cp, const int n_nodes, const int n_edges, const int nObs
               ,const std::vector<T> observation, const std::vector<int> Eu, const std::vector<int> Ev
               ,const std::vector<T> edgeWeight, const std::vector<T> nodeWeight)
{
    cp->main_graph = Graph<T>(n_nodes);
    cp->dim = nObs;
    //--------fill the vertices--------------------------------------------
    VertexAttributeMap<T> vertex_attribute_map = boost::get(
            boost::vertex_bundle, cp->main_graph);
    VertexIterator<T> ite_nod = boost::vertices(cp->main_graph).first;
    //the node attributes used to fill each node
    for(int ind_nod = 0; ind_nod < n_nodes; ind_nod++ )
    {
        VertexAttribute<T> v_attribute (nObs);
        //fill the observation of v_attribute
        v_attribute.observation[0] = observation[ind_nod];
        //and its weight
        v_attribute.weight = nodeWeight[ind_nod];
        //set the attributes of the current node
        vertex_attribute_map[*ite_nod++] = v_attribute;
    }
    //--------build the edges-----------------------------------------------
    EdgeAttributeMap<T> edge_attribute_map = boost::get(boost::edge_bundle
            , cp->main_graph);
    for( int ind_edg = 0; ind_edg < n_edges; ind_edg++ )
    {   //add edges in each direction
        addDoubledge(cp->main_graph, boost::vertex(Eu[ind_edg]
                    , cp->main_graph), boost::vertex(Ev[ind_edg]
                    , cp->main_graph), edgeWeight[ind_edg],2 * ind_edg
                    , edge_attribute_map);
    }
}
//===========================================================================
//=====================      SET SPEED    ===================================
//===========================================================================
template<typename T>
void set_speed(CP::CutPursuit<T> * cp, const T speed, const float verbose)
{
    if (speed == 3)
    {
         if (verbose > 0)
        {
            std::cout << "PARAMETERIZATION = LUDICROUS SPEED" << std::endl;
        }
        cp->parameter.flow_steps  = 1;
        cp->parameter.kmeans_ite  = 3;
        cp->parameter.kmeans_resampling = 1;
        cp->parameter.max_ite_main = 5;
        cp->parameter.backward_step = false;
        cp->parameter.stopping_ratio = 0.001;
    }
    if (speed == 2)
    {
         if (verbose > 0)
        {
            std::cout << "PARAMETERIZATION = FAST" << std::endl;
        }
        cp->parameter.flow_steps  = 2;
        cp->parameter.kmeans_ite  = 5;
        cp->parameter.kmeans_resampling = 2;
        cp->parameter.max_ite_main = 5;
        cp->parameter.backward_step = true;
        cp->parameter.stopping_ratio = 0.001;
    }
    else if (speed == 0)
    {
         if (verbose > 0)
        {
            std::cout << "PARAMETERIZATION = SLOW" << std::endl;
        }
        cp->parameter.flow_steps  = 4;
        cp->parameter.kmeans_ite  = 8;
        cp->parameter.kmeans_resampling = 5;
        cp->parameter.max_ite_main = 20;
        cp->parameter.backward_step = true;
        cp->parameter.stopping_ratio = 0;
    }
    else if (speed == 1)
    {
        if (verbose > 0)
        {
            std::cout << "PARAMETERIZATION = STANDARD" << std::endl;
        }
        cp->parameter.flow_steps  = 3;
        cp->parameter.kmeans_ite  = 5;
        cp->parameter.kmeans_resampling = 2;
        cp->parameter.max_ite_main = 10;
        cp->parameter.backward_step = true;
        cp->parameter.stopping_ratio = 0.0001;
    }
}
}
