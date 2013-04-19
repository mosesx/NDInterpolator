/**
 * MoReTools.h
 *
 *  Created on: \date Mar 28, 2011
 *      Author: \author mrehberg
 */

#ifndef MORETOOLS_H_
#define MORETOOLS_H_

#include "utility.hpp"

namespace NDInterpolator {
  namespace ublas = boost::numeric::ublas;

  /**
   * Reads the SIM data from Jochen Siehrs model reduction tool. It always assumes that the slow
   * variables preceede the fast variables.
   * @param nx Number of reaction progress variables (slow states).
   * @param ny Number of fast variables (fast states).
   * @param file input file.
   * @param slowVarInd Indices of slow variables to use, from the range [0,ny-1]. If length=0 then full
   * range is assumed.
   * @return Vector of size 2, first element: matrix of grid points, second element: matrix of data points.
   */
  inline std::vector<ublas::matrix<double, ublas::column_major> > readSim(const unsigned nx, 
									  const unsigned ny, 
									  const std::string file,
									  std::vector<unsigned> slowVarInd = 
									  std::vector<unsigned>()) {
    std::vector<ublas::matrix<double, ublas::column_major> > ret(2);
    // read the whole file and afterwards split grid and data points
    ublas::matrix<double, ublas::column_major> temp = readMatrix<2> (file);
    // data points
    ret[0] = ublas::trans(ublas::subrange(temp, 0, temp.size1(), 0, nx));

    // index for columns, either all (if length=0) or use the argument
    if (slowVarInd.size()==0) {
      slowVarInd.resize(ny);
      for (unsigned i=0; i<ny;++i)
	slowVarInd[i]=i+nx;
    }
    else { // adjust the index by nx
      for (auto it=slowVarInd.begin();it != slowVarInd.end() ;++it)
	(*it)=(*it)+nx;
    }
    //ret[1] = ublas::trans(ublas::subrange(temp, 0, temp.size1(), nx, nx + ny));
    ret[1] = ublas::trans(cpColumns(temp,slowVarInd));
    
    return ret;
  }

  /**
   * Read the tangent space data from Jochen Siehrs Model Reduction Tool (MoRe). Assumption: slow
   * variables always precede the fast variables.
   * @param nx Number of reaction progress variables (slow states).
   * @param ny Number of fast variables (fast states).
   * @param file input file.
   * @return Vector of size 1 containing tangent/jacobian matrizes for each grid point (standard
   * format.
   */
  inline std::vector<ublas::matrix<double, ublas::column_major> > readMTV(const unsigned nx, 
									  const unsigned ny, 
									  const std::string file,
									  std::vector<unsigned> slowVarInd = 
									  std::vector<unsigned>()) {
    /*
     * Format in mtv_solution: y direction: points, x direction: dh1/dx1, dh2/dx1,...dhn/dx1,
     * dh1/dx2, dh2/dx2, ..., dhn/dx2, dh1/dxm, ...dhn/dxm
     * Now: 
     * 1. Extract all relevant columns, either all derivatives of fast variables with respect to slow variables
     *    or use index vectors, length of each block in one row is (nx + ny + 1), i.e. nx variables with respect to
     *    x_j, ny variables with respect to x_j and temperature with respect to x_j
     *    Use additional index vector slowVarMTVInd for indexing mtv columns.
     * 2. Reshape rows of result of 1. and put into function result accordingly. 
     */
    unsigned ny_used=0; // how many fast vars do we want in the output
    auto count=0;
    std::vector<unsigned> slowVarMTVInd;

    if (slowVarInd.size()==0) {
      slowVarMTVInd.resize(ny*nx);
      for (unsigned j=0; j<nx; ++j) {
	for (unsigned i=0; i<ny;++i){
	  slowVarMTVInd[count]=j*(nx+ny+1)+nx+i;
	  count++;
	}
      }
      ny_used=ny;
    }
    else { // adjust the index by nx
      ny_used=static_cast<unsigned>(slowVarInd.size());
      // new size, because for each slow variable used there are nx derivatives
      slowVarMTVInd.resize(nx*ny_used);
      for (unsigned j=0; j<nx; ++j) {
	for (unsigned i=0; i<ny_used;++i){
	  slowVarMTVInd[count]=j*(nx+ny+1) + nx + slowVarInd[i];
	  count++;
	}
      }
    }

    ublas::matrix<double, ublas::column_major> fastDeriv;
    { // extra block to free memory of temp.
      ublas::matrix<double, ublas::column_major> temp = readMatrix<2> (file);    
      fastDeriv=cpColumns(temp,slowVarMTVInd);
    }
    
    // start of phase 2.
    // allocate result, with right size jacobians
    std::vector<ublas::matrix<double, ublas::column_major> > ret(fastDeriv.size1(),
								 ublas::matrix<double, ublas::column_major> (ny_used, nx));
    
    for (unsigned i = 0; i < fastDeriv.size1(); ++i) {
      for (unsigned j = 0; j < nx; ++j) {
	noalias(ublas::column(ret[i], j)) = ublas::subrange(ublas::row(fastDeriv, i), j * ny_used,
							    (j+1) * ny_used);
      }
    }
    
    return ret;
  }
  
  /**
   * Creates a shape optimized interpolator object from model reduction data.
   * @param nx Number of reaction progress variables (slow states)
   * @param ny Number of fast states.
   * @param scale Initial scale for the interpolator.
   * @param path Path in which to look for the data files.
   * @param simFile File that holds the manifold data.
   * @param mtvFile File that holds the tangent space vectors
   * @tparam interp Which interpolator type is to be used.
   * @return Interpolator object.
   */
  template<class interp>
  inline interp createInterp(const unsigned nx, const unsigned ny,
			     const double scale, std::string path,
			     std::string simFile = "sim_solution.out",
			     std::string mtvFile = "mtv_solution.out") {
    
    // retrieve grid and data.
    std::vector<ublas::matrix<double, ublas::column_major> > simData = readSim(nx, ny, path + simFile);
  
    interp ret(simData[0], simData[1], scale, 100, .1);
    ret.optimizeScale();
  
  
    return ret;
  }
									
  /**
   * Creates a shape optimized hermite interpolator object from model reduction data.
   * @param nx Number of reaction progress variables (slow states)
   * @param ny Number of fast states.
   * @param scale Initial scale for the interpolator.
   * @param path Path in which to look for the data files.
   * @param simFile File that holds the manifold data.
   * @param mtvFile File that holds the tangent space vectors
   * @param ppp Points per patch used for partition of unity interpolators.
   * @param overlap Overlap used for partition of unity interpolators.
   * @tparam interp Which interpolator type is to be used.
   * @return Interpolator object.
   */
  template<class interp>
  inline interp createInterpH(const unsigned nx, const unsigned ny,
			      const double scale, const std::string path,
			      const std::string simFile = "sim_solution.out",
			      const std::string mtvFile = "mtv_solution.out",
			      const unsigned ppp = 50, const double overlap = .33,
			      const std::vector<unsigned> slowVarInd=std::vector<unsigned>(),
			      const bool useBump = false) {
    
    // retrieve grid and data.
    std::vector<ublas::matrix<double, ublas::column_major> > simData = readSim(nx, ny, path + simFile,slowVarInd);
    // retrieve mtvs
    std::vector<ublas::matrix<double, ublas::column_major> > mtvData = readMTV(nx, ny, path + mtvFile,slowVarInd);
    
    // construct interpolator and return, due to incompatible constructor interfaces
    // PU and normal interpolators have to be treated differently.
    interp ret(simData[0], simData[1], mtvData, scale, ppp, overlap,useBump);
    //interp ret(simData[0], simData[1], mtvData, scale);
    ret.optimizeScale();
    return ret;
  } 

  /**
   * Creates a shape optimized hermite interpolator object from model reduction data.
   * @param nx Number of reaction progress variables (slow states)
   * @param ny Number of fast states.
   * @param scale Initial scale for the interpolator.
   * @param path Path in which to look for the data files.
   * @param simFile File that holds the manifold data.
   * @param mtvFile File that holds the tangent space vectors
   * @param ppp Points per patch used for partition of unity interpolators.
   * @param overlap Overlap used for partition of unity interpolators.
   * @tparam interp Which interpolator type is to be used.
   * @return Interpolator object.
   */
  template<class interp>
  inline interp createInterpH(const unsigned nx, const unsigned ny,
			      const double scale, const double left, const double right, 
			      const std::string path,
			      const std::string simFile = "sim_solution.out",
			      const std::string mtvFile = "mtv_solution.out",
			      const unsigned ppp = 50, const double overlap = .33,
			      const std::vector<unsigned> slowVarInd=std::vector<unsigned>(),
			      const bool useBump = false) {
    
    // retrieve grid and data.
    std::vector<ublas::matrix<double, ublas::column_major> > simData = readSim(nx, ny, path + simFile,slowVarInd);
    // retrieve mtvs
    std::vector<ublas::matrix<double, ublas::column_major> > mtvData = readMTV(nx, ny, path + mtvFile,slowVarInd);
    
    // construct interpolator and return, due to incompatible constructor interfaces
    // PU and normal interpolators have to be treated differently.
    interp ret(simData[0], simData[1], mtvData, scale, ppp, overlap,useBump);
    //interp ret(simData[0], simData[1], mtvData, scale);
    ret.optimizeScale(left, right);
    return ret;
  } 
}
#endif 
  /* MORETOOLS_H_ */
