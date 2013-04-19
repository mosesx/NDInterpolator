/**
 * MultiRBFInterpolatorH.h
 *
 *  Created on: \date Nov 26, 2010
 *      Author: \author mrehberg
 */

#ifndef MULTIRBFINTERPOLATORH_H_
#define MULTIRBFINTERPOLATORH_H_

// std includes
#include <vector>
#include <cassert>
#include <typeinfo>
#include <math.h>

// ublas includes
#include "boost/numeric/ublas/symmetric.hpp"
#include "boost/numeric/ublas/operation.hpp"

// ublas bindings includes
#include "boost/numeric/bindings/ublas/matrix.hpp"
#include "boost/numeric/bindings/ublas/vector.hpp"
#include "boost/numeric/bindings/lapack/driver/gesv.hpp"

// NDInterpolator includes
#include "../MultiNDInterpolator.hpp"
#include "RBFs.hpp"

#include "../tools/utility.hpp"
#include "goodc/goodc.hpp"

namespace NDInterpolator {
  namespace lapack = boost::numeric::bindings::lapack;
  namespace blas = boost::numeric::bindings::blas;
  namespace ublas = boost::numeric::ublas;
  
  /**
   * \brief Multi dimensional output Hermite Interpolator.
   *
   * Implementation of the NDInterpolator::MultiNDInterpolator interface for Hermite interpolation,
   * that is the object implements functions \f$ f: \mathrm{R}^{inDim} \rightarrow \mathrm{R}^{outDim} \f$.
   * A Lagrange version can be found in NDInterpolator::MultiRBFInterpolator. The class is the
   * multidimensional extension of NDInterpolator::RBFInterpolatorH.
   *
   * \tparam RTYPE radial basis function type, e.g. NDInterpolator::GaussianRBF<double>.
   */
  template<class RTYPE>
  class MultiRBFInterpolatorH: public MultiNDInterpolator {
  public:
    
    /* Instead of using various template arguments, we just take the value_type from 
     * the corresponding radial basis function. Make sure that the rbf class possesses 
     * something like 'typedef T value_type'
     */
    typedef typename RTYPE::value_type value_type;

    typedef ublas::matrix<value_type, ublas::column_major> boostmatrix_type;
    typedef std::vector<ublas::matrix<value_type, ublas::column_major> > diffdata_type;

    MultiRBFInterpolatorH() {
      inDim = 0;
      outDim = 0;
      numOfPoints = 0;
      this->scale = value_type();
    }

    /**
     * Create a MultiRBFInterpolatorH object. The numOfPoints centers are given in the matrix grid.
     * Function values are in data and the std::vector diffData contains numOfPoints jacobians.
     * @param grid Matrix of size inDim times numOfPoints
     * @param data Matrix of size outDim times numOfPoints
     * @param diffData std::vector of length numOfPoints containing jacobians of dimension
     * outDim times inDim, i.e.
     * \f[ (\frac{\partial f_i}{\partial x_j})_{i,j}. \f]
     * @param scale Positive scaling parameter for the RBF.
     */
    MultiRBFInterpolatorH(
			  const ublas::matrix<value_type, ublas::column_major>& grid,
			  const ublas::matrix<value_type, ublas::column_major>& data,
			  const std::vector<ublas::matrix<value_type,
							  ublas::column_major> >& diffData,
			  const value_type& scale);

    /**
     * Assign values to every member and reinitialize the interpolator object.
     * @param grid Set of center points.
     * @param data Matrix of data points, each column is associated to one column in grid.
     * @param diffData Vector of jacobians. Each entry in the std::vector is associated
     * to one center point and is a matrix of size outDim x outDim representing the jacobian in
     * that point.
     * @param scale Scaling factor for the underlying RBF function.
     */
    void
    assign(
	   const ublas::matrix<value_type, ublas::column_major>& grid,
	   const ublas::matrix<value_type, ublas::column_major>& data,
	   const std::vector<ublas::matrix<value_type,
					   ublas::column_major> >& diffData,
	   const value_type& scale);

    /**
     * Computes and returns the interpolation matrix for a certain shape/scale parameter \f$c\f$. Attention:
     * this does not mean the interpolator uses that c for interpolation after a call to this method.
     * Use NDInterplator::MultiRBFInterpolatorH::assign for that purpose.
     * @param c Scale/shape parameter that is used in calculating the interpolation matrix
     * @return
     */
    ublas::matrix<value_type, ublas::column_major>
    getInterpMatrix(const value_type c) const;

    /**
     * Returns the data matrix used to create the interpolation object.
     * @return
     */
    ublas::matrix<value_type, ublas::column_major> getData() const;

    /**
     * Set the scaling parameter for the RBF to a new value and re-init. After a call,
     * the interpolation object is ready to be used again.
     * @param scale New scale
     */
    void setNewScale(const value_type scale) {
      this->rbf = RTYPE(scale);
      this->scale = scale;
      init();
    }

    /**
     * Try to calculate the optimal scaling parameter for the interpolator.
     * The calculated scale is used for all computations after a call to this method.
     */
    void optimizeScale() {
      MultiRBFInterpolatorH<RTYPE>& thisRef = *this;
      BrentsMethod<MultiRBFInterpolatorH<RTYPE> > bm(thisRef);
      // compute new shape parameter and adjust the interpolation object
      this->setNewScale(bm.minNeu(.001, 1. * this->scale, 32));
    }

    /**
     * Try to calculate the optimal scaling parameter for the interpolator.
     * The calculated scale is used for all computations after a call to this method.
     */
    void optimizeScale(const double left, const double right) {
      MultiRBFInterpolatorH<RTYPE>& thisRef = *this;
      BrentsMethod<MultiRBFInterpolatorH<RTYPE> > bm(thisRef);
      // compute new shape parameter and adjust the interpolation object
      this->setNewScale(bm.minNeu(left, right, 32));
    }

    /**
     * What scale parameter is used for the calculations.
     * @return
     */
    value_type getScale() {
      return this->scale;
    }
    
    ublas::vector<value_type>
    eval(const ublas::vector<value_type>& inPoint);
    
    ublas::vector<value_type>
    evalDiff(const ublas::vector<value_type>& inPoint,
	     const unsigned int alpha);
    
    ublas::vector<value_type>
    evalDiff2(const ublas::vector<value_type>& inPoint,
	      const unsigned alpha) const;

    ublas::vector<value_type> evalDiffMixed(
					    const ublas::vector<value_type>& inPoint, const unsigned alpha1,
					    const unsigned alpha2) const;

    ublas::vector<value_type>
    evalGrad(const ublas::vector<value_type>& inPoint,
	     const unsigned alpha) const;

    void evalFuncAndJac(const ublas::vector<value_type>& inPoint,
			ublas::vector<value_type>& outPoint,
			ublas::matrix<value_type, ublas::column_major>& jac);

    ublas::matrix<value_type, ublas::column_major> evalJac(
							   const ublas::vector<value_type>& inPoint);


    std::vector<ublas::matrix<value_type, ublas::column_major> > evalHess(
									  const ublas::vector<value_type>& inPoint) const;

    /**
     * In case the default constructor is used to instantiate the object, this function
     * can be used to check that the respective dimensions aren't zero while evaluation.
     */
    inline void checkDims() const {
#ifndef NDEBUG
      if (grid.size1() == 0 || grid.size2() == 0) {
	std::cerr << "***** ERROR throw by '" << __func__ << "' in file "
		  << __FILE__
		  << ": The private member 'grid' hasn't been filled. "
		  << std::endl;
	abort(); //stop computations immediately
      }
#endif
    }

    /**
     * Debug function, outputs sizes of all member variables.
     */
    inline void info() {
      std::cout << "rbf field = " << rbf.getScale() << std::endl;
      std::cout << "scale = " << scale << std::endl << std::endl;
      std::cout << "grid = " << grid << std::endl << std::endl;
    }

    virtual ~MultiRBFInterpolatorH();

  private:

    RTYPE rbf;
    value_type scale;

    ublas::vector<value_type> outPointGlob;
    ublas::matrix<value_type, ublas::column_major> outPointGlobJac;
    ublas::vector<value_type> gridColumn;
    value_type norm;

    ublas::matrix<value_type, ublas::column_major> grid;
    ublas::matrix<value_type, ublas::column_major> data;
    std::vector<ublas::matrix<value_type, ublas::column_major> > diffData;
    ublas::matrix<value_type, ublas::column_major> lambda;

    // init the interpolation object
    void init();

#ifndef _WITHOUT_SERIALIZATION_
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version) {

      ar & boost::serialization::base_object<MultiNDInterpolator>(*this);

      ar & rbf;
      ar & scale;

      ar & outPointGlob;
      ar & outPointGlobJac;
      ar & gridColumn;
      ar & norm;

      ar & grid;
      ar & data;
      ar & diffData;
      ar & lambda;
    }
#endif

  }; //end of class


  //#################### MEMBER INITIALISATION ################################# 

  template<class RTYPE> //constructor
  inline MultiRBFInterpolatorH<RTYPE>::MultiRBFInterpolatorH(
							     const ublas::matrix<typename RTYPE::value_type, ublas::column_major>& grid,
							     const ublas::matrix<typename RTYPE::value_type, ublas::column_major>& data,
							     const std::vector<ublas::matrix<typename RTYPE::value_type,
							     ublas::column_major> >& diffData,
							     const typename RTYPE::value_type& scale) :
    rbf(RTYPE(scale)), scale(scale), grid(grid),
    data(ublas::trans(data)), diffData(diffData),
    lambda(ublas::trans(data)) {

    // initialize base members, attention: transpose in above initialization has no effect on local
    // variables!!
    inDim = static_cast<unsigned>(grid.size1());
    outDim = static_cast<unsigned>(data.size1());
    numOfPoints = static_cast<unsigned>(grid.size2());
    
    outPointGlob.resize(numOfPoints * (inDim + 1));
    outPointGlobJac.resize(inDim, numOfPoints * (inDim + 1));
    gridColumn.resize(inDim);
    // input check, assume that grid has the right size
    if (data.size2() != numOfPoints)
      throw std::length_error(
			      "MultiRBFInterpolatorH::Constructor: Number of interpolation points (grid.size2()) and number of data points (data.size2()) do not match.");

    if (diffData.size() != numOfPoints)
      throw std::length_error(
			      "MultiRBFInterpolatorH::Constructor: Number of jacobians (diffData.size()) does not match number of centers (numOfPoints).");

    for (unsigned i = 0; i < numOfPoints; i++) {
      if (diffData[i].size1() != outDim)
	throw std::length_error(
				"MultiRBFInterpolatorH::Constructor: Number of rows of jacobian	<< i << does not match outDim.");
      if (diffData[i].size2() != inDim)
	throw std::length_error(
				"MultiRBFInterpolatorH::Constructor: Number of columns of jacobian << i << does not match inDim");
    }

    // initialize the interpolation object
    if ((grid.size1() != 0) && (grid.size2() != 0) && (data.size1() != 0)
	&& (data.size2() != 0) && (diffData.size() != 0))
      init();
  }

  template<class RTYPE>
  inline ublas::matrix<typename RTYPE::value_type, ublas::column_major> MultiRBFInterpolatorH<
    RTYPE>::getInterpMatrix(const typename RTYPE::value_type c) const {
    // local RBF with input scale.
    RTYPE localRBF(c);
    // allocate space for the interpolation matrix
    ublas::matrix<typename RTYPE::value_type, ublas::column_major> iMatrix(
									   numOfPoints, numOfPoints);

    // matrix will be symmetric --> use symmetric adaptor to compute just half of it.
    ublas::symmetric_adaptor < ublas::matrix<typename RTYPE::value_type,
					     ublas::column_major>, ublas::upper > iMatrixAdapt(iMatrix);
    for (unsigned int i = 0; i < numOfPoints; i++) {
      for (unsigned int j = i; j < numOfPoints; j++) {
	iMatrixAdapt(i, j) = localRBF.eval(
					   ublas::norm_2(
							 ublas::column(grid, i) - ublas::column(grid, j)),
					   ublas::column(grid, i), ublas::column(grid, j));
      }
    }
    return iMatrixAdapt;
  }

  template<class RTYPE>
  inline ublas::matrix<typename RTYPE::value_type, ublas::column_major> MultiRBFInterpolatorH<
    RTYPE>::getData() const {
    return data;
  }

  template<class RTYPE>
  inline ublas::vector<typename RTYPE::value_type> 
  MultiRBFInterpolatorH<RTYPE>::eval(const ublas::vector<typename RTYPE::value_type>& inPoint) {
    // check if grid is empty
    checkDims();

    ublas::vector<double> ret(outDim);
    for (unsigned l = 0; l< outDim; ++l)
      ret(l)=0;
    
    for (unsigned int i = 0; i < numOfPoints; i++) {
      for (unsigned k = 0; k < inDim; ++k) {
       	gridColumn(k) = grid(k, i);
      }
      norm = ublas::norm_2(inPoint - gridColumn);

      // norm=0;
      // for (unsigned k = 0; k < inDim; ++k) {
      // 	norm+= (inPoint(k)-grid(k,i))*(inPoint(k)-grid(k,i));
      // }
      // norm = std::sqrt(norm);
      
      outPointGlob(i * (inDim + 1)) = rbf.eval(norm);
      for (unsigned int j = 0; j < inDim; j++){
	outPointGlob(i * (inDim + 1) + j + 1) = rbf.evalDiff1(norm, inPoint, gridColumn, j + 1);
      }
    }
    // Lambda^T*evalRBF
    return ublas::prod(outPointGlob, lambda);
    //    return ret;
  }

  template<class RTYPE>
  inline ublas::vector<typename RTYPE::value_type> MultiRBFInterpolatorH<RTYPE>::evalDiff(
											  const ublas::vector<typename RTYPE::value_type>& inPoint,
											  const unsigned alpha) {
    // check if grid is empty
    checkDims();

    for (unsigned i = 0; i < numOfPoints; i++) {
      for (unsigned k = 0; k < inDim; ++k) {
	gridColumn(k) = grid(k, i);
      }
      norm = ublas::norm_2(inPoint - gridColumn);

      outPointGlob(i * (inDim + 1)) = rbf.evalDiff1(norm, inPoint,
						    gridColumn, alpha);
      for (unsigned int j = 0; j < inDim; j++) {
	if (j + 1 != alpha) {
	  outPointGlob(i * (inDim + 1) + j + 1) = rbf.evalDiffMixed(norm,
								    inPoint, gridColumn, alpha, j + 1);
	} else {
	  outPointGlob(i * (inDim + 1) + j + 1) = rbf.evalDiff2(norm,
								inPoint, gridColumn, alpha);
	}
      }
    }

    return ublas::prod(outPointGlob, lambda);
  }

  template<class RTYPE>
  inline ublas::vector<typename RTYPE::value_type> MultiRBFInterpolatorH<RTYPE>::evalDiff2(
											   const ublas::vector<typename RTYPE::value_type>& inPoint,
											   const unsigned alpha) const {
    // check if grid is empty
    checkDims();

    ublas::vector<typename RTYPE::value_type>
      evalRBF(numOfPoints * (inDim + 1));

    for (unsigned i = 0; i < numOfPoints; i++) {
      evalRBF(i * (inDim + 1)) = rbf.evalDiff2(
					       ublas::norm_2(inPoint - ublas::column(grid, i)), inPoint,
					       ublas::column(grid, i), alpha);
      for (unsigned j = 0; j < inDim; j++) {
	if (j + 1 != alpha)
	  evalRBF(i * (inDim + 1) + j + 1) = rbf.evalDiff3DMixed(
								 ublas::norm_2(inPoint - ublas::column(grid, i)),
								 inPoint, ublas::column(grid, i), alpha, j + 1);
	else
	  evalRBF(i * (inDim + 1) + j + 1) = rbf.evalDiff3(
							   ublas::norm_2(inPoint - ublas::column(grid, i)),
							   inPoint, ublas::column(grid, i), alpha);
      }
    }

    return ublas::prod(evalRBF, lambda);
  }

  template<class RTYPE>
  inline ublas::vector<typename RTYPE::value_type> MultiRBFInterpolatorH<RTYPE>::evalDiffMixed(
											       const ublas::vector<typename RTYPE::value_type>& inPoint,
											       const unsigned alpha1, const unsigned alpha2) const {

    // check if grid is empty
    checkDims();

    ublas::vector<typename RTYPE::value_type>
      evalRBF(numOfPoints * (inDim + 1));

    for (unsigned i = 0; i < numOfPoints; i++) {
      evalRBF(i * (inDim + 1)) = rbf.evalDiffMixed(
						   ublas::norm_2(inPoint - ublas::column(grid, i)), inPoint,
						   ublas::column(grid, i), alpha1, alpha2);
      for (unsigned j = 0; j < inDim; j++) {
	if (j + 1 != alpha1 && j + 1 != alpha2)
	  evalRBF(i * (inDim + 1) + j + 1) = rbf.evalDiff3Mixed(
								ublas::norm_2(inPoint - ublas::column(grid, i)),
								inPoint, ublas::column(grid, i), alpha1, alpha2, j + 1);
	else if (j + 1 != alpha1) // j+1 == alpha2
	  evalRBF(i * (inDim + 1) + j + 1) = rbf.evalDiff3DMixed(
								 ublas::norm_2(inPoint - ublas::column(grid, i)),
								 inPoint, ublas::column(grid, i), alpha2, alpha1);
	else
	  // j+1 == alpha 1
	  evalRBF(i * (inDim + 1) + j + 1) = rbf.evalDiff3DMixed(
								 ublas::norm_2(inPoint - ublas::column(grid, i)),
								 inPoint, ublas::column(grid, i), alpha1, alpha2);
      }
    }

    return ublas::prod(evalRBF, lambda);
  }

  template<class RTYPE>
  inline ublas::vector<typename RTYPE::value_type> MultiRBFInterpolatorH<RTYPE>::evalGrad(
											  const ublas::vector<typename RTYPE::value_type>& inPoint,
											  const unsigned alpha) const {
    // check if grid is empty
    checkDims();
    assert(alpha > 0 && alpha < outDim);
    return ublas::row(evalJac(inPoint), alpha - 1);
  }

  //@todo optimize by eliminating norm computation
  template<class RTYPE>
  inline void MultiRBFInterpolatorH<RTYPE>::evalFuncAndJac(
							   const ublas::vector<typename RTYPE::value_type>& inPoint,
							   ublas::vector<typename RTYPE::value_type>& outPoint,
							   ublas::matrix<typename RTYPE::value_type, ublas::column_major>& jac) {

    // check if grid is empty
    checkDims();

    for (unsigned i = 0; i < numOfPoints; i++) {
      for (unsigned k = 0; k < inDim; ++k) {
	gridColumn(k) = grid(k, i);
      }
      norm = ublas::norm_2(inPoint - gridColumn);

      // evaluate for outpoint
      outPointGlob(i * (inDim + 1)) = rbf.eval(norm);

      for (unsigned k = 0; k < inDim; k++) {
	// evaluation for jac
	outPointGlobJac(k, i * (inDim + 1)) = rbf.evalDiff1(norm, inPoint,
							    gridColumn, k + 1);
	// reuse for outpoint
	outPointGlob(i * (inDim + 1) + k + 1) = outPointGlobJac(k,
								i * (inDim + 1));

	for (unsigned int j = 0; j < inDim; j++) {
	  if (j + 1 != k + 1) {
	    outPointGlobJac(k, i * (inDim + 1) + j + 1)
	      = rbf.evalDiffMixed(norm, inPoint, gridColumn,
				  k + 1, j + 1);
	  } else {
	    outPointGlobJac(k, i * (inDim + 1) + j + 1)
	      = rbf.evalDiff2(norm, inPoint, gridColumn, k + 1);
	  }
	}
      }
    }
    // Lambda^T*evalRBF
    outPoint = ublas::prod(outPointGlob, lambda);
    ///@todo do not like this transpose in return but works and still faster than old jacobian
    //return ublas::trans(ublas::prod(outPointGlobJac, lambda));
    jac = ublas::trans(ublas::prod(outPointGlobJac, lambda));
  }

  //@todo optimize by eliminating norm computation
  template<class RTYPE>
  inline ublas::matrix<typename RTYPE::value_type, ublas::column_major> MultiRBFInterpolatorH<
    RTYPE>::evalJac(
		    const ublas::vector<typename RTYPE::value_type>& inPoint) {

    // check if grid is empty
    checkDims();

    for (unsigned i = 0; i < numOfPoints; i++) {
      for (unsigned k = 0; k < inDim; ++k) {
	gridColumn(k) = grid(k, i);
      }
      norm = ublas::norm_2(inPoint - gridColumn);

      for (unsigned k = 0; k < inDim; k++) {
	outPointGlobJac(k, i * (inDim + 1)) = rbf.evalDiff1(norm, inPoint,
							    gridColumn, k + 1);
	for (unsigned int j = 0; j < inDim; j++) {
	  if (j + 1 != k + 1) {
	    outPointGlobJac(k, i * (inDim + 1) + j + 1)
	      = rbf.evalDiffMixed(norm, inPoint, gridColumn,
				  k + 1, j + 1);
	  } else {
	    outPointGlobJac(k, i * (inDim + 1) + j + 1)
	      = rbf.evalDiff2(norm, inPoint, gridColumn, k + 1);
	  }
	}
      }
    }
    ///@todo do not like this transpose in return but works and still faster than old jacobian
    return ublas::trans(ublas::prod(outPointGlobJac, lambda));

  }


  template<class RTYPE>
  inline std::vector<ublas::matrix<typename RTYPE::value_type,
				   ublas::column_major> > MultiRBFInterpolatorH<RTYPE>::evalHess(
												 const ublas::vector<typename RTYPE::value_type>& inPoint) const {
    // check if grid is empty
    checkDims();
    std::vector<ublas::matrix<typename RTYPE::value_type, ublas::column_major> >
      ret(outDim);
    for (unsigned i = 0; i < outDim; i++) {
      ret[i].resize(inDim, inDim);
    }
    ublas::matrix<typename RTYPE::value_type, ublas::column_major> hess(inDim,
									inDim);
    ublas::symmetric_adaptor < ublas::matrix<typename RTYPE::value_type,
					     ublas::column_major>, ublas::upper > hessAdapt(hess);
    ublas::vector<typename RTYPE::value_type> retVec(outDim);

    for (unsigned i = 0; i < inDim; i++) {
      for (unsigned j = i + 1; j < inDim; j++) {
	retVec = this->evalDiffMixed(inPoint, i + 1, j + 1);
	for (unsigned k = 0; k < outDim; k++)
	  ret[k](i, j) = retVec(k);
      }
      retVec = this->evalDiff2(inPoint, i + 1);
      for (unsigned k = 0; k < outDim; k++)
	ret[k](i, i) = retVec(k);
    }

    return ret;
  }

  /*
   * init the interpolation object. Unlike in the nonhermitian case the interpolation matrix
   * is not symmetric.
   */
  template<class RTYPE>
  inline void MultiRBFInterpolatorH<RTYPE>::init() {
    // allocate space for the interpolation matrix, non-hermite interpolation --> one data point for every
    // interpolation point
    ublas::matrix<typename RTYPE::value_type, ublas::column_major> iMatrix(
									   numOfPoints * (inDim + 1), numOfPoints * (inDim + 1));
    // matrix for holding data to be able to use lapack
    lambda.resize(numOfPoints * (inDim + 1), outDim);

    // overall matrix consists of symmetric submatrices of size inDim+1
    // S= fkt-Value  gradient'
    //    gradient   hessian
    ublas::matrix<typename RTYPE::value_type, ublas::column_major> subTemp(
									   inDim + 1, inDim + 1);
    ublas::symmetric_adaptor < ublas::matrix<typename RTYPE::value_type,
					     ublas::column_major>, ublas::upper > subTempAdapt(subTemp);

    // outer for loops iterate over points, inner for-loops for gradient and hessian
    // computation
    for (unsigned int i = 0; i < numOfPoints; i++) {
      //std::cout << i << std::endl;
      for (unsigned int j = 0; j < numOfPoints; j++) {
	for (unsigned int k = 1; k < inDim + 1; k++) {
	  for (unsigned int l = k + 1; l < inDim + 1; l++) {
	    // uper triangular part off hessian (without diagonal)
	    subTempAdapt(k, l) = rbf.evalDiffMixed(
						   ublas::norm_2(
								 ublas::column(grid, i) - ublas::column(
													grid, j)), ublas::column(grid, i),
						   ublas::column(grid, j), k, l);
	  } //l
	  // diagonal of hessian
	  subTempAdapt(k, k)
	    = rbf.evalDiff2(
			    ublas::norm_2(
					  ublas::column(grid, i) - ublas::column(
										 grid, j)),
			    ublas::column(grid, i), ublas::column(grid, j),
			    k);
	  // gradient
	  subTempAdapt(0, k)
	    = rbf.evalDiff1(
			    ublas::norm_2(
					  ublas::column(grid, i) - ublas::column(
										 grid, j)),
			    ublas::column(grid, i), ublas::column(grid, j),
			    k);
	}// k
	// function value
	subTempAdapt(0, 0) = rbf.eval(
				      ublas::norm_2(
						    ublas::column(grid, i) - ublas::column(grid, j)),
				      ublas::column(grid, i), ublas::column(grid, j));

	// put submatrix into large iMatrix, every submatrix has size inDim+1 --> advance that far in each step
	// inDim + 1 is added to the end-indices because subrange expects stop to be 1 further than
	// the wanted number, i.e. to end and include the 2nd column one would have to put 3 into subrange
	noalias(
		ublas::subrange(iMatrix, i * (inDim + 1),
				i * (inDim + 1) + inDim + 1, j * (inDim + 1),
				j * (inDim + 1) + inDim + 1)) = subTempAdapt;
      } // j

      // build rhs
      for (unsigned m = 0; m < outDim; m++) {
	lambda(i * (inDim + 1), m) = data(i, m);
	// can't use columns since subrange is a matrix (again remember stop index + 1)
	noalias(
		ublas::subrange(lambda, i * (inDim + 1) + 1,
				i * (inDim + 1) + inDim + 1, m, m + 1))
	  = ublas::trans(
			 ublas::subrange(diffData[i], m, m + 1, 0, inDim));

      } //m
    }// i

    // solve and write out the solution
    ublas::vector<int> ipiv(numOfPoints * (inDim + 1));
    lapack::gesv(iMatrix, ipiv, lambda); //matrix will be overwritten
  }

  /**
   * \brief an alternative assignment member
   */
  template<class RTYPE>
  inline void MultiRBFInterpolatorH<RTYPE>::assign(
						   const ublas::matrix<typename RTYPE::value_type, ublas::column_major>& g,
						   const ublas::matrix<typename RTYPE::value_type, ublas::column_major>& d,
						   const std::vector<ublas::matrix<typename RTYPE::value_type,
										   ublas::column_major> >& dD,
						   const typename RTYPE::value_type& s) {

    inDim = g.size1();
    outDim = d.size1();
    numOfPoints = g.size2();

    // input check, assume that grid has the right size
    if (d.size2() != numOfPoints)
      throw std::length_error(
			      "MultiRBFInterpolatorH::Constructor: Number of interpolation points (grid.size2()) and number of data points (data.size()) do not match.");

    if (dD.size() != numOfPoints)
      throw std::length_error(
			      "MultiRBFInterpolatorH::Constructor: Number of jacobians (diffData.size()) does not match number of centers (numOfPoints).");

    for (unsigned i = 0; i < numOfPoints; i++) {
      if (dD[i].size1() != outDim)
	throw std::length_error(
				"MultiRBFInterpolatorH::Constructor: Number of rows of jacobian	<< i << does not match outDim.");
      if (dD[i].size2() != inDim)
	throw std::length_error(
				"MultiRBFInterpolatorH::Constructor: Number of columns of jacobian << i << does not match inDim");
    }

    this->rbf = RTYPE(s);
    this->scale = s;
    this -> grid = g;
    this->data = ublas::trans(d);
    this->diffData = dD;
    this->lambda = ublas::trans(d);

    // init the shit out of it.
    init();
  }

  template<class RTYPE>
  inline MultiRBFInterpolatorH<RTYPE>::~MultiRBFInterpolatorH() {
  }

} //end of namespace 

#endif /* MULTIRBFINTERPOLATORH_H_ */
