/**
 * MultiDimRBFInterpolator.h
 *
 *  Created on: \date Nov 25, 2010
 *      Author: \author mrehberg
 */

#ifndef MULTIDIMRBFINTERPOLATOR_H_
#define MULTIDIMRBFINTERPOLATOR_H_

#include "../MultiNDInterpolator.hpp"

// std includes
#include <vector>
#include <cassert>

//ublas includes
#include "boost/numeric/ublas/symmetric.hpp"

// ublas lapack bindings includes
#include "boost/numeric/bindings/lapack/driver/posv.hpp"
#include "boost/numeric/bindings/ublas/matrix.hpp"
#include "boost/numeric/bindings/ublas/symmetric.hpp"

// NDInterpolator includes
#include "RBFs.hpp"

#include "../tools/utility.hpp"
#include "goodc/goodc.hpp"

namespace NDInterpolator {
  namespace lapack = boost::numeric::bindings::lapack;
  namespace ublas = boost::numeric::ublas;

  /**
   * \brief Multi dimensional output Lagrange interpolator.
   *
   * The class is a multidimensional extension of NDInterpolator::RBFInterpolator in the sense that
   * it supports multiple output dimensions, i.e. represents a function \f$ f: \mathrm{R}^n \rightarrow
   * \mathrm{R}^m \f$.
   * A version for Hermite interpolation can be found in NDInterpolator::MultiRBFInterpolatorH.
   * \tparam RTYPE Type of radial basis function to be used, e.g. NDInterpolator::GaussianRBF<double>
   */
  template<class RTYPE>
  class MultiRBFInterpolator: virtual public MultiNDInterpolator {
  public:
    typedef typename RTYPE::value_type value_type;

    MultiRBFInterpolator();

    /**
     * Creates a MultiRBFInterpolator object with center points in grid and the function values in
     * data. The parameter scale specifies the scaling of the underlying RBF. The object interpolates
     * functions \f$ f: \mathrm{R}^{inDim} \rightarrow \mathrm{R}^{outDim} \f$ based on numOfPoints
     * center points.
     * @param grid Matrix with dimension: inDim times numOfPoints.
     * @param data Matrix with dimension: outDim times numOfPoints.
     * @param scale Real number greater than 0.
     * @return
     */
    MultiRBFInterpolator(const ublas::matrix<value_type, ublas::column_major>& grid,
			 const ublas::matrix<value_type, ublas::column_major>& data,
			 const value_type scale);

    /**
     * Assign new values for all members and reinitialize the interpolation object.
     * @param grid New grid.
     * @param data New interpolation data.
     * @param scale New scale for the underlying RBF.
     */
    void assign(const ublas::matrix<value_type, ublas::column_major>& grid,
		const ublas::matrix<value_type, ublas::column_major>& data,
		const value_type scale) {
      // initialize base members
      inDim = grid.size1();
      outDim = data.size1();
      numOfPoints = grid.size2();

      // initialize members
      this->rbf = RTYPE(scale);
      this->scale = scale;
      this->grid = grid;
      this->data = ublas::trans(data);
      this->lambda = ublas::trans(data);

      // input check, assume that grid has the right size
      if (data.size2() != numOfPoints)
	throw std::length_error(
				"MultiRBFInterpolator::assign: Number of interpolation points (grid.size2()) and number of data points (data.size()) do not match.");

      // initialize the interpolation object
      init();
    }

    /**
     * Computes and returns the interpolation matrix for a certain shape/scale parameter c. Attention:
     * this does not mean the interpolator uses that c for interpolation after a call to this method.
     * Use NDInterplator::MultiRBFInterpolatorH::assign for that purpose.
     * @param c Scale/shape paramter that is used in calculating the interpolation matrix
     * @return
     */
    ublas::matrix<value_type, ublas::column_major> getInterpMatrix(const value_type c) const {
      // local RBF with input scale. 
      RTYPE localRBF(c);
      // allocate space for the interpolation matrix
      ublas::matrix<typename RTYPE::value_type, ublas::column_major> iMatrix(numOfPoints, numOfPoints);
      
      // matrix will be symmetric --> use symmetric adaptor to compute just half of it.
      ublas::symmetric_adaptor < ublas::matrix<typename RTYPE::value_type,
					       ublas::column_major>, ublas::upper > iMatrixAdapt(iMatrix);
      for (unsigned int i = 0; i < numOfPoints; i++) {
	for (unsigned int j = i; j < numOfPoints; j++) {
	  iMatrixAdapt(i, j)
	    = localRBF.eval(ublas::norm_2( ublas::column(grid, i) - ublas::column(grid, j)),
			    ublas::column(grid, i), ublas::column(grid, j));
	}
      }
      return iMatrixAdapt;
    }

    /**
     * Returns the data matrix used to create the interpolation object.
     * @return
     */
    ublas::matrix<value_type, ublas::column_major> getData() const {
      return data;
    }

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
     * @param tol Termination tolerance of the optimization algorithm. The calculated scale
     * is used for all computations after a call to this method.
     */
    void optimizeScale(const double tol = 1e-12) {
      MultiRBFInterpolator<RTYPE>& thisRef = *this;
      BrentsMethod<MultiRBFInterpolator<RTYPE> > bm(thisRef);
      // compute new shape parameter and adjust the interpolation object
      this->setNewScale(bm.minimum(this->scale, 100. * this->scale, tol));
    }

    /**
     * What scale parameter is used for the calculations.
     * @return
     */
    value_type getScale() {
      return this->scale;
    }

    ublas::vector<value_type> eval(const ublas::vector<value_type>& inPoint) const;
    
    void eval(const ublas::vector<value_type>& inPoint,
	      ublas::vector<value_type>& outPoint) const;

    ublas::vector<value_type> evalDiff(const ublas::vector<value_type>& inPoint,
	     const unsigned int alpha) const;

    ublas::vector<value_type> evalDiff2(const ublas::vector<value_type>& inPoint,
	      const unsigned alpha) const;

    ublas::vector<value_type> evalDiffMixed(const ublas::vector<value_type>& inPoint, const unsigned alpha1,
					    const unsigned alpha2) const;

    ublas::vector<value_type>
    evalGrad(const ublas::vector<value_type>& inPoint,
	     const unsigned alpha) const;

    ublas::matrix<value_type, ublas::column_major> evalJac(const ublas::vector<value_type>& inPoint) const;

    std::vector<ublas::matrix<value_type, ublas::column_major> > evalHess(const ublas::vector<value_type>& inPoint) const;

    virtual ~MultiRBFInterpolator();

  private:
    RTYPE rbf;
    value_type scale;

    ublas::matrix<value_type, ublas::column_major> grid;
    ublas::matrix<value_type, ublas::column_major> data;
    ublas::matrix<value_type, ublas::column_major> lambda;

    void init();
#ifndef _WITHOUT_SERIALIZATION_
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version) {

      ar & boost::serialization::base_object<MultiNDInterpolator>(*this);

      ar & rbf;
      ar & scale;
      ar & grid;
      ar & data;
      ar & lambda;
    }
#endif
  };

  template<class RTYPE>
  MultiRBFInterpolator<RTYPE>::MultiRBFInterpolator() {
    inDim = 0;
    outDim = 0;
    numOfPoints = 0;
  }
  
  template<class RTYPE>
  MultiRBFInterpolator<RTYPE>::MultiRBFInterpolator(const ublas::matrix<typename RTYPE::value_type, ublas::column_major>& grid,
						    const ublas::matrix<typename RTYPE::value_type, ublas::column_major>& data,
						    const typename RTYPE::value_type scale) :
    rbf(GaussianRBF<typename RTYPE::value_type> (scale)), scale(scale),
    grid(grid), data(ublas::trans(data)), lambda(ublas::trans(data)) {
    // initialize base members
    inDim = grid.size1();
    outDim = data.size1();
    numOfPoints = grid.size2();

    // input check, assume that grid has the right size
    if (data.size2() != numOfPoints)
      throw std::length_error(
			      "MultiRBFInterpolator::Constructor: Number of interpolation points (grid.size2()) and number of data points (data.size()) do not match.");
    
    // initialize the interpolation object
    init();
  }

  template<class RTYPE>
  inline ublas::vector<typename RTYPE::value_type> MultiRBFInterpolator<RTYPE>::eval(
										     const ublas::vector<typename RTYPE::value_type>& inPoint) const {

    // temp vector to hold evaluation of
    ublas::vector<typename RTYPE::value_type> evalRBF(numOfPoints);
    for (unsigned int i = 0; i < numOfPoints; i++) {
      evalRBF(i) = rbf.eval(ublas::norm_2(inPoint - ublas::column(grid, i)),
			    inPoint, ublas::column(grid, i));
    }
    // data^T*evalRBF
    return ublas::prod(evalRBF, lambda);
  }

  template<class RTYPE>
  ublas::vector<typename RTYPE::value_type> MultiRBFInterpolator<RTYPE>::evalDiff(
										  const ublas::vector<typename RTYPE::value_type>& inPoint,
										  const unsigned int alpha) const {
    ublas::vector<typename RTYPE::value_type> evalRBF(numOfPoints);
    for (unsigned i = 0; i < numOfPoints; i++) {
      evalRBF(i) = rbf.evalDiff1(
				 ublas::norm_2(inPoint - ublas::column(grid, i)), inPoint,
				 ublas::column(grid, i), alpha);
    }
    return ublas::prod(evalRBF, lambda);
  }

  template<class RTYPE>
  ublas::vector<typename RTYPE::value_type> MultiRBFInterpolator<RTYPE>::evalDiff2(
										   const ublas::vector<typename RTYPE::value_type>& inPoint,
										   const unsigned alpha) const {
    ublas::vector<typename RTYPE::value_type> evalRBF(numOfPoints);
    for (unsigned i = 0; i < numOfPoints; i++) {
      evalRBF(i) = rbf.evalDiff2(
				 ublas::norm_2(inPoint - ublas::column(grid, i)), inPoint,
				 ublas::column(grid, i), alpha);
    }
    return ublas::prod(evalRBF, lambda);
  }

  template<class RTYPE>
  ublas::vector<typename RTYPE::value_type> MultiRBFInterpolator<RTYPE>::evalDiffMixed(
										       const ublas::vector<typename RTYPE::value_type>& inPoint,
										       const unsigned alpha1, const unsigned alpha2) const {
    ublas::vector<typename RTYPE::value_type> evalRBF(numOfPoints);
    for (unsigned i = 0; i < numOfPoints; i++) {
      evalRBF(i) = rbf.evalDiffMixed(
				     ublas::norm_2(inPoint - ublas::column(grid, i)), inPoint,
				     ublas::column(grid, i), alpha1, alpha2);
    }
    return ublas::prod(evalRBF, lambda);
  }

  template<class RTYPE>
  ublas::vector<typename RTYPE::value_type> MultiRBFInterpolator<RTYPE>::evalGrad(
										  const ublas::vector<typename RTYPE::value_type>& inPoint,
										  const unsigned alpha) const {
    assert(alpha>0 && alpha < outDim);
    return ublas::row(evalJac(inPoint), alpha - 1);
  }

  template<class RTYPE>
  inline ublas::matrix<typename RTYPE::value_type, ublas::column_major> MultiRBFInterpolator<
    RTYPE>::evalJac(
		    const ublas::vector<typename RTYPE::value_type>& inPoint) const {
    ublas::matrix<typename RTYPE::value_type, ublas::column_major> jac(outDim,
								       inDim);
    for (unsigned i = 0; i < inDim; i++) {
      noalias(ublas::column(jac, i)) = evalDiff(inPoint, i + 1);
    }
    return jac;
  }

  template<class RTYPE>
  std::vector<ublas::matrix<typename RTYPE::value_type, ublas::column_major> > MultiRBFInterpolator<
    RTYPE>::evalHess(
		     const ublas::vector<typename RTYPE::value_type>& inPoint) const {

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

  template<class RTYPE>
  void MultiRBFInterpolator<RTYPE>::init() {

    // allocate space for the interpolation matrix, non-hermite interpolation --> one data point for every
    // interpolation point
    ublas::matrix<typename RTYPE::value_type, ublas::column_major> iMatrix(
									   numOfPoints, numOfPoints);

    // matrix will be symmetric --> use symmetric adaptor to compute just half of it.
    ublas::symmetric_adaptor < ublas::matrix<typename RTYPE::value_type,
					     ublas::column_major>, ublas::upper > iMatrixAdapt(iMatrix);
    for (unsigned int i = 0; i < numOfPoints; i++) {
      for (unsigned int j = i; j < numOfPoints; j++) {
	iMatrixAdapt(i, j) = rbf.eval(
				      ublas::norm_2(
						    ublas::column(grid, i) - ublas::column(grid, j)),
				      ublas::column(grid, i), ublas::column(grid, j));
      }
    }

    // solve and write out the solution, until here lambda contains data.
    lapack::posv(iMatrixAdapt, lambda);
  }

  template<class RTYPE>
  MultiRBFInterpolator<RTYPE>::~MultiRBFInterpolator() {

  }

}

#endif /* MULTIDIMRBFINTERPOLATOR_H_ */
