/**
 * RBFInterpolator.h
 *
 *  Created on: \date Oct 21, 2010
 *      Author: \author mrehberg
 */

#ifndef RBFINTERPOLATOR_H_
#define RBFINTERPOLATOR_H_

#include "../NDInterpolator.hpp"
#include "boost/numeric/ublas/symmetric.hpp"

// include types and driver to be able to use posv for solving
#include "boost/numeric/bindings/lapack/driver/posv.hpp"

//#include "boost/numeric/bindings/lapack/computational/potrf.hpp"
//#include "boost/numeric/bindings/lapack/computational/pocon.hpp"
//#include "boost/numeric/bindings/lapack/computational/potrs.hpp"

#include "boost/numeric/bindings/ublas/matrix.hpp"
#include "boost/numeric/bindings/ublas/symmetric.hpp"

#include "RBFs.hpp"

#include "goodc/goodc.hpp"

namespace NDInterpolator {

namespace lapack = boost::numeric::bindings::lapack;

/** \brief Radial basis function interpolator.
 *
 * Most basic interpolator class. Given a set of n-dimensional centers \f$ \Xi \f$ and
 * respective function values \f$ f_i \f$, such that
 * \f[ f(\xi_i)=f_i, \quad \xi_i \in \Xi \f]
 * holds for some unknown function \f$ f \f$ the class can be used to solve the problem of
 * finding a function \f$ h(x) \f$ such that the interpolation constrain is fullfilled, i.e.
 * \f[ h(\xi_i)=f_i \f].
 *
 * The class can also be used to evaluate derivatives of \f$ h(x) \f$ up to second order.
 *
 * To include derviative information, i.e. solve a Hermite interpolation problem use
 * NDInterpolator::RBFInterpolatorH.
 * The class keeps its own copy of the set of centers.
 * \tparam RTYPE Type of radial basis function to be used, e.g. NDInterpolator::GaussianRBF<double>
 */
template<class RTYPE>
class RBFInterpolator: public NDInterpolator {
public:
	typedef typename RTYPE::value_type value_type;

	/**
	 * Create RBF interpolator object with given grid, data and a scale factor for the radial basis function
	 * @param grid Grid of interpolation points.
	 * @param data Data in the interpolation points, make sure that the they are in the same order as the grid points.
	 * @param scale Greater than zero, scale factor for various RBFs.
	 * @return
	 */
	RBFInterpolator(const ublas::matrix<value_type, ublas::column_major>& grid,
			const ublas::vector<value_type>& data, const value_type scale);

	RBFInterpolator();

	/**
	 *	Assign new values to all members and re-init. After a call, the interpolation object
	 *	is ready to be used again.
	 * @param grid New grid.
	 * @param data New data vector.
	 * @param scale New scaling parameter for the RBF.
	 */
	void assign(const ublas::matrix<value_type, ublas::column_major>& grid,
			const ublas::vector<value_type>& data, const value_type scale) {
		// initialize base members
		inDim = grid.size1();
		numOfPoints = grid.size2();

		// input check, assume that grid has the right size
		if (data.size() != numOfPoints)
			throw std::length_error(
					"RBFInterpolator::assign: Number of interpolation points (grid.size2()) and number of data points (data.size()) do not match.");

		this->grid = grid;
		this->data = data;
		this->lambda = data;
		this->scale = scale;
		this->rbf = RTYPE(scale);
		// initialize the interpolation object
		init();
	}

	/**
	 * Set the scaling parameter for the RBF to a new value and re-init. After a call,
	 * the interpolation object is ready to be used again.
	 * @param scale New scale
	 */
	void setNewScale(const value_type scale) {
		this->scale = scale;
		this->rbf = RTYPE(scale);
		// init the interpolation object.
		init();
	}

	/**
	 * Try to calculate the optimal scaling parameter for the interpolator. The calculated scale
	 * is used for all computations after a call to this method.
	 * @param tol Termination tolerance of the optimization algorithm.
	 */
	void optimizeScale(const double tol=1e-12){
		RBFInterpolator<RTYPE>& thisRef=*this;
		BrentsMethod<RBFInterpolator<RTYPE> > bm(thisRef);
		// compute new shape parameter and adjust the interpolation object
		this->setNewScale(bm.minimum(this->scale, 100.*this->scale, tol));
	}

	/**
	 * What scale parameter is used for the calculations.
	 * @return
	 */
	value_type getScale(){
		return this->scale;
	}

	/**
	 * Computes and returns the interpolation matrix for a certain shape/scale parameter c. Attention:
	 * this does not mean the interpolator uses that c for interpolation after a call to this method.
	 * Use NDInterplator::RBFInterpolator::assign or NDInterpolator::RBFInterpolator::setNewScale
	 * for that purpose.
	 * @param c Scale/shape parameter that is used in calculating the interpolation matrix
	 * @return
	 */
	ublas::matrix<value_type, ublas::column_major> getInterpMatrix(
			const value_type c) const {

		// local RBF
		RTYPE localRBF(c);

		// allocate space for the interpolation matrix, non-hermite interpolation --> one data point for every
		// interpolation point
		ublas::matrix<value_type, ublas::column_major> iMatrix(numOfPoints,
				numOfPoints);

		// matrix will be symmetric --> use symmetric adaptor to compute just half of it.
		ublas::symmetric_adaptor<
				ublas::matrix<value_type, ublas::column_major>, ublas::upper>
				iMatrixAdapt(iMatrix);
		for (unsigned int i = 0; i < numOfPoints; i++) {
			for (unsigned int j = i; j < numOfPoints; j++) {
				iMatrixAdapt(i, j) = localRBF.eval(ublas::norm_2(ublas::column(
						grid, i) - ublas::column(grid, j)), ublas::column(grid,
						j), ublas::column(grid, j));
			}
		}
		return iMatrixAdapt;
	}

	/**
	 * Returns the data used to create the interpolation object as a matrix (!).
	 * @return
	 */
	ublas::matrix<value_type, ublas::column_major> getData() const {
		ublas::matrix<value_type, ublas::column_major> ret(numOfPoints, 1);
		noalias(ublas::column(ret, 0)) = data;
		return ret;
	}

	value_type eval(const ublas::vector<value_type>& inPoint) const;

	value_type evalDiff(const ublas::vector<value_type>& inPoint,
			const unsigned int alpha) const;

	value_type
			evalDiff2(const ublas::vector<value_type>& inPoint,
					const unsigned alpha) const;

	value_type evalDiffMixed(const ublas::vector<value_type>& inPoint,
			const unsigned alpha1, const unsigned alpha2) const;

	ublas::vector<value_type>
	evalGrad(const ublas::vector<value_type>& inPoint) const;

	ublas::matrix<value_type>
	evalHess(const ublas::vector<value_type>& inPoint) const;

	//	ublas::matrix<value_type> getIMatrix() const {
	//		return this->iMatrix;
	//	}
	virtual ~RBFInterpolator();

private:
	RTYPE rbf;
	value_type scale;

	ublas::matrix<value_type, ublas::column_major> grid;
	ublas::vector<value_type> data;
	ublas::vector<value_type> lambda;

	void init();

#ifndef _WITHOUT_SERIALIZATION_
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {

		ar & boost::serialization::base_object<NDInterpolator>(*this);

		ar & rbf;
		ar & scale;
		ar & grid;
		ar & data;
		ar & lambda;
	}
#endif
};

template<class RTYPE>
RBFInterpolator<RTYPE>::RBFInterpolator() {
	inDim = 0;
	numOfPoints = 0;
}

template<class RTYPE>
RBFInterpolator<RTYPE>::RBFInterpolator(const ublas::matrix<
		typename RTYPE::value_type, ublas::column_major>& grid,
		const ublas::vector<typename RTYPE::value_type>& data,
		const typename RTYPE::value_type c) :
	rbf(RTYPE(c)), scale(c), grid(grid), data(data), lambda(data) {
	// initialize base members
	inDim = grid.size1();
	numOfPoints = grid.size2();

	// input check, assume that grid has the right size
	if (data.size() != numOfPoints)
		throw std::length_error(
				"RBFInterpolator::Constructor: Number of interpolation points (grid.size2()) and number of data points (data.size()) do not match.");

	// initialize the interpolation object
	init();
}

template<class RTYPE>
inline typename RTYPE::value_type RBFInterpolator<RTYPE>::eval(
		const ublas::vector<typename RTYPE::value_type>& inPoint) const {
	typename RTYPE::value_type ret = 0.;
	for (unsigned int i = 0; i < numOfPoints; i++) {
		ret += lambda(i) * rbf.eval(ublas::norm_2(inPoint - ublas::column(grid,
				i)), inPoint, ublas::column(grid, i));
	}
	return ret;
}

template<class RTYPE>
inline typename RTYPE::value_type RBFInterpolator<RTYPE>::evalDiff(
		const ublas::vector<typename RTYPE::value_type>& inPoint,
		const unsigned alpha) const {
	typename RTYPE::value_type ret = 0.;
	for (unsigned i = 0; i < numOfPoints; i++) {
		ret += lambda(i) * rbf.evalDiff1(ublas::norm_2(inPoint - ublas::column(
				grid, i)), inPoint, ublas::column(grid, i), alpha);
	}
	return ret;
}

template<class RTYPE>
inline typename RTYPE::value_type RBFInterpolator<RTYPE>::evalDiff2(
		const ublas::vector<typename RTYPE::value_type>& inPoint,
		const unsigned alpha) const {
	typename RTYPE::value_type ret = 0.;
	for (unsigned i = 0; i < numOfPoints; i++) {
		ret += lambda(i) * rbf.evalDiff2(ublas::norm_2(inPoint - ublas::column(
				grid, i)), inPoint, ublas::column(grid, i), alpha);
	}
	return ret;
}

template<class RTYPE>
inline typename RTYPE::value_type RBFInterpolator<RTYPE>::evalDiffMixed(
		const ublas::vector<typename RTYPE::value_type>& inPoint,
		const unsigned alpha1, const unsigned alpha2) const {
	typename RTYPE::value_type ret = 0.;
	for (unsigned i = 0; i < numOfPoints; i++) {
		ret += lambda(i) * rbf.evalDiffMixed(ublas::norm_2(inPoint
				- ublas::column(grid, i)), inPoint, ublas::column(grid, i),
				alpha1, alpha2);
	}
	return ret;
}

template<class RTYPE>
inline ublas::vector<typename RTYPE::value_type> RBFInterpolator<RTYPE>::evalGrad(
		const ublas::vector<typename RTYPE::value_type>& inPoint) const {
	ublas::vector<typename RTYPE::value_type> retVec(inDim);
	for (unsigned i = 0; i < inDim; i++)
		retVec(i) = 0;

	typename RTYPE::value_type norm;
	for (unsigned i = 0; i < numOfPoints; i++) {
		norm = ublas::norm_2(inPoint - ublas::column(grid, i));
		for (unsigned j = 0; j < inDim; j++) {
			retVec(j) += lambda(i) * rbf.evalDiff1(norm, inPoint,
					ublas::column(grid, i), j + 1);
		}
	}
	//	for (unsigned i=0; i<inDim;i++){
	//		retVec(i)=this->evalDiff(inPoint,i+1);
	//	}
	return retVec;
}

template<class RTYPE>
inline ublas::matrix<typename RTYPE::value_type> RBFInterpolator<RTYPE>::evalHess(
		const ublas::vector<typename RTYPE::value_type>& inPoint) const {
	ublas::matrix<typename RTYPE::value_type, ublas::column_major> ret(inDim,
			inDim);
	ublas::symmetric_adaptor<ublas::matrix<typename RTYPE::value_type,
			ublas::column_major>, ublas::upper> retAdapt(ret);
	for (unsigned i = 0; i < inDim; i++) {
		for (unsigned j = i + 1; j < inDim; j++) {
			retAdapt(i, j) = this->evalDiffMixed(inPoint, i + 1, j + 1);
		}
		retAdapt(i, i) = this->evalDiff2(inPoint, i + 1);
	}
	return ret;
}

template<class RTYPE>
void RBFInterpolator<RTYPE>::init() {

	// temporary matrix for holding data to be able to use lapack (needs Matrix)
	ublas::matrix<typename RTYPE::value_type, ublas::column_major> b(
			numOfPoints, 1);
	noalias(ublas::column(b, 0)) = data;

	// solve and write out the solution
	ublas::matrix<typename RTYPE::value_type, ublas::column_major> iMatrix =
			this->getInterpMatrix(scale);
	ublas::symmetric_adaptor<ublas::matrix<typename RTYPE::value_type,
			ublas::column_major>, ublas::upper> iMatrixAdapt(iMatrix);

	lapack::posv(iMatrixAdapt, b);
	noalias(lambda) = ublas::column(b, 0);


//	// rcond wird später die Ausgabe, anorm ist Eingabeparameter
//	double rcond;
//	double anorm=ublas::norm_1(iMatrixAdapt);
//	// Choleskyzerlegung von A, wobei A überschrieben wird.
//	lapack::potrf(iMatrixAdapt);
//	// Konditionszahl ausrechnen, nach dem call steht die in rcond
//	lapack::pocon(iMatrixAdapt,anorm,rcond);
//	// gleichungssystem lösen, Rechte Seite b wird mit Lösung überschrieben.
//	lapack::potrs(iMatrixAdapt,b);
//	noalias(lambda) = ublas::column(b, 0);

}

template<class RTYPE>
RBFInterpolator<RTYPE>::~RBFInterpolator() {
}

}

#endif /* RBFINTERPOLATOR_H_ */
