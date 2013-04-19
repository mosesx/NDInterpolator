/**
 * RBFInterpolatorH.h
 *
 *  Created on: \date Nov 4, 2010
 *      Author: \author mrehberg
 */

#ifndef RBFINTERPOLATORH_H_
#define RBFINTERPOLATORH_H_

#include "../NDInterpolator.hpp"
#include "boost/numeric/ublas/symmetric.hpp"

// include types and driver to be able to use posv for solving
#include "boost/numeric/bindings/lapack/driver/gesv.hpp"
#include "boost/numeric/bindings/ublas/matrix.hpp"
#include "boost/numeric/bindings/ublas/vector.hpp"

#include "RBFs.hpp"

#include "./goodc/goodc.hpp"

namespace NDInterpolator {

namespace lapack = boost::numeric::bindings::lapack;

/** \brief Radial Basis Function interpolator for hermite interpolation.
 *
 * In contrast to NDInterpolator::RBFInterpolator the class will also use derivative information
 * for interpolation. That means besides the values \f$ f(\xi)\f$ on can also specify
 * \f$ \nabla f(\xi) \f$.
 * The class will keep its own copy of the centers.
 * \tparam RTYPE Type of radial basis function to be used, e.g. NDInterpolator::GaussianRBF<double>
 */
 template<class RTYPE>
   class RBFInterpolatorH: public NDInterpolator {
public:
	typedef typename RTYPE::value_type value_type;

	/**
	 * Create RBF interpolator object with given grid, data, first order differential data diffData and a scale factor for the radial basis function
	 * @param grid Grid of interpolation points.
	 * @param data Data in the interpolation points, make sure that the they are in the same order as the grid points.
	 * @param diffData First order differential data, each column is the gradient for one data point.
	 * @param scale Greater than zero, scale factor for various RBFs.
	 * @return
	 */
	RBFInterpolatorH(
			const ublas::matrix<value_type, ublas::column_major>& grid,
			const ublas::vector<value_type>& data,
			const ublas::matrix<value_type, ublas::column_major>& diffData,
			const value_type scale);

	RBFInterpolatorH();

	/**
	 * Assign new values to all members and re-init. After a call, the interpolation object
	 * is ready to be used again.
	 * @param grid New grid.
	 * @param data New data.
	 * @param diffData New derivative data.
	 * @param scale New scale.
	 */
	void assign(const ublas::matrix<value_type, ublas::column_major>& grid,
			const ublas::vector<value_type>& data,
			const ublas::matrix<value_type, ublas::column_major>& diffData,
			const value_type scale) {
		// initialize base members
		inDim = grid.size1();
		numOfPoints = grid.size2();

		// input check, assume that grid has the right size
		if (data.size() != numOfPoints)
			throw std::length_error(
					"RBFInterpolatorH::Constructor: Number of interpolation points (grid.size2()) and number of data points (data.size()) do not match.");

		if (diffData.size1() != inDim || diffData.size2() != numOfPoints)
			throw std::length_error(
					"RBFInterpolatorH::Constructor: Number of columns of diffData does not match number of interpolation points in grid or number of rows of diffData does not match inDim");
		// initialize the interpolation object
		this->rbf = RTYPE(scale);
		this->grid = grid;
		this->data = data;
		this->diffData = diffData;
		this->lambda = data;

		// reinit the object
		init();
	}

	/**
	 * Set the scaling parameter for the RBF to a new value and re-init. After a call,
	 * the interpolation object is ready to be used again.
	 * @param scale New scale
	 */
	void setNewScale(const value_type scale) {
		this->rbf = RTYPE(scale);
		init();
	}

	/**
	 * Try to calculate the optimal scaling parameter for the interpolator.
	 * @param tol Termination tolerance of the optimization algorithm. The calculated scale
	 * is used for all computations after a call to this method.
	 */
	void optimizeScale(const double tol = 1e-12) {
		RBFInterpolatorH<RTYPE>& thisRef = *this;
		BrentsMethod<RBFInterpolatorH<RTYPE> > bm(thisRef);
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

	/**
	 * Computes and returns the interpolation matrix for a certain shape/scale parameter c. Attention:
	 * this does not mean the interpolator uses that c for interpolation after a call to this method.
	 * Use NDInterplator::RBFInterpolatorH::assign or NDInterpolator::RBFInterpolatorH::setNewScale
	 * for that purpose.
	 * @param scale Scale/shape parameter that is used in calculating the interpolation matrix
	 * @return Interpolation matrix for scale.
	 */
	ublas::matrix<value_type, ublas::column_major> getInterpMatrix(
			const value_type scale) const {
		// local RBF
		RTYPE localRBF(scale);

		// allocate space for the interpolation matrix, non-hermite interpolation --> one data point for every
		// interpolation point
		ublas::matrix<value_type, ublas::column_major> iMatrix(numOfPoints,
				numOfPoints);

		// matrix will be symmetric --> use symmetric adaptor to compute just half of it.
		ublas::symmetric_adaptor < ublas::matrix<value_type,
				ublas::column_major>, ublas::upper > iMatrixAdapt(iMatrix);
		for (unsigned int i = 0; i < numOfPoints; i++) {
			for (unsigned int j = i; j < numOfPoints; j++) {
				iMatrixAdapt(i, j)
						= localRBF.eval(
								ublas::norm_2(
										ublas::column(grid, i) - ublas::column(
												grid, j)),
								ublas::column(grid, j), ublas::column(grid, j));
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

	virtual ~RBFInterpolatorH();

private:
	RTYPE rbf;
	value_type scale;

	ublas::matrix<value_type, ublas::column_major> grid;
	ublas::vector<value_type> data;
	ublas::matrix<value_type, ublas::column_major> diffData;
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
		ar & diffData;
		ar & lambda;
	}
#endif
};

template<class RTYPE>
RBFInterpolatorH<RTYPE>::RBFInterpolatorH() {
	inDim = 0;
	numOfPoints = 0;
}

template<class RTYPE>
RBFInterpolatorH<RTYPE>::RBFInterpolatorH(
		const ublas::matrix<value_type, ublas::column_major>& grid,
		const ublas::vector<typename RTYPE::value_type>& data,
		const ublas::matrix<typename RTYPE::value_type, ublas::column_major>& diffData,
		const typename RTYPE::value_type c) :
	rbf(RTYPE(c)), scale(c), grid(grid), data(data), diffData(diffData),
			lambda(data) {
	// initialize base members
	inDim = grid.size1();
	numOfPoints = grid.size2();

	// input check, assume that grid has the right size
	if (data.size() != numOfPoints)
		throw std::length_error(
				"RBFInterpolatorH::Constructor: Number of interpolation points (grid.size2()) and number of data points (data.size()) do not match.");

	if (diffData.size1() != inDim || diffData.size2() != numOfPoints)
		throw std::length_error(
				"RBFInterpolatorH::Constructor: Number of columns of diffData does not match number of interpolation points in grid or number of rows of diffData does not match inDim");
	// initialize the interpolation object
	init();
}

template<class RTYPE>
inline typename RTYPE::value_type RBFInterpolatorH<RTYPE>::eval(
		const ublas::vector<typename RTYPE::value_type>& inPoint) const {
	typename RTYPE::value_type ret = 0.;
	for (unsigned int i = 0; i < numOfPoints; i++) {
		ret += lambda(i * (inDim + 1)) * rbf.eval(
				ublas::norm_2(inPoint - ublas::column(grid, i)), inPoint,
				ublas::column(grid, i));
		for (unsigned int j = 0; j < inDim; j++)
			ret += lambda(i * (inDim + 1) + j + 1) * rbf.evalDiff1(
					ublas::norm_2(inPoint - ublas::column(grid, i)), inPoint,
					ublas::column(grid, i), j + 1);
	}
	return ret;
}

template<class RTYPE>
inline typename RTYPE::value_type RBFInterpolatorH<RTYPE>::evalDiff(
		const ublas::vector<typename RTYPE::value_type>& inPoint,
		const unsigned alpha) const {
	typename RTYPE::value_type ret = 0.;
	for (unsigned i = 0; i < numOfPoints; i++) {
		ret += lambda(i * (inDim + 1)) * rbf.evalDiff1(
				ublas::norm_2(inPoint - ublas::column(grid, i)), inPoint,
				ublas::column(grid, i), alpha);
		for (unsigned int j = 0; j < inDim; j++) {
			if (j + 1 != alpha)
				ret += lambda(i * (inDim + 1) + j + 1) * rbf.evalDiffMixed(
						ublas::norm_2(inPoint - ublas::column(grid, i)),
						inPoint, ublas::column(grid, i), alpha, j + 1);
			else
				ret += lambda(i * (inDim + 1) + j + 1) * rbf.evalDiff2(
						ublas::norm_2(inPoint - ublas::column(grid, i)),
						inPoint, ublas::column(grid, i), alpha);
		}
	}
	return ret;
}

template<class RTYPE>
inline typename RTYPE::value_type RBFInterpolatorH<RTYPE>::evalDiff2(
		const ublas::vector<typename RTYPE::value_type>& inPoint,
		const unsigned alpha) const {
	typename RTYPE::value_type ret = 0.;
	for (unsigned i = 0; i < numOfPoints; i++) {
		ret += lambda(i * (inDim + 1)) * rbf.evalDiff2(
				ublas::norm_2(inPoint - ublas::column(grid, i)), inPoint,
				ublas::column(grid, i), alpha);
		for (unsigned j = 0; j < inDim; j++) {
			if (j + 1 != alpha)
				ret += lambda(i * (inDim + 1) + j + 1) * rbf.evalDiff3DMixed(
						ublas::norm_2(inPoint - ublas::column(grid, i)),
						inPoint, ublas::column(grid, i), alpha, j + 1);
			else
				ret += lambda(i * (inDim + 1) + j + 1) * rbf.evalDiff3(
						ublas::norm_2(inPoint - ublas::column(grid, i)),
						inPoint, ublas::column(grid, i), alpha);
		}
	}
	return ret;
}

template<class RTYPE>
inline typename RTYPE::value_type RBFInterpolatorH<RTYPE>::evalDiffMixed(
		const ublas::vector<typename RTYPE::value_type>& inPoint,
		const unsigned alpha1, const unsigned alpha2) const {
	typename RTYPE::value_type ret = 0.;
	for (unsigned i = 0; i < numOfPoints; i++) {
		ret += lambda(i * (inDim + 1)) * rbf.evalDiffMixed(
				ublas::norm_2(inPoint - ublas::column(grid, i)), inPoint,
				ublas::column(grid, i), alpha1, alpha2);
		for (unsigned j = 0; j < inDim; j++) {
			if (j + 1 != alpha1 && j + 1 != alpha2)
				ret += lambda(i * (inDim + 1) + j + 1) * rbf.evalDiff3Mixed(
						ublas::norm_2(inPoint - ublas::column(grid, i)),
						inPoint, ublas::column(grid, i), alpha1, alpha2, j + 1);
			else if (j + 1 != alpha1) // j+1 == alpha2
				ret += lambda(i * (inDim + 1) + j + 1) * rbf.evalDiff3DMixed(
						ublas::norm_2(inPoint - ublas::column(grid, i)),
						inPoint, ublas::column(grid, i), alpha2, alpha1);
			else
				// j+1 == alpha 1
				ret += lambda(i * (inDim + 1) + j + 1) * rbf.evalDiff3DMixed(
						ublas::norm_2(inPoint - ublas::column(grid, i)),
						inPoint, ublas::column(grid, i), alpha1, alpha2);
		}
	}

	return ret;
}

template<class RTYPE>
inline ublas::vector<typename RTYPE::value_type> RBFInterpolatorH<RTYPE>::evalGrad(
		const ublas::vector<typename RTYPE::value_type>& inPoint) const {
	ublas::vector<typename RTYPE::value_type> ret(inDim);
	for (unsigned i = 0; i < inDim; i++)
		ret(i) = this->evalDiff(inPoint, i + 1);

	return ret;
}

template<class RTYPE>
inline ublas::matrix<typename RTYPE::value_type> RBFInterpolatorH<RTYPE>::evalHess(
		const ublas::vector<typename RTYPE::value_type>& inPoint) const {
	ublas::matrix<typename RTYPE::value_type, ublas::column_major> ret(inDim,
			inDim);
	ublas::symmetric_adaptor < ublas::matrix<typename RTYPE::value_type,
			ublas::column_major>, ublas::upper > retAdapt(ret);
	for (unsigned i = 0; i < inDim; i++) {
		for (unsigned j = i + 1; j < inDim; j++) {
			retAdapt(i, j) = this->evalDiffMixed(inPoint, i + 1, j + 1);
		}
		retAdapt(i, i) = this->evalDiff2(inPoint, i + 1);
	}

	return ret;
}

/*
 * init the interpolation object. Unlike in the nonhermitian case the interpolation matrix
 * is not symmetric.
 */
template<class RTYPE>
void RBFInterpolatorH<RTYPE>::init() {
	// allocate space for the interpolation matrix, non-hermite interpolation --> one data point for every
	// interpolation point
	ublas::matrix<typename RTYPE::value_type, ublas::column_major> iMatrix(
			numOfPoints * (inDim + 1), numOfPoints * (inDim + 1));

	// temporary matrix for holding data to be able to use lapack (needs Matrix)
	ublas::matrix<typename RTYPE::value_type, ublas::column_major> b(
			numOfPoints * (inDim + 1), 1);

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
		b(i * (inDim + 1), 0) = data(i);
		// can't use columns since subrange is a matrix (again remember stop index + 1)
		noalias(
				ublas::subrange(b, i * (inDim + 1) + 1,
						i * (inDim + 1) + inDim + 1, 0, 1)) = ublas::subrange(
				diffData, 0, inDim, i, i + 1);

	}// i

	// solve and write out the solution
	ublas::vector<int> ipiv(numOfPoints * (inDim + 1));
	lapack::gesv(iMatrix, ipiv, b);

	// debug
	lambda.resize(numOfPoints * (inDim + 1));
	noalias(lambda) = ublas::column(b, 0);
}

template<class RTYPE>
RBFInterpolatorH<RTYPE>::~RBFInterpolatorH() {
}

}

#endif /* RBFINTERPOLATORH_H_ */
