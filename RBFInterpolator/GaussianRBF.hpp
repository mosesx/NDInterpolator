/**
 * GaussianRBF.h
 *
 *  Created on: \date Oct 21, 2010
 *      Author: \author mrehberg
 * 
 *  Changed to meet definitive compile time evaluation on: \date Dec 1, 2010
 *  by marc
 *      
 */

#ifndef GAUSSIANRBF_H_
#define GAUSSIANRBF_H_

// std includes
#include <cmath>
#include <math.h>
// ublas includes
#include "boost/numeric/ublas/matrix.hpp"
#include "boost/numeric/ublas/matrix_proxy.hpp"
#include "boost/numeric/ublas/vector.hpp"
// NDInterpolator includes
#include "../tools/templateProgs.hpp"
#include "../tools/archive.hpp"

namespace NDInterpolator {

namespace ublas = boost::numeric::ublas;

/**
 * \brief Implementation of the Gaussian radial basis function
 *
 * The Gaussian basis function is defined as
 * \f[ \phi(x,y)=\mathrm{e}^{-c^2 \|x-y\|_2} \f]
 * with \f$c>0\f$ as scaling factor.
 * \tparam valueType Template parameter specifying the type of the entries of \f$ x,y \f$.
 */
template<class valueType>
class GaussianRBF {
public:
	// typedef to be able to use in template expressions from other classes
	typedef valueType value_type;

	//!serves also as default construction. NOTE: since \f$ c\f$ enters quadratically (and powers of it) everywhere, it does not matter whether c is positive or negative
	GaussianRBF(const value_type& scale = 1.) :
		c(scale) {
		assert(std::abs(c) > valueType()); //!o.k. it shouldn't be zero
	}

	value_type gaussian(const value_type& r) const {
		return exp(-TMP::ntimesx<2>(c * r));
	}

	value_type eval(const value_type& r, const ublas::vector<value_type>& x,
			const ublas::vector<value_type>& xj) const {
		return gaussian(r);
	}

	value_type eval(const value_type& r) const {
		return gaussian(r);
	}

	value_type evalDiff1(const value_type& r,
			const ublas::vector<value_type>& x,
			const ublas::vector<value_type>& xj, const unsigned alpha) const {
		return -2 * TMP::ntimesx<2>(c) * (x(alpha - 1) - xj(alpha - 1))
				* gaussian(r);
	}

	value_type evalDiff2(const value_type& r,
			const ublas::vector<value_type>& x,
			const ublas::vector<value_type>& xj, const unsigned alpha) const {
		return -2 * TMP::ntimesx<2>(c) * gaussian(r) * (1 - 2
				* TMP::ntimesx<2>(c) * TMP::ntimesx<2>(
				x(alpha - 1) - xj(alpha - 1)));
	}

	value_type evalDiffMixed(const value_type& r,
			const ublas::vector<value_type>& x,
			const ublas::vector<value_type>& xj, const unsigned alpha1,
			const unsigned alpha2) const {

		return 4 * TMP::ntimesx<4>(c) * (x(alpha1 - 1) - xj(alpha1 - 1)) * (x(
				alpha2 - 1) - xj(alpha2 - 1)) * gaussian(r);

	}

	value_type evalDiff3(const value_type& r,
			const ublas::vector<value_type>& x,
			const ublas::vector<value_type>& xj, const unsigned alpha) const {
		return 4 * TMP::ntimesx<4>(c) * gaussian(r) * (x(alpha - 1) - xj(
				alpha - 1)) * (3 - 2 * TMP::ntimesx<2>(c) * TMP::ntimesx<2>(
				(x(alpha - 1) - xj(alpha - 1))));
	}

	value_type evalDiff3DMixed(const value_type& r,
			const ublas::vector<value_type>& x,
			const ublas::vector<value_type>& xj, const unsigned alpha1,
			const unsigned alpha2) const {
		return 4 * TMP::ntimesx<4>(c) * gaussian(r) * (1 - 2 * TMP::ntimesx<2>(
				c) * TMP::ntimesx<2>(x(alpha1 - 1) - xj(alpha1 - 1))) * (x(
				alpha2 - 1) - xj(alpha2 - 1));
	}

	value_type evalDiff3Mixed(const value_type& r,
			const ublas::vector<value_type>& x,
			const ublas::vector<value_type>& xj, const unsigned alpha1,
			const unsigned alpha2, const unsigned alpha3) const {
		return -8 * TMP::ntimesx<6>(c) * (x(alpha1 - 1) - xj(alpha1 - 1)) * (x(
				alpha2 - 1) - xj(alpha2 - 1))
				* (x(alpha3 - 1) - xj(alpha3 - 1)) * gaussian(r);
	}

	const value_type& getScale() const {
		return c;
	}
	value_type& getScale() {
		return c;
	}

private:
	value_type c;
#ifndef _WITHOUT_SERIALIZATION_
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
		ar & c;
	}
#endif
};
}

#endif /* GAUSSIANRBF_H_ */
