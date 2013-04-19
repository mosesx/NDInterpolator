#ifndef P_D_GENERAL_MULTIQUADRIC_RBF_H
#define P_D_GENERAL_MULTIQUADRIC_RBF_H

#include <cmath>
#include <cassert>

#include <stdio.h> //for using 'abort()'
#include <stdlib.h>

// boost includes
#include "boost/numeric/ublas/matrix.hpp"
#include "boost/numeric/ublas/matrix_proxy.hpp"
#include "boost/numeric/ublas/vector.hpp"

// recursion templates for basis function
#include "KernelRecursive.hpp"

// NDInterpolator includes
#include "../tools/archive.hpp"

namespace NDInterpolator {

namespace ublas = boost::numeric::ublas;

/**
 * \brief Implements General multiquadrics (BETA < 0 corresponds to the inverse multiquadric), i.e. $\f[ \phi(x,y) := \left(1+ 2(0.5*(r/c)^2)^{\beta/2}, \quad r:= \|x-y\|_2^2, x,y \in \mathbf{R}^n \right)\f]
 *
 * NOTE: Schaback uses a slightly other scaling than the usual one given by \f$ (c^2 + r^2)^{\beta/2}\f$. Note that an outer parameter does not affect the intrpolation linear system, yet it produces differently scaled values. This is evident if you compare them: \f$ (c^2 + r^2)^{\beta/2} \not= (1+r^2/c^2)^{\beta/2}. \f$
 *
 * CAUTION: Only the inverse multiquadric is a positive definite rbf!
 *
 * It is possible that this class has some (numerical as well as run-time) advantages towards the 'InverseMultiquadric'
 *
 * Reference: [R. SCHABACK, <I>"Programming Hints for Kernel-Based Methods"</I>,  Draft 2010, esp. p. 5 (inv.) multiquadric, 13 (1st derivative), 16 (2nd derivatives)]
 *
 * \tparam BETA exponent of the multiquadric \f$ \beta \f$
 * \tparam T Template parameter specifying the type of the entries of \f$ x,y \f$.
 */
template<int BETA, class T>
class GeneralMultiquadricRBF {
public:
	typedef T value_type;
	typedef ublas::vector<T> UblasVecType;

	//!you can also adjust the second parameter as a special feature
	GeneralMultiquadricRBF(const T& c = 1., const T& b = 1.) :
	c_(c){//( (c < T()) ? -c : c ) {
	  //assert(c > T()); //must be strictly positive, but in any rbf so far (esp. in this one) $c$ enters as $c^n, \ n \in 2\mathbf{Z}$ 
	  assert(std::abs(c) > T()); //don't permit 0 since one devides by c
	}

	const T& getScale() const {
		return c_;
	}
	T& getScale() {
		return c_;
	}

	T eval(const T& r, const UblasVecType& x, const UblasVecType& xj) const {
		return mq_derivative<0, BETA> (r, c_);
	}

	T eval(const value_type& r) const {
			return mq_derivative<0, BETA> (r,c_);
		}

	//! \f$ \frac{\partial}{\partial x_i} \phi(x,y) \f$
	T evalDiff1(const T& r, const UblasVecType& x, const UblasVecType& xj,
			const unsigned alpha) const {
		return TMP::ntimesx<-2>(c_) * mq_derivative<1, BETA> (r, c_) * (x(alpha
				- 1) - xj(alpha - 1));
	}

	//! \f$\frac{\partial^2}{\partial x_i \partial x_i} \phi(x,y) \f$
	T evalDiff2(const T& r, const UblasVecType& x, const UblasVecType& xj,
			const unsigned alpha) const {
		return TMP::ntimesx<-2>(c_) * mq_derivative<1, BETA> (r, c_)
				+ mq_derivative<2, BETA> (r, c_) * TMP::ntimesx<-4>(c_)
						* TMP::ntimesx<2>(x(alpha - 1) - xj(alpha - 1));
	}

	//! \f$ \frac{\partial^2}{\partial x_i \partial x_k} \phi(x,y), \ i \not= j \f$. Note that parameter vec 'xj' is fixed and we only take derivatives w.r.t. vec 'x'.
	T evalDiffMixed(const T& r, const UblasVecType& x, const UblasVecType& xj,
			const unsigned alpha1, const unsigned alpha2) const {
		return mq_derivative<2, BETA> (r, c_) * TMP::ntimesx<-4>(c_) * (x(
				alpha1 - 1) - xj(alpha1 - 1))
				* (x(alpha2 - 1) - xj(alpha2 - 1));
	}

	//! ******* The following function bodies haven't been defined so far due to laziness ;). For reasons concerning security of the computation, an error will be printed and the computation will be stopped immediately.
	T evalDiff3(const T& r, const UblasVecType& x, const UblasVecType& xj,
			const unsigned alpha) const {
		std::cout
				<< "**** ERROR: Derivative 'evalDiff3' for GeneralMultiquadricRBF not yet defined "
				<< std::endl;
		abort();
	}

	T evalDiff3DMixed(const T& r, const UblasVecType& x,
			const UblasVecType& xj, const unsigned alpha1,
			const unsigned alpha2) const {
		std::cout
				<< "**** ERROR: Derivative 'evalDiff3DMixed' for GeneralMultiquadricRBF not yet defined "
				<< std::endl;
		abort();
	}

	T evalDiff3Mixed(const T& r, const UblasVecType& x, const UblasVecType& xj,
			const unsigned alpha1, const unsigned alpha2, const unsigned alpha3) const {
		std::cout
				<< "**** ERROR: Derivative 'evalDiff3Mixed' for GeneralMultiquadricRBF not yet defined "
				<< std::endl;
		abort();
	}

private:
	T c_;

#ifndef _WITHOUT_SERIALIZATION_
	// serialization, all we need to know is the shape parameter
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
		ar & c_;
	}
#endif

};

} //end namespace 

#endif 
