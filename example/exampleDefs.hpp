/**
 * exampleDefs.hpp
 *
 *  Created on: \date Jun 16, 2011
 *      Author: \author mrehberg
 */

#ifndef EXAMPLEDEFS_HPP_
#define EXAMPLEDEFS_HPP_

#include "boost/numeric/ublas/vector.hpp"
namespace ublas = boost::numeric::ublas;

inline double sin(const ublas::vector<double>& inPoint) {
  return sin(inPoint(0));
}

inline double peaks1d(const ublas::vector<double>& inPoint) {
  return 3*(1-inPoint(0))*(1-inPoint(0))*exp(-(inPoint(0)*inPoint(0))) 
    - 10*(inPoint(0)/5 - inPoint(0)*inPoint(0)*inPoint(0))*exp(-inPoint(0)*inPoint(0))  
    - 1/3*exp(-(inPoint(0)+1)*(inPoint(0)+1));
}

inline double dpeaks1d(const ublas::vector<double>& inPoint) {
  return (2*(inPoint(0)+1)*exp(-(inPoint(0)+1)*(inPoint(0)+1)))/3
    + 20*inPoint(0)*(inPoint(0)/5 - inPoint(0)*inPoint(0)*inPoint(0))*exp(-inPoint(0)*inPoint(0))
    - 10*(1/5 - 3*inPoint(0)*inPoint(0))*exp(-inPoint(0)*inPoint(0))
    - 6*(1-inPoint(0))*(1-inPoint(0))*inPoint(0)*exp(-inPoint(0)*inPoint(0))
    - 6*(1-inPoint(0))*exp(-inPoint(0)*inPoint(0));
}

inline double peaks(const ublas::vector<double>& inPoint) {
  return 3*(1-inPoint(0))*(1-inPoint(0))*exp(-(inPoint(0)*inPoint(0)) - (inPoint(1)+1)*(inPoint(1)+1)) 
    - 10*(inPoint(0)/5 - inPoint(0)*inPoint(0)*inPoint(0) - 
	  inPoint(1)*inPoint(1)*inPoint(1)*inPoint(1)*inPoint(1))*exp(-inPoint(0)*inPoint(0)-
								      inPoint(1)*inPoint(1))  
    - 1/3*exp(-(inPoint(0)+1)*(inPoint(0)+1) - inPoint(1)*inPoint(1));
}

/**
 * Analytic form of the rosenbrock function for testing.
 * @param inPoint Point at which to evaluate the function.
 * @return
 */
inline double rosenbrock(const ublas::vector<double>& inPoint) {

	return 100 * (inPoint(1) - inPoint(0) * inPoint(0)) * (inPoint(1)
			- inPoint(0) * inPoint(0)) + (1 - inPoint(0)) * (1 - inPoint(0));
}

/**
 * Analytic form of the first derivatives of the rosenbrock function.
 * @param inPoint Point at which to evaluate.
 * @param alpha Direction of derivative.
 * @return
 */
inline double diffRosenbrock(const ublas::vector<double>& inPoint,
		const unsigned alpha) {
	if (alpha == 1)
		return -2 * (1 - inPoint(0)) - 400 * inPoint(0) * (-inPoint(0)
				* inPoint(0) + inPoint(1));
	else
		return 200 * (inPoint(1) - inPoint(0) * inPoint(0));
}

/**
 * Second derivative of rosenbrock function.
 * @param inPoint
 * @param alpha1 Direction of first derivative.
 * @param alpha2 Direction of second derivative.
 * @return
 */
inline double diff2Rosenbrock(const ublas::vector<double>& inPoint,
		const unsigned alpha1, const unsigned alpha2) {
	if (alpha1 != alpha2) {
		return -400 * inPoint(0);
	} else if (alpha1 == 1)
		return 2 + 800 * inPoint(0) * inPoint(0) - 400 * (-inPoint(0)
				* inPoint(0) + inPoint(1));
	else
		return 200.;
}

/**
 * Analytic form of the 2d sin function function for testing.
 * @param inPoint Point at which to evaluate the function.
 * @return
 */
inline double sinCos(const ublas::vector<double>& inPoint) {
	return 10 * std::sin(inPoint(0)) + 10 * std::cos(inPoint(1));
}

/**
 * Analytic form of the first derivatives of the rosenbrock function.
 * @param inPoint Point at which to evaluate.
 * @param alpha Direction of derivative.
 * @return
 */
inline double diffSinCos(const ublas::vector<double>& inPoint,
		const unsigned alpha) {
	if (alpha == 1)
		return 10 * std::cos(inPoint(0));
	else
		return -10 * std::sin(inPoint(1));
}

/**
 * Second derivative of rosenbrock function.
 * @param inPoint
 * @param alpha1 Direction of first derivative.
 * @param alpha2 Direction of second derivative.
 * @return
 */
inline double diff2SinCos(const ublas::vector<double>& inPoint,
		const unsigned alpha1, const unsigned alpha2) {
	if (alpha1 != alpha2) {
		return 0;
	} else if (alpha1 == 1)
		return -10 * std::sin(inPoint(0));
	else
		return -10 * std::cos(inPoint(1));
}

#endif /* EXAMPLEDEFS_HPP_ */
