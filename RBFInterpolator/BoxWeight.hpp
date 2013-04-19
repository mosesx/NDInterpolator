/**
 * boxWeight.h
 *
 *  Created on: \date Nov 15, 2010
 *      Author: \author mrehberg
 */

#ifndef BOXWEIGHT_H_
#define BOXWEIGHT_H_

/** \brief Weight function for partition of unity.
 *
 * Radialised 1-d Polynomial that is 0 on border of domain and 1 in the middle. Twice continously
 * differentiable.  based on Tobor, I.; Reuter, P. & Schlick, C. Efficient Reconstruction of Large
 * Scattered Geometric Datasets using the Partition of Unity and Radial Basis Functions
 */
namespace NDInterpolator {

class BoxWeight {

public:
	double eval(const ublas::vector<double>& inPoint, const ublas::vector<
			double>& edgesL, const ublas::vector<double>& edgesR) const;
	double
	evalDiff(const ublas::vector<double>& inPoint,
			const ublas::vector<double>& edgesL,
			const ublas::vector<double>& edgesR, const unsigned alpha) const;

	double
	evalDiff2(const ublas::vector<double>& inPoint,
			const ublas::vector<double>& edgesL,
			const ublas::vector<double>& edgesR, const unsigned alpha) const;

	double evalDiffMixed(const ublas::vector<double>& inPoint,
			const ublas::vector<double>& edgesL,
			const ublas::vector<double>& edgesR, const unsigned alpha1,
			const unsigned alpha2) const;

private:
	// serialization.
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
	}

};

inline double BoxWeight::eval(const ublas::vector<double>& inPoint,
		const ublas::vector<double>& edgesL,
		const ublas::vector<double>& edgesR) const {

	double ret = 1.;
	for (unsigned i = 0; i < inPoint.size(); i++) {
		ret *= ((inPoint(i) - edgesL(i)) * (edgesR(i) - inPoint(i))) / (.25
				* ((edgesR(i) - edgesL(i)) * (edgesR(i) - edgesL(i))));
	}
	// mathematica simplification of Poly(1-ret) in Horner form
	// poly: -6x^5+15x^4-10x^3+1
	return ret * ret * ret * (10 + ret * (-15 + 6 * ret));
}

inline double BoxWeight::evalDiff(const ublas::vector<double>& inPoint,
		const ublas::vector<double>& edgesL,
		const ublas::vector<double>& edgesR, const unsigned alpha) const {

	// initialize with factor from derivation
	const double dervFac = -(edgesR(alpha - 1) + edgesL(alpha - 1) - 2
			* inPoint(alpha - 1)) / ((inPoint(alpha - 1) - edgesL(alpha - 1))
			* (edgesR(alpha - 1) - inPoint(alpha - 1)));

	double ret = 1.;
	for (unsigned i = 0; i < inPoint.size(); i++) {
		ret *= ((inPoint(i) - edgesL(i)) * (edgesR(i) - inPoint(i))) / (.25
				* ((edgesR(i) - edgesL(i)) * (edgesR(i) - edgesL(i))));
	}

	// mathematica horner scheme
	return ret * ret * ret * (-30 * dervFac + ret * (60 * dervFac - 30 * ret
			* dervFac));
}

inline double BoxWeight::evalDiff2(const ublas::vector<double>& inPoint,
		const ublas::vector<double>& edgesL,
		const ublas::vector<double>& edgesR, const unsigned alpha) const {

	// common denominator for derivative factors
	const double denom = ((inPoint(alpha - 1) - edgesL(alpha - 1)) * (edgesR(
			alpha - 1) - inPoint(alpha - 1)));
	// initialize factor for first order derivative
	const double dervFac = -(edgesR(alpha - 1) + edgesL(alpha - 1) - 2
			* inPoint(alpha - 1)) / denom;
	// initialize factor for second order derivative
	const double dervFac2 = 2 / denom;

	double ret = 1.0;
	for (unsigned i = 0; i < inPoint.size(); i++) {
		ret *= ((inPoint(i) - edgesL(i)) * (edgesR(i) - inPoint(i))) / (.25
				* ((edgesR(i) - edgesL(i)) * (edgesR(i) - edgesL(i))));
	}

	// mathematica Horner Form
	return ret * ret * ret * (60 * dervFac * dervFac - 30 * dervFac2 + ret
			* (-180 * dervFac * dervFac + 60 * dervFac2 + (120 * dervFac
					* dervFac - 30 * dervFac2) * ret));

}

double BoxWeight::evalDiffMixed(const ublas::vector<double>& inPoint,
		const ublas::vector<double>& edgesL,
		const ublas::vector<double>& edgesR, const unsigned alpha1,
		const unsigned alpha2) const {

	const double dervFacA1 = (edgesR(alpha1 - 1) + edgesL(alpha1 - 1) - 2
			* inPoint(alpha1 - 1))
			/ ((inPoint(alpha1 - 1) - edgesL(alpha1 - 1)) * (edgesR(alpha1 - 1)
					- inPoint(alpha1 - 1)));

	const double dervFacA2 = (edgesR(alpha2 - 1) + edgesL(alpha2 - 1) - 2
			* inPoint(alpha2 - 1))
			/ ((inPoint(alpha2 - 1) - edgesL(alpha2 - 1)) * (edgesR(alpha2 - 1)
					- inPoint(alpha2 - 1)));

	double ret = 1.0;
	for (unsigned i = 0; i < inPoint.size(); i++) {
		ret *= ((inPoint(i) - edgesL(i)) * (edgesR(i) - inPoint(i))) / (.25
				* ((edgesR(i) - edgesL(i)) * (edgesR(i) - edgesL(i))));
	}

	// mathematica Horner Form
	return dervFacA1 * dervFacA2 * ret * ret * ret * (90 + ret * (-240 + 150
			* ret));
}
}

#endif /* BOXWEIGHT_H_ */
