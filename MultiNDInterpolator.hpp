/**
 * MultiNDInterpolator.h
 *
 *  Created on: \date Nov 25, 2010
 *      Author: \author mrehberg
 */

#ifndef MULTINDINTERPOLATOR_H_
#define MULTINDINTERPOLATOR_H_

// std includes
#include <exception>
// ublas includes
#include "boost/numeric/ublas/matrix.hpp"
#include "boost/numeric/ublas/matrix_proxy.hpp"
#include "boost/numeric/ublas/vector.hpp"

// NDInterpolator includes
#include "./tools/archive.hpp"

/** \brief Interface for n-dimensional interpolator object
 */
namespace NDInterpolator {
namespace ublas = boost::numeric::ublas;

class MultiNDInterpolator {
public:

	virtual ~MultiNDInterpolator() {
	}

	/**
	 * Retrieve the dimension of the underlying domain, i.e. dimension of centers.
	 * @return
	 */
	unsigned int getInDim() const {
		return inDim;
	}

	/**
	 * Retrieve the dimension of the output domain, i.e. dimension of data.
	 * @return
	 */
	unsigned getOutDim() const {
		return outDim;
	}

	/**
	 * Retrieve the number of points used to build the interpolation object.
	 * @return
	 */
	unsigned int getNumOfPoints() const {
		return numOfPoints;
	}

protected:
	unsigned int numOfPoints;
	unsigned int inDim;
	unsigned outDim;

#ifndef _WITHOUT_SERIALIZATION_
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
		ar & numOfPoints;
		ar & inDim;
		ar & outDim;
	}
#endif
};

}

#endif /* MULTINDINTERPOLATOR_H_ */
