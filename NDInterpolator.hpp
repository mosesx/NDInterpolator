/**
 * NDInterpolator.h
 *
 *  Created on: Oct 20, 2010
 *      Author: mrehberg
 */

/** \mainpage NDInterpolator library
 * \author Marcel Rehberg
 * \author Marc Fein
 * \version 0.8.8 Changelog.txt
 * \todo 3rd derivative in General/InverseMultiquadricRBF.
 *
 * \section Abstract
 * The library provides several classes for N-dimensional interpolation, especially on scattered
 * (read nonregular grids). All of the classes are based on radial basis functions. A good review
 * is
 * - M. D. Buhmann (2000). Radial basis functions. Acta Numerica 2000, 9, pp 1-38.
 *
 * For hermite interpolation with radial basis functions see
 * - Zong-min Wu (1992). Hermite-Birkhoff Interpolation of Scattered Data by Radial Basis Functions.
 * Approximation Theory and its Applications, Springer
 * Netherlands, 1992, 8, pp 1-10.
 *
 * The partition of unity approach is partially based on
 * - Wendland, H. Fast Evaluation of Radial Basis Functions: Methods Based on Partition of Unity.
 * Approximation Theory X: Wavelets, Splines, and Applications,
 * Vanderbilt University Press, 2002, 473-483.
 *
 * and
 * - I. Tobor, P. Reuter, C. Schlick
 * Efficient Reconstruction of Large Scattered Geometric Datasets using the Partition of Unity
 * and Radial Basis Functions (2003).
 *
 * A comprehensive textbook on the matter is
 * - Scattered Data Approximation by Holger Wendland published in Cambridge University Press (2005).
 *
 * \section Interface
 * The library provides two main interfaces for interpolation:
 * \li NDInterpolator::NDInterpolator : Interpolation of functions \f$ f: \mathrm{R}^n \rightarrow
 * \mathrm{R} \f$
 * \li NDInterpolator::MultiNDInterpolator : Interpolation of functions \f$ f: \mathrm{R}^n
 * \rightarrow \mathrm{R}^m \f$.
 *
 * Both provide specializations for Lagrange and Hermite interpolation, an overview is given in
 * the following list:
 * - NDInterpolator::RBFInterpolator: N-dimensional input, 1-dimensional output, Lagrange interpolation
 * - NDInterpolator::RBFInterpolatorH: N-dimensional input, 1-dimensional output, Hermite interpolation
 * - NDInterpolator::RBFInterpolatorPU: N-dimensional input, 1-dimensional output, Lagrange interpolation,
 * partition of unity method.
 * - NDInterpolator::RBFInterpolatorPUH: N-dimensional input, 1-dimensional output, Hermite interpolation,
 * partition of unity method.
 * - NDInterpolator::MultiRBFInterpolator: N-dimensional input, M-dimensional output, Lagrange interpolation
 * - NDInterpolator::MultiRBFInterpolatorH: N-dimensional input, M-dimensional output, Hermite interpolation
 * - NDInterpolator::MultiRBFInterpolatorPU: N-dimensional input, M-dimensional output, Lagrange interpolation,
 * partition of unity method.
 * - NDInterpolator::MultiRBFInterpolatorPUH: N-dimensional input, M-dimensional output, Hermite interpolation,
 * partition of unity method.
 *
 * All single dimension output RBF interpolators or multi dimension output RBFInterpolators can
 * be brought into scope by including RBFInterpolators.h or MultiRBFInterpolators.h respectively.
 *
 *\section hallo 3rd Party Code
 * The code uses <a href="http://www.netlib.org/lapack/">LAPACK</a> for solving the linear systems,
 * <a href="http://www.boost.org">Boost</a>
 * <a href="http://www.boost.org/doc/libs/1_45_0/libs/numeric/ublas/doc/index.htmuBlas">uBlas 1.45
 * </a> for linear algebra,
 * <a href="https://svn.boost.org/svn/boost/sandbox/numeric_bindings/">numeric_bindings</a> for
 * accessing lapack with ublas types,
 * <a href="http://www.boost.org/doc/libs/1_45_0/libs/multi_array/doc/index.html">boost::multi_array</a>
 * for storage and <a href="http://www.boost.org/doc/libs/1_45_0/libs/serialization/doc/index.html">boost::serialization</a>
 * for serialization and saving of interpolator objects. The support for serialization can be turned
 * of.
 *
 * \section Installation
 * The library is a header only, however it relies on LAPACK and if wanted by the user on
 * the boost::serialization library.
 *
 * \subsection Boost
 * The <a href="http://www.boost.org">boost c++ libraries</a> aim at expanding the c++ standard
 * library. To install boost libraries download the appropriate version from the website and
 * extract the archive somewhere on your computer, say BOOST_ROOT. Details:
 *
 * - boost::ublas: The ublas library consists solely of header files. To be able to compile you need to
 * add the following include path to your project (i.e. make file, compiler options, ...)
 * \verbatim -I"/path/to/BOOST_ROOT" \endverbatim
 * - boost::multi_array: Header only library. The boost::ublas include is enough to get it to
 * work.
 * - boost::serialization: Binary library, if you do not want serialization support, e.g. be able
 * to save the status of an interpolator object you can skip this step.
 * General installation instructions for boost binary libraries can be found starting
 * <a href="http://www.boost.org/doc/libs/1_45_0/more/getting_started/index.html">here</a>
 * and more detailed for *nix systems
 * <a href="http://www.boost.org/doc/libs/1_45_0/more/getting_started/unix-variants.html#prepare-to-use-a-boost-library-binary">here</a>.
 * In short (*nix):
 * 	- Install \em bjam, most easily through your systems package managment.
 * 	- Open BOOST_ROOT in a command line and try
 * \verbatim ./bootstrap.sh --help \endverbatim
 * to see some options for configuration (especially regarding installation directories).
 * To configure the boost::serialization library use
 * \verbatim ./bootstrap.sh --with-libraries=serialization \endverbatim
 * Compilation is started with
 * \verbatim ./bjam \endverbatim
 * The libraries can be found in
 * \verbatim BOOST_ROOT/stage/lib \endverbatim which is the path you have to add to your compiler
 * options.
 *
 * \subsection Numeric_bindings
 * Can be <a href="https://svn.boost.org/svn/boost/sandbox/numeric_bindings/">browsed</a> and
 * downloaded via svn. The svn command would look like
 * \verbatim svn co http://svn.boost.org/svn/boost/sandbox/numeric_bindings folderName \endverbatim
 * to check out the current version into the folder \em folderName. Again you have to adjust your
 * include path with \verbatim -I"/path/toBindings/folderName"\endverbatim .
 *
 * \subsection LAPACK
 * It is assumed that \em lapack.h is in the global include path and for execution of the
 * example of course the library itself (typically liblapack3.x.x.so) should be in the global
 * library path or in LD_LIBRARY_PATH, too. Typically it is sufficient to install the distributions
 * packaged version of LAPACK to fullfill the requirements.
 *
 * \subsection Serialization
 * If you do not want serialization support you have to define \verbatim _WITHOUT_SERIALIZATION_ \endverbatim
 * for example by adding \verbatim -D _WITHOUT_SERIALIZATION_ \endverbatim to your g++ compiler flags.
 *
 * \section Example
 * The file \em example.cpp in the example folder holds an example for basic usage. The file
 * \em optShape.cpp demonstrates shape optimization and serialization, i.e. write the status of the
 * interpolator to a file. In the prepared makefile you have to adjust the paths and if you
 * want to compile with serialization support (change WO_SERIAL to TRUE). If you disable serialization
 * you want be able to compile and run \em optShape, note however that shape parameter optimization
 * does not depend on serialization, this is just due to the structure of the example.
 *
 * If you have enabled serialization support and the boost::serialization library is not in one of
 * the classic paths make sure you expand your \em LD_LIBRARY_PATH by issuing
 * \verbatim export LD_LIBRARY_PATH=/path/to/serialization/library \endverbatim
 */

#ifndef NDINTERPOLATOR_H_
#define NDINTERPOLATOR_H_

// std includes
#include <exception>
// boost::ublas includes
#include "boost/numeric/ublas/matrix.hpp"
#include "boost/numeric/ublas/matrix_proxy.hpp"
#include "boost/numeric/ublas/vector.hpp"
//boost::serialization includes
#include "tools/archive.hpp"

/** \brief Namespace for all library classes. */
namespace NDInterpolator {
namespace ublas = boost::numeric::ublas;

class NDInterpolator {
public:

	virtual ~NDInterpolator() {
	}

	/**
	 * Retrieve the dimension of the underlying domain.
	 * @return
	 */
	unsigned int getInDim() const {
		return inDim;
	}

	/**
	 * Compability function returning the output dimension 1.
	 * @return
	 */
	unsigned getOutDim() const {
		return 1;
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

private:
#ifndef _WITHOUT_SERIALIZATION_
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
		ar & numOfPoints;
		ar & inDim;
	}
#endif
};

}

#endif /* NDINTERPOLATOR_H_ */
