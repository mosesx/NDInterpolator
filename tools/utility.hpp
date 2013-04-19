/**
 * utility: Collection of useful global methods. By now mainly matrix/vector i/o.
 *
 *  Created on: \date Nov 11, 2010
 *      Author: \author mrehberg
 *      \todo deal with includes
 */

#ifndef UTILITY_H_
#define UTILITY_H_

// include all librarys needed for i/o
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

// boost libraries
#include "boost/numeric/ublas/matrix.hpp"
#include "boost/numeric/ublas/matrix_proxy.hpp"
#include "boost/numeric/ublas/vector.hpp"
#include "boost/numeric/ublas/vector_proxy.hpp"
#include "boost/numeric/ublas/io.hpp"

// NDInterpolator libraries
#include "../NDInterpolator.hpp"
#include "../MultiNDInterpolator.hpp"

namespace NDInterpolator {
  namespace ublas = boost::numeric::ublas;

  /**
   * Write a Matrix in tab separated file
   * @param outMatrix Matrix to write
   * @param filename Output file. Must specify a valid location.
   */
  inline void writeMatrix(const ublas::matrix<double> outMatrix,
		   const std::string filename) {

    std::ofstream outFile(filename.c_str());
    for (unsigned i = 0; i < outMatrix.size1(); i++) {
      for (unsigned j = 0; j < outMatrix.size2(); j++) {
	outFile << std::setprecision(16) << outMatrix(i, j) << "\t";
      }
      outFile << "\n";
    }
    outFile.close();
  }

  /**
   * Write a Vector as column in file or to std::cout
   * @param outVector Vector to write.
   * @param filename Output file. Must specify a valid location, if "" print to std::cout
   * @tparam Vector to print must have standard \c begin() and \c end() iterators.
   */
  template<class vectorType>
  inline void writeVector(const vectorType outVector,
		   const std::string filename = "") {

    std::ostream* outStream;
    std::ofstream fStream;

    if (filename!="") {
      fStream.open(filename.c_str());
      outStream = &fStream;
    }
    else {
      outStream= &std::cout;
    }

    for (auto it=outVector.begin(); it != outVector.end(); ++it) {
      (*outStream) << std::setprecision(16) << (*it) << std::endl;
    }
    (*outStream) << std::endl;

    if (filename != "") {
      fStream.close();
    }
  }

  /**
   * Read a Matrix from a file
   * Format:
   * - 1 ublas: (numOfRows,numOfColumns)((row1 comma separated),(row2 comma separated), ...,(rowN))
   * - 2 tab delimiter separated (default "\t")
   * @param filename Name of the file to read from.
   * @tparam format Unsigned int indicating the format of the file, see above.
   * @tparam delim Used in case of delimiter separated format, default="\t".
   * @return
   */
  template<unsigned format, char delim = '\t'>
    ublas::matrix<double> readMatrix(const std::string filename) {
    ublas::matrix<double, ublas::column_major> retMatrix;
    std::ifstream inFile(filename.c_str());
    switch (format) {
    case 1:
      inFile >> std::setprecision(17) >> retMatrix;
      inFile.close();
      break;
    case 2: {
      unsigned nCol = 0;
      unsigned nRow = 0;
      for (std::string row; std::getline(inFile, row, '\n');) {
	std::istringstream fLine(row);
	retMatrix.resize(nRow + 1, nCol, true);

	nCol = 0;
	for (std::string field; std::getline(fLine, field, delim);) {
	  if (nRow == 0)
	    retMatrix.resize(retMatrix.size1(), nCol + 1, true);
	  retMatrix(nRow, nCol) = std::strtod(field.c_str(), 0);
	  ++nCol;
	}
	++nRow;
      }
      inFile.close();
      break;
    }
    default:
      std::cout << std::setprecision(16)
		<< "utility::readMatrix: unknown format specifier" << std::endl;
    }
    return retMatrix;
  }

  /**
   * Evaluate many points with an univariate interpolator object. Basically used for profiling.
   * @param interp Reference to interpolator object.
   * @param evalGrid Set of evaluation points, each column is one point.
   * @return Output of the evaluation
   */
  template<class interpType>
    inline ublas::vector<double> evalMany(const interpType& interp,
					  const ublas::matrix<double, ublas::column_major>& evalGrid) {
    ublas::vector<double> res(evalGrid.size2());
    for (unsigned i = 0; i < evalGrid.size2(); i++)
      res(i) = interp.eval(ublas::column(evalGrid, i));

    return res;
  }

  /**
   * Evaluate the derivative in many points with an univariate interpolator object. Used for profiling.
   * @param interp Reference to interpolator object.
   * @param evalGrid Set of evaluation points, each column is one point.
   * @param alpha Direction of derivation, no input check!
   * @return Output of evaluation.
   */
  template<class interpType>
    inline ublas::vector<double> evalManyDiff(const interpType& interp,
					      const ublas::matrix<double, ublas::column_major>& evalGrid,
					      const unsigned alpha) {
    ublas::vector<double> res(evalGrid.size2());
    for (unsigned i = 0; i < evalGrid.size2(); i++)
      res(i) = interp.evalDiff(ublas::column(evalGrid, i), alpha);

    return res;
  }

  /**
   * Evaluate the second derivative in many points with an univariate interpolation object. Used
   * for profiling.
   * @param interp Reference to interpolator object
   * @param evalGrid Set of evaluation points, each column is one point.
   * @param alpha Direction of derivation, no input check!
   * @return Output of evaluation
   */
  template<class interpType>
    inline ublas::vector<double> evalManyDiff2(
					       const interpType& interp,
					       const ublas::matrix<double, ublas::column_major>& evalGrid,
					       const unsigned alpha) {
    ublas::vector<double> res(evalGrid.size2());
    for (unsigned i = 0; i < evalGrid.size2(); i++)
      res(i) = interp.evalDiff2(ublas::column(evalGrid, i), alpha);

    return res;
  }

  /**
   * Evaluate the mixed derivative in many points with an univariate interpolation object.
   * @param interp Reference to interpolator object.
   * @param evalGrid Set of evaluation points, each column is one point.
   * @param alpha1 Direction of first derivative, no input check!
   * @param alpha2 Direction of second derivative, no input check!
   * @return Output of evaluation
   */
  template<class interpType>
    inline ublas::vector<double> evalManyDiffMixed(
						   const interpType& interp,
						   const ublas::matrix<double, ublas::column_major>& evalGrid,
						   const unsigned alpha1, const unsigned alpha2) {
    ublas::vector<double> res(evalGrid.size2());
    for (unsigned i = 0; i < evalGrid.size2(); i++)
      res(i) = interp.evalDiffMixed(ublas::column(evalGrid, i), alpha1,
				    alpha2);

    return res;
  }

  /**
   * Evaluate many points with multivariate interpolator object. Used for profiling.
   * @param interp Reference to interpolator object.
   * @param evalGrid Set of evaluation points, each column is one point.
   * @return Output matrix.
   */
  template <class interpType>
    inline ublas::matrix<double, ublas::column_major> evalMany(
							       interpType& interp,
							       const ublas::matrix<double, ublas::column_major>& evalGrid) {
    ublas::matrix<double, ublas::column_major> res(interp.getOutDim(),
						   evalGrid.size2());
    double ret;
    for (unsigned i = 0; i < evalGrid.size2(); i++){
      noalias(ublas::column(res, i)) = interp.eval(ublas::column(evalGrid, i));
    }
    return res;
  }


  /**
   * Evaluate many points with multivariate interpolator object.
   * @param interp Reference to multiinterpolator object
   * @param evalGrid Set of evaluation points, each column is one point
   * @param alpha Direction of derivative.
   * @return Output matrix
   */
  template <class interpType>
    inline ublas::matrix<double, ublas::column_major> evalManyDiff(
								   interpType& interp,
								   const ublas::matrix<double, ublas::column_major>& evalGrid,
								   unsigned alpha) {
    ublas::matrix<double, ublas::column_major> res(interp.getOutDim(),
						   evalGrid.size2());
    for (unsigned i = 0; i < evalGrid.size2(); i++)
      noalias(ublas::column(res, i)) = interp.evalDiff(
						       ublas::column(evalGrid, i), alpha);

    return res;
  }

  /**
   * Evaluate many points with multivariate interpolator object, second derivative
   * @param interp Reference to multiinterpolator object
   * @param evalGrid Set of evaluation points, each column is one point.
   * @param alpha direction of derivative.
   * @return Output matrix.
   */
  template <class interpType>
    inline ublas::matrix<double, ublas::column_major> evalManyDiff2(
								    interpType& interp,
								    const ublas::matrix<double, ublas::column_major>& evalGrid,
								    unsigned alpha) {
    ublas::matrix<double, ublas::column_major> res(interp.getOutDim(),
						   evalGrid.size2());
    for (unsigned i = 0; i < evalGrid.size2(); i++)
      noalias(ublas::column(res, i)) = interp.evalDiff2(
							ublas::column(evalGrid, i), alpha);

    return res;
  }

  /**
   * Evaluate many points with multivariate interpolator object, mixed derivative.
   * @param interp Reference to multiinterpolator object.
   * @param evalGrid Set of evaluation points, each column is one point.
   * @param alpha1 Direction of first derivative.
   * @param alpha2 Direction of second derivative.
   * @return Output matrix
   */
  template <class interpType>
    inline ublas::matrix<double, ublas::column_major> evalManyDiffMixed(
									interpType& interp,
									const ublas::matrix<double, ublas::column_major>& evalGrid,
									unsigned alpha1, unsigned alpha2) {
    ublas::matrix<double, ublas::column_major> res(interp.getOutDim(),
						   evalGrid.size2());
    for (unsigned i = 0; i < evalGrid.size2(); i++)
      noalias(ublas::column(res, i)) = interp.evalDiffMixed(
							    ublas::column(evalGrid, i), alpha1, alpha2);

    return res;
  }

  /**
   * Computes an approximation for the fill distance of the set G. The fill distance is defined
   * as
   * \f[ \sup_{x\in \Omega} \min_{j=1,...,n} \| x-x_j \|_2.\f]
   * Since we do not know \f$ \Omega \f$ we use
   * \f[ \max_{i=1,...,n} \min_{j=1,...,n; i \neq j} \|x_i-x_j\|_2. \f]
   * @param G Set of points, every row corresponds to a dimension.
   * @tparam matrixType Type of G.
   * @return Estimate of fill distance.
   */
  template<class matrixType = ublas::matrix<double, ublas::column_major> >
    inline double fillDistance(const matrixType& G) {
    double ret = 0, min;
    const size_t n = G.size2();
    ublas::vector<double> mins(n);
    for (size_t i = 0; i < n; ++i) {
      mins(i) = std::numeric_limits<double>::max();
      for (size_t j = 0; j < n; ++j) {
	if (i != j) {
	  min = ublas::norm_2(ublas::column(G, i) - ublas::column(G, j));
	  if (mins(i) > min) {
	    mins(i) = min;
	  }
	}
      }
    }
    return *std::max_element(mins.begin(), mins.end());
  }

  /**
   * Evaluates the interpolator interp on the grid and compares the result to the data given. Comparison
   * is made in 1,2 and inf-norm.
   * @param interp Reference to MultiNDInterpolator object.
   * @param grid Set of evaluation points, each column is one point
   * @param data Data, each column represents the value of the underlying function at a point from grid.
   * @return Matrix of size outDim times 3, giving the 1,2 and inf-norm respectively in each column
   * for each output dimension.
   */
  template <class interpType>
    inline ublas::matrix<double, ublas::column_major> compareOnSet(
								   interpType& interp,
								   const ublas::matrix<double, ublas::column_major>& grid,
								   const ublas::matrix<double, ublas::column_major>& data) {

    unsigned numOfPoints = grid.size2();
    unsigned outDim = data.size1();
    if (numOfPoints != data.size2())
      throw std::length_error(
			      "NDInterpolator::compareOnSet: Number of columns in data does not match numOfPoints");
    ublas::matrix<double, ublas::column_major> diffRes = data - evalMany<interpType>(
										     interp, grid);
    ublas::matrix<double, ublas::column_major> ret(outDim, 3);

    for (unsigned i = 0; i < outDim; ++i) {
      ret(i, 0) = norm_1(ublas::row(diffRes, i));
      ret(i, 1) = norm_2(ublas::row(diffRes, i));
      ret(i, 2) = norm_inf(ublas::row(diffRes, i));
    }

    return ret;
  }

#ifndef _WITHOUT_SERIALIZATION_
  /**
   * Save the status of an RBFInterpolator into the text file filename
   * @param interpRef Reference to interpolator that should be saved.
   * @param filename Name of the file to write to.
   * @tparam interp Type of interpolator object the function is called with, e.g.
   * NDInterpolator::RBFInterpolator<NDInterpolator::GaussianRBF<double> >.
   */
  template<class interp>
    inline void saveRBFInterp(const interp& interpRef, const std::string filename) {
    std::ofstream ofs(filename.c_str());
    boost::archive::text_oarchive outArchive(ofs);
    outArchive << interpRef;
    ofs.close();
  }

  /**
   * Load the status of an RBFInterpolator from the textfile filename. The returned interpolator
   * object is ready to be used.
   * @param filename Name of the file to read.
   * @return Interpolation object of class interp specified in the template argument.
   * @tparam interp Type of interpolator object the function is called with, e.g.
   * NDInterpolator::RBFInterpolator<NDInterpolator::GaussianRBF<double> >.
   */
  template<class interp>
    inline interp loadRBFInterp(const std::string filename) {
    std::ifstream ifs(filename.c_str());
    boost::archive::text_iarchive inArchive(ifs);
    interp ret;
    inArchive >> ret;
    ifs.close();
    return ret;
  }
#endif

  /**
   * Q & D vector indirect for standard vectors. Extracts all entries given by ind from fullVec into
   * the result. Both vectors have to implement the [.] operator for accessing entries.
   * @param fullVec Full vector.
   * @param ind Vector of indices to extract.
   * @return std::vector of entries specified by ind.
   * @tparam storeType Type of the full vector, e.g. std::vector<double>.
   * @tparam indexType Type of the vector of indices, e.g. std::vector<size_t>.
   */
  template<class storeType, class indexType>
    inline std::vector<storeType> stdVectorIndirect(const std::vector<storeType>& fullVec, 
						    const indexType& ind) {

    std::vector<storeType> ret(ind.size());
    for (unsigned i = 0; i < ind.size(); i++) {
      ret[i] = fullVec[ind[i]];
    }
    return ret;
  }

  /**
   * Generates a rectangular grid with numOfPoints(i) points in dimension i between leftCorner(i) and
   * rightCorner(i)
   * @param leftCorner Vector of length inDim, specifying the lower, left start point of the grid.
   * @param rightCorner Vector of length inDim, specifying the upper, right end point of the grid.
   * @param numOfPoints Vector of length inDim, specifying the number of points for each dimension.
   * @return Grid of size inDim times cumprod(numOfPoints).
   */
  ublas::matrix<double, ublas::column_major> genRecGrid(const ublas::vector<double> leftCorner, 
							const ublas::vector<double> rightCorner,
							const ublas::vector<unsigned> numOfPoints) {

    unsigned inDim = static_cast<unsigned>(leftCorner.size());
    if (rightCorner.size() != inDim)
      throw std::length_error("Length of rightCorner does not match length of leftCorner.");
    if (numOfPoints.size() != inDim)
      throw std::length_error("Length of numOfPoints does not match length of leftCorner.");

    // compute overall num of Points and diff between two points in each dimension.
    unsigned N = 1;
    ublas::vector<unsigned> k(inDim);
    ublas::vector<double> alpha(inDim);
    for (unsigned i = 0; i < inDim; i++) {
      N *= numOfPoints(i);
      k(i) = 0;
      alpha(i) = (rightCorner(i) - leftCorner(i)) / (numOfPoints(i) - 1);
    }

    /*
     * real dirty algorithm. Using a bool to check for overflow of the counting variable
     * @todo recursive algorithm
     */
    bool overFlow = false;
    ublas::matrix<double, ublas::column_major> grid(inDim, N);
    for (unsigned i = 0; i < N; i++) {
      for (unsigned j = 0; j < inDim; j++) {
	grid(j, i) = leftCorner(j) + alpha(j) * k(j);
      }
      for (unsigned l = inDim; l > 0; l--) {
	// last dimension was overflowing
	if (overFlow) {
	  if (k(l - 1) < numOfPoints(l - 1) - 1) {
	    k(l - 1)++;
	    overFlow = false;
	  } else {
	    k(l - 1) = 0;
	  }
	} else {
	  if (l == inDim) {
	    if (k(l - 1) < numOfPoints(l - 1) - 1) {
	      k(l - 1)++;
	    } else {
	      k(l - 1) = 0;
	      overFlow = true;
	    }
	  }
	}
      }
    }
    return grid;
  }

  /**
   * Extracts given columns from a matrix. Index is 0 based
   * @param A Input matrix.
   * @param B Index vector indicating the columns.
   * @return Extracted columns
   * @tparam matrix Type of matrix, must have \c size1(), \c size2() functions and \c (r,c) operator
   * for element access. 
   * @tparam ind Type of index vector, must have \c [].
   */
  template<class matrix, class indVec>
  inline matrix cpColumns(const matrix& A, const indVec& ind) {
    
    matrix ret(A.size1(), ind.size());
    for (unsigned i=0; i < ind.size(); ++i){
      for (unsigned k=0; k < A.size1(); ++k) {
	ret(k,i)=A(k,ind[i]);
      }
    }
    return ret;
  }


  /**
   * Compares floating point numbers and returns true if they are close together. Close is absolutly
   * and relatively given by tol.
   * @param a
   * @param b
   * @param tol
   * @return true if a==b nearly holds
   */
  template<class inputType>
    inline bool compFloat(const inputType a, const inputType b,
			  const inputType tol = 1e-14) {
    return std::abs(a - b) <= tol * std::max(1.0,std::max(std::abs(a), std::abs(b)));
  }

  /**
   * \brief Kronecker delta: \f[ \delta_{ij} = 1, i = j \ \delta_{ij} = 0, i\not= j. \f]
   */
  template<class IX>
    inline IX kronecker(IX i, IX j) {
    return ((i == j) ? 1 : 0);
  }

  /**
   * \brief Some elementary operations
   */
  template<class T>
    class BasicMultiplication {
  public:
    static inline T apply(const T& a, const T& b) {
      return a * b;
    }

    static inline T apply(T& a, const T& b) {
      return a *= b;
    }

    static inline T apply(const T& b, T& a) {
      return a *= b;
    }

  };

  template<class T>
    class BasicAddition {
  public:
    static inline T apply(const T& a, const T& b) {
      return a + b;
    }

    static inline T apply(T& a, const T& b) {
      return a += b;
    }

    static inline T apply(const T& b, T& a) {
      return a += b;
    }
  };

} //end namespace 
#endif /* UTILITY_H_ */
