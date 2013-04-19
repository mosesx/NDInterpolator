#ifndef USEFUL_EXTENSIONS_H
#define USEFUL_EXTENSIONS_H

#include <cassert>
#include "../../tools/utility.hpp"

namespace NDInterpolator {


  /**
   * \brief <I>Concatenates</I> two matrices \f$ A \in \mathbf{R}^{m \times n}\f$ and \f$ B \in \mathbf{R}^{m \times r} \f$ such that the result describes a new matrix \f$ C = [A|B] \in \mathbf{R}^{m \times (n+r)}\f$.
   *
   * \tparam BM Boost-compliant matrix
   */
  template<class BM>
  inline void concatenate(BM& C, const BM& A, const BM& B) {
    if (A.size1() != B.size1()) {
      std::cout << A.size1() << "\t" << B.size1() << std::endl;
    }
    assert(A.size1() == B.size1());
    assert((C.size1() == A.size1()) && (C.size2() == (A.size2() + B.size2())));

    size_t ccol = A.size2() + B.size2();

    for (size_t i = 0; i < A.size1(); ++i) {
      //fill first part of C with A
      for (size_t j = 0; j < A.size2(); ++j) {
	C(i, j) = A(i, j);
      }
      //fill second part of C with B
      for (size_t j = A.size2(); j < ccol; ++j) {
	C(i, j) = B(i, j - A.size2());
      }
    }
  }

  /**
   * \brief Concatenates two matrices in the same fashion as in the preceding function. However it uses as temporary object which is overwritten with the concatenation.
   */
  template<class BM>
  inline BM concatenate(const BM& A, const BM& B) {
    assert(A.size1() == B.size1());

    size_t ccol = A.size2() + B.size2();

    BM C(A.size1(), ccol);

    for (size_t i = 0; i < A.size1(); ++i) {
      //fill first part of C with A
      for (size_t j = 0; j < A.size2(); ++j) {
	C(i, j) = A(i, j);
      }
      //fill second part of C with B
      for (size_t j = A.size2(); j < ccol; ++j) {
	C(i, j) = B(i, j - A.size2());
      }
    }

    return C;
  }

  /**
   * \brief forms identity matrix, stored as <B> full </B> Boost-compliant matrix. This should only be used when you intend to use it with Lapack.
   */
  template<class BM>
  inline void identity_full(BM& I) {
    assert(I.size1() == I.size2());
    for (size_t i = 0; i < I.size1(); ++i) {
      for (size_t j = 0; j < I.size2(); ++j) {
	I(i, j) = (double) kronecker<size_t> (i, j);
      }
    }
  }

  /**
   * \brief forms identity matrix, stored as <B> full </B> Boost-compliant matrix. This time, however, a temporary is used.
   * \param n order of the identity matrix
   *
   *  NOTE: template argument must be specified explicitly here,e.g.:
   * \code
   std::cout << identity_full<boost::numeric::ublas::matrix<double,boost::numeric::ublas::column_major> >(4) << std::endl;
   * \endcode
   */
  template<class BM>
  inline BM identity_full(size_t n) {
    BM Id(n, n);
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
	Id(i, j) = (double) kronecker<size_t> (i, j);
      }
    }
    return Id;
  }

  /**
   * \brief A compile-time TMP norm selector for Boost vectors and matrices
   * \tparam C specifies the norm you wanna use: '1', '2' and 'i'( or 'I') stand for the \f$ l_1, l_2\f$ and \f$ l_{\infty}\f$ norms, respectively.
   * \tparam RT real return type (default: double). Mind: Even when you use <I>complex numbers</I>, the corresponding norm is from \f$ \mathbf{R}\f$!
   */
  template<char C, class RT = double> class Norm;

  //!partial specialisations
  template<class RT>
  class Norm<'1', RT> {
  public:
    typedef RT return_type;

    template<class V>
    static inline RT norm(const V& v) {
      return boost::numeric::ublas::norm_1(v);
    }
  };

  template<class RT>
  class Norm<'2', RT> {
  public:
    typedef RT return_type;

    template<class V>
    static inline RT norm(const V& v) {
      return boost::numeric::ublas::norm_2(v);
    }
  };

  template<class RT>
  class Norm<'i', RT> {
  public:
    typedef RT return_type;

    template<class V>
    static inline RT norm(const V& v) {
      return boost::numeric::ublas::norm_inf(v);
    }
  };

  template<class RT>
  class Norm<'I', RT> {
  public:
    typedef RT return_type;

    template<class V>
    static inline RT norm(const V& v) {
      return boost::numeric::ublas::norm_inf(v);
    }
  };

} //end namespace 

#endif
