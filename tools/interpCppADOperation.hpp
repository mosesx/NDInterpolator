/** \file interpCppADOperation.hpp
 *
 * Header file implementing all functions that enable the MultiNDInterpolator to be used in
 * a CPPAD tape. That is the interpolator can be used like any other CPPAD function including
 * evaluation of derivatives up to second order.
 *
 * \section dummy Usage
 *
 * See CppADExample.cpp
 *
 * It is possible to use several interpolators. They have to be collected as
 * \code CppAD::vector<interpType> interpolators; \endcode
 * ATTENTION: The vector "interpolators" is created in this header file. After creating interpolator
 * objects elsewhere fill it with:
 * \code interpolators.push_back(interpRef); \endcode
 * where interpRef is a pointer to the interpolator object. If you use several different
 * interpolator objects make sure you remember the order in the vector since the index is used by
 * CPPAD to identify the right interpolator for an operation.
 *
 * The type of the interpolator is set in a different header file \code interpCppADSettings.hpp \endcode.
 * Create this file localy in the directory where you plan to use \code interpCppADOperation \endcode.
 * This could look like this:
 * \code 
 * #ifndef _INTERPCPPADSETTINGS_
 * #define _INTERPCPPADSETTINGS_
 * 
 * typedef NDInterpolator::MultiRBFInterpolatorPUH<
 *    NDInterpolator::GeneralMultiquadricRBF<-1, double>, 2, 1> interpType;
 *
 * #endif
 *
 * By including this file where ever you need the interpolator type you make sure you always use the 
 * right type.
 *
 * You might also add a macro definition for the vector type to be used.
 * \code #define _INTERPCPPADVECTORTYPE_ CppAD::vector \endcode
 * Otherwise CppAD::vector is used as default type.
 *
 * Also make sure you add \code -I./ \endcode to your includepath, so that this file finds the
 * interpCppADSettings.hpp.
 *
 * If X is some input AD-variable and Y the corresponding output the interpolators can be used like
 * \code interp(id,X,Y); \endcode
 * where "id" is the interpolator you want to use at that point. After calling "interp" Y will
 * contain the result of the evaluation.
 *
 *  Created on: \date Jun 3, 2011
 *      Author: \author mrehberg
 */

#ifndef INTERPCPPADOPERATION_H_
#define INTERPCPPADOPERATION_H_

#include <vector>
#include "cppad/cppad.hpp"
#include "boost/numeric/ublas/symmetric.hpp"
#include "../MultiRBFInterpolators.hpp"
#include "interpCppADSettings.hpp"

namespace ublas = boost::numeric::ublas;

namespace {



  /**
   * This vector will contain pointers to the interpolators used in user code.
   */

  CppAD::vector<interpType*> interpolators;

  /**
   * Forward evaluation of the interpolator up to order 2.
   * @param id Which interpolator,i.e. interpolators[id] is used.
   * @param k Order of evaluation.
   * @param n Domain space dimension.
   * @param m Range space dimension.
   * @param vx Which input variables are truly variables in the CPPAD sense.
   * @param vy [RETURN] Which output variables are truly variables in the CPPAD sense.
   * @param tx Taylor coefficients for \f$ X(t) \f$..
   * @param ty [RETURN] Taylor Coefficients for \f$ Y(t)=F(X(t)) \f$.
   * @return True if succeeded, generally false for k>2
   */
  template<class base>
  bool interpForward(std::size_t id, std::size_t k, std::size_t n, std::size_t m,
		     const CppAD::vector<bool>& vx, CppAD::vector<bool>& vy,
		     const CppAD::vector<base>& tx, CppAD::vector<base>& ty) {

    // copy input to ublas vector for evaluating interpolator
    ublas::vector<base> inPoint(n);

    for (unsigned j = 0; j < n; j++)
      inPoint(j) = tx[j * (k + 1)];

    // call to afun --> vx.size>0 and k==0
    if (vx.size() > 0) {
      ublas::vector<base> outPoint(m);
      outPoint = interpolators[id]->eval(inPoint);

      //interpolators[id]->eval(inPoint,outPoint);
      // in rbf-interpolation all output is variable since each depends on all inputs
      for (unsigned j = 0; j < m; j++) {
	vy[j] = true;
	ty[j * (k + 1) + k] = outPoint(j);
      }
      return true;
    }

    if (k == 0) {
      ublas::vector<base> outPoint(m);
      outPoint = interpolators[id]->eval(inPoint);
      // in rbf-interpolation all output is variable since each depends on all inputs
      for (unsigned j = 0; j < m; j++) {
	ty[j * (k + 1) + k] = outPoint(j);
      }
      return true;
    }

    // first order taylor coeffs
    if (k == 1) {
      base sum;
      ublas::matrix<base, ublas::column_major> jac =
	interpolators[id]->evalJac(inPoint);
      for (unsigned i = 0; i < m; i++) {
	sum = 0;
	for (unsigned j = 0; j < n; j++) {
	  sum += jac(i, j) * tx[j * (k + 1) + k];
	}
	ty[i * (k + 1) + k] = sum;
      }
      return true;
    }

    // second order taylor coeffs
    if (k == 2) {
      std::vector<ublas::matrix<base, ublas::column_major> > hess =
	interpolators[id]->evalHess(inPoint);
      ublas::matrix<base, ublas::column_major> jac =
	interpolators[id]->evalJac(inPoint);
      ublas::symmetric_adaptor<ublas::matrix<base, ublas::column_major>,
			       ublas::upper> hessAdapt(hess[0]);
      base sumOuter, sumInner;
      for (unsigned i = 0; i < m; i++) {
	sumOuter = 0;
	hessAdapt = ublas::symmetric_adaptor<ublas::matrix<base,
							   ublas::column_major>, ublas::upper>(hess[i]);
	for (unsigned j = 0; j < n; j++) {
	  sumInner = 0;
	  for (unsigned l = 0; l < n; l++) {
	    sumInner += hessAdapt(j, l) * tx[l * (k + 1) + 1];
	  }
	  sumOuter += sumInner * tx[j * (k + 1) + 1] + 2 * jac(i, j)
	    * tx[j * (k + 1) + k];
	}
	ty[i * (k + 1) + k] = 0.5 * sumOuter;
      }
      return true;
  }

    // no case met, evaluation failed
    return false;
}

  /**
   * Reverse evalution of interpolator up to order 2.
   * @param id Which interpolator is used.
   * @param k Order of evaluation.
   * @param n Domain space dimension.
   * @param m Range space dimension.
   * @param tx Taylor coefficients for \f$ X(t) \f$.
   * @param ty Taylor coefficients for \f$ Y(T)=F(X(T)) \f$.
   * @param px [RETURN] Taylor coefficients for \f$ H({x_j^l})=G(F({x_j^l})) \f$.
   * @param py Taylor coefficients for \f$ \partial G/\partial {y_j^l} \f$
   * @return True if succeeded, generally false for k>1.
   */
  template<class base>
  bool interpReverse(std::size_t id, std::size_t k, std::size_t n, std::size_t m,
		     const CppAD::vector<base>& tx, const CppAD::vector<base>& ty,
		     CppAD::vector<base>& px, const CppAD::vector<base>& py) {

    // vector for evaluating the interpolator
    ublas::vector<base> inPoint(n);
    for (unsigned j = 0; j < n; j++)
      inPoint(j) = tx[j * (k + 1)];
    // also jacobian can be used in any case
    ublas::matrix<base, ublas::column_major> jac = interpolators[id]->evalJac(
									      inPoint);

    if (k == 0) {
      base sum;
      for (unsigned j = 0; j < n; j++) {
	sum = 0;
	for (unsigned i = 0; i < m; i++) {
	  sum += py[i * (k + 1)] * jac(i, j);
	}
	px[j * (k + 1)] = sum;
      }
      return true;
    }
    if (k == 1) {
      ublas::vector<base> x1(n);
      for (unsigned j = 0; j < n; j++)
	x1(j) = tx[j * (k + 1) + 1];

      // get hessian information.
      std::vector<ublas::matrix<base, ublas::column_major> > hess =
	interpolators[id]->evalHess(inPoint);
      // fill up the symmetric part
      for (unsigned i = 0; i < m; i++) {
	for (unsigned j = 0; j < n; j++) {
	  for (unsigned l = 0; l < j; l++) {
	    hess[i](j, l) = hess[i](l, j);
	  }
	}
      }

      base sum0;
      base sum1;
      for (unsigned j = 0; j < n; j++) {
	sum0 = 0;
	sum1 = 0;
	for (unsigned i = 0; i < m; i++) {
	  sum0 += py[i * (k + 1)] * jac(i, j) + py[i * (k + 1) + 1]
	    * ublas::inner_prod(ublas::column(hess[i], j), x1);
	  sum1 += py[i * (k + 1) + 1] * jac(i, j);
	}
	px[j * (k + 1)] = sum0;
	px[j * (k + 1) + 1] = sum1;

      }
      return true;
    }
    // no case met
    return false;
  }

/**
 * Compute sparsity pattern for forward evaluation of jacobian, i.e. \f$ S= J*R \f$
 * @param id Which interpolator is used.
 * @param n Domain space dimension.
 * @param m Range space dimension.
 * @param q Number of columns of R.
 * @param r Sparsity pattern for R.
 * @param s [RETURN] Sparsity pattern for S.
 * @return True if succeeded.
 */
bool interpForJacSparse(size_t id, size_t n, size_t m, size_t q,
			const CppAD::vector<std::set<size_t> >& r,
			CppAD::vector<std::set<size_t> >& s) {

  /* sparsity only possible if a whole column of r is zero
   * start with emptyColumns={} and add index if is zero element.
   * start with fullColumns={} and add index if is not zero element.
   * remove from emptyColumns if index is not zero element.
   * remove from fullColumns never.
   * if emptyColumns.size() stays zero or full columns.size() reaches q --> break
   */
  std::set<size_t> emptyColumns;
  std::set<size_t> fullColumns;
  for (size_t i = 0; i < n; i++) {
    for (size_t l = 0; l < q; l++) {
      if (r[i].count(l) == 0 && fullColumns.count(l) == 0)
	emptyColumns.insert(l);
      else {
	emptyColumns.erase(l);
	fullColumns.insert(l);
      }
    }
    // not one element in row is zero
    if (emptyColumns.size() == 0 || fullColumns.size() == q)
      i = n;
  }

  // unless emptyColumns contains an entry all column indices have to be added to set
  for (unsigned j = 0; j < m; j++) {
    s[j].clear();
    for (unsigned l = 0; l < q; l++) {
      // column in r is not completly empty
      if (emptyColumns.count(l) == 0)
	s[j].insert(l);
    }
  }

  return true;
}

/**
 * Sparsity pattern for reverse evaluation of Jacobian, i.e. \f[ R^T=J^T*S^T \f]
 * @param id Interpolator identification.
 * @param n Domain space dimension.
 * @param m Range space dimension.
 * @param q Number of columns in R.
 * @param r [RETURN] Sparsity pattern for R.
 * @param s Sparsity pattern for S.
 * @return True if succeeded.
 */
bool interpRevJacSparse(size_t id, size_t n, size_t m, size_t q,
			CppAD::vector<std::set<size_t> >& r,
			const CppAD::vector<std::set<size_t> >& s) {

  std::set<size_t> emptyColumns;
  std::set<size_t> fullColumns;
  for (size_t j = 0; j < m; j++) {
    for (size_t l = 0; l < q; l++) {
      if (s[j].count(l) == 0 && fullColumns.count(l) == 0)
	emptyColumns.insert(l);
      else {
	emptyColumns.erase(l);
	fullColumns.insert(l);
      }
    }
    // not one element in row is zero
    if (emptyColumns.size() == 0 || fullColumns.size() == q)
      j = m;
  }

  // unless emptyColumns contains an entry all column indices have to be added to set
  for (unsigned i = 0; i < n; i++) {
    r[i].clear();
    for (unsigned l = 0; l < q; l++) {
      // column in r is not completly empty
      if (emptyColumns.count(l) == 0)
	r[i].insert(l);
    }
  }

  return true;
}

/**
 * Sparsity of reverse Hessian evaluation, i.e. \f$ V(x)=(g°f)^(2)*R \f$
 * @param id Number of interpolator used.
 * @param n Domain space dimension.
 * @param m Range space dimension.
 * @param q Number of columns of R.
 * @param r Sparsity pattern for R.
 * @param s Sparsity pattern for \f$ \nabla g \f$
 * @param t [RETURN] Sparsity pattern for \f$ \nabla g \mathcal{Jac}_f \f$.
 * @param u Sparsity pattern for \f$ \nabla^2 g \mathcal{Jac}_f R \f$.
 * @param v [RETURN] Sparsity pattern for \f$ V \f$.
 * @return True if succeeded.
 */
bool interpRevHesSparse(size_t id, size_t n, size_t m, size_t q,
			const CppAD::vector<std::set<size_t> >& r,
			const CppAD::vector<bool>& s, CppAD::vector<bool>& t,
			const CppAD::vector<std::set<size_t> >& u,
			CppAD::vector<std::set<size_t> >& v) {

  // is first diff of g completly zero?
  bool gZero = true;
  for (unsigned j = 0; j < m; j++) {
    // one non-zero element is enough to make result of grad g * jac f not sparse
    if (s[j] == true) {
      gZero = false;
    }
  }
  // grad g has nonzero entries --> t has (in general) all nonzero entries
  if (gZero == false)
    for (size_t i = 0; i < n; i++)
      t[i] = true;
  else
    for (size_t i = 0; i < n; i++)
      t[i] = false;

  /* second order stuff. The (i,j) Element of the Hessian of g°f can be written as
   * Jac(f)^T * Hess(g) * Jac(f) + something depending on grad(g) and second order derivatives
   * of f.
   * The input u contains sparsity of Hess(g)*Jac(f)*R and we need sparsity of that times Jac(f)^T
   * --> reuse revJacobianSparse with output v since it has the right size.
   */

  interpRevJacSparse(id, n, m, q, v, u);

  /* Second order stuff: 0 Matrix if grad(g)=0: {grad(g) * f^(2)(:,i,j)}_(i,j) (f^(2) as 3 dim struct)
   * Otherwise no sparsity --> if multiplied by R:
   * 1: 0 Matrix if grad(g)=0
   * 2: Zero column where R has Zero column
   *
   * Case 1: If grad(g)=0 --> Hess(g)=0 and therefore u=0 --> interpRevJacSparse gives v=0 --> right
   * result.
   * Case 2: If grad(g) != 0:
   * subcase: Hess(g)=0 but R!=0 --> do something
   * subcase: Hess(g)!= 0 R==0 all done with above code (R==0 --> u==0 --> v==0) (remember grad(g)
   * is multiplied by R too)
   * subcase: Hess(g)!=0 R!=0 all done with above code since overall sparsity only depends on R since
   * Hess(g) is multiplied from both sides with non-sparse matrix (jac^T and jac) --> destroys all
   * sparsity --> only Sparse columns of R are preserved.
   *
   * Now do something case: depends only on R: sparse columns of R are preserved --> use old method
   * {grad(g)*f^(2)(:,i,j)}_{i,j} plays role of non-sparse jacobian --> dim=(n,n)
   */
  interpForJacSparse(id, n, n, q, r, v);

  return true;
}
  
// define default vector type if not done in interpCppADSettings.hpp
#ifndef _INTERPCPPADVECTORTYPE_
#define _INTERPCPPADVECTORTYPE_ CppAD::vector
#endif
  
CPPAD_USER_ATOMIC(
		  rbfInterpolator ,
		  _INTERPCPPADVECTORTYPE_,
		  double ,
		  interpForward<double> ,
		  interpReverse<double> ,
		  interpForJacSparse ,
		  interpRevJacSparse ,
		  interpRevHesSparse
		  )
}

#endif /* INTERPCPPADOPERATION_H_ */
