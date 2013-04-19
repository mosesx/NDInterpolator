#ifndef DETERMINE_A_GOOD_C_H
#define DETERMINE_A_GOOD_C_H

#include <iostream>
#include <cassert>
#include <cmath>
#include <typeinfo>
#include <limits>  //for machine eps etc.
//needed for condition number estimation and solution of s.p.d. systems
#include "boost/numeric/ublas/symmetric.hpp"

#include "boost/numeric/bindings/ublas/matrix.hpp"
#include "boost/numeric/bindings/ublas/symmetric.hpp"
#include "boost/numeric/bindings/lapack/driver/posv.hpp"
#include "boost/numeric/bindings/lapack/computational/potrf.hpp"
#include "boost/numeric/bindings/lapack/computational/pocon.hpp"
#include "boost/numeric/bindings/lapack/computational/potrs.hpp"

#include "boost/numeric/bindings/lapack/driver/gesv.hpp"
#include "boost/numeric/bindings/lapack/driver/sysv.hpp"

#include "useful.hpp"
#include "boostmatrixtraits.hpp"
#include "setup.hpp"

#include "boost/math/tools/minima.hpp"

//======== TODO: Decide whether you want some output (1) or not (0) ============
#define SHOW_ME_SOME_OUTPUT_PAL 0
//==============================================================================

namespace NDInterpolator {

/**
 * \brief Corresponds to function <TT> #define SHFT(a,b,c,d) (a)=(b);(b)=(c);(c)=(d);</TT> in [PRESS et al., "Numerical Recipes in C", §10, p. 400].
 */
template<class T>
inline void shift(T& a, T& b, T& c, T& d) {
	a = b;
	b = c;
	c = d;
}

/**
 * \brief This function can be found in Press' book, Appendix B, p. 942.
 */
template<class T>
inline T sign(const T& a, const T& b) {
	return ((b >= T()) ? std::abs(a) : -std::abs(a));
}

/**
 * \brief Similarly programmed as std::pair<T1,T2>.
 *
 *This is equivalent to using std::pair<T1, std::pair<T2,T3> >, but more intuitive. Besides, it provides a nice output operator ;)
 */
template<class T1, class T2, class T3>
class triple {
public:
	typedef T1 first_type;
	typedef T2 second_type;
	typedef T3 third_type;

	T1 first;
	T2 second;
	T3 third;

	triple(const T1& a = T1(), const T2& b = T2(), const T3& c = T3()) :
		first(a), second(b), third(c) {
	}

	//copy construction
	template<class U, class V, class W>
	triple(const triple<U, V, W>& p) :
		first(p.first), second(p.second), third(p.third) {
	}

	//special feature: nice output
	friend std::ostream& operator<<(std::ostream& os, const triple& tri) {
		os << "<" << tri.first << ", " << tri.second << ", " << tri.third
				<< ">" << std::endl;
		return os;
	}
};

/**
 * \brief a 6-tuple. Equivalent to std::pair<triple<·,·,·>, triple<·,·,·> >, but more intuitive in usage
 */
template<class T1, class T2, class T3, class T4, class T5, class T6>
class sixtuple {
public:
	typedef T1 first_type;
	typedef T2 second_type;
	typedef T3 third_type;
	typedef T4 forth_type;
	typedef T5 fifth_type;
	typedef T6 sixth_type;

	T1 first;
	T2 second;
	T3 third;
	T4 forth;
	T5 fifth;
	T6 sixth;

	sixtuple(const T1& a = T1(), const T2& b = T2(), const T3& c = T3(),
			const T4& d = T4(), const T5& e = T5(), const T6& f = T6()) :
		first(a), second(b), third(c), forth(d), fifth(e), sixth(f) {
	}

	template<class U, class V, class W, class X, class Y, class Z>
	sixtuple(const sixtuple<U, V, W, X, Y, Z>& p) :
		first(p.first), second(p.second), third(p.third), forth(p.forth),
				fifth(p.fifth), sixth(p.sixth) {
	}

	//special feature: nice output
	friend std::ostream& operator<<(std::ostream& os, const sixtuple& six) {
		os << "<" << six.first << ", " << six.second << ", " << six.third
				<< ", " << six.forth << ", " << six.fifth << ", " << six.sixth
				<< ">" << std::endl;
		return os;
	}
};

/**
 * \brief Choose solver for symmetric system \f[ A\cdot X = B, \f] where \f$ A\f$ is an \f$ n \times n \f$ matrix, which might be p.d. as well.
 \tparam BM Boost matrix type
 \tparam C char specifying the solver you want to use
 \tparam UL if the matrix is symmetric take the upper (default) or lower triangular part of it
 * 
 * USAGE: Solve the above linear system by invoking, e.g.
 * \code
 SymmetricSystemSolver<boost::numeric::ublas::matrix<double,boost::numeric::ublas::column_major>,'p',boost::numeric::ublas::upper>::compute(A,B,1.75e-16);
 *\endcode
 */
template<class BM, char C, class UL = boost::numeric::ublas::upper>
class SymmetricSystemSolver {
public:
	static inline void compute(BM& A, BM& B,
			const typename BM::value_type& ts = 0.) {
		//throw error because this solver hasn't been specified so far
		printf("%c[%d;%d;%dm", 0x1B, 1, 31, 40); //colorful output ;)
		std::cerr << "***** ERROR thrown by '" << __func__ << "' in file "
				<< __FILE__ << ": The solver '" << C
				<< "' hasn't been defined so far, pal ;)" << std::endl;
		printf("%c[%dm", 0x1B, 0); //reset shell colors
		abort(); //stop computations immediately
	}
};

/**
 * \brief specialisations: solve general \f$ n \times n\f$ system
 * \tparam BM, UL see above
 * \param A lhs of linear system
 * \param B rhs of linear system
 * \param ts threshold value for checking whether rcond is approximately zero or not. This is only meaningful in conjunction with 'c'. Otherwise a default value is taken.
 */
template<class BM, class UL>
class SymmetricSystemSolver<BM, 'g', UL> {
public:
	static inline void compute(BM& A, BM& B,
			const typename BM::value_type& ts = 0.) {
		BoostMatrixTraits<int>::BoostVecType ipiv(A.size1()); //pivoting
		boost::numeric::bindings::lapack::gesv(A, ipiv, B);
	}
};

//! solve general symmetric system
template<class BM, class UL>
class SymmetricSystemSolver<BM, 's', UL> {
public:
	static inline void compute(BM& A, BM& B,
			const typename BM::value_type& ts = 0.) {
		BoostMatrixTraits<int>::BoostVecType ipiv(A.size1()); //pivoting
		//! use adaptor
		boost::numeric::ublas::symmetric_adaptor<BM, UL> Adapt(A);
		//std::cout << Adapt.size1() << "\t" << Adapt.size2() << std::endl;
		boost::numeric::bindings::lapack::sysv(Adapt, ipiv, B);
	}
};

//! solve general p.d. system
template<class BM, class UL>
class SymmetricSystemSolver<BM, 'p', UL> {
public:
	static inline void compute(BM& A, BM& B,
			const typename BM::value_type& ts = 0.) {
		//! use adaptor
		boost::numeric::ublas::symmetric_adaptor<BM, UL> Adapt(A);
		boost::numeric::bindings::lapack::posv(Adapt, B);
	}
};

//! solve general p.d. system WITH reciprocal condition number estimation and ROBUSTNESS check. NOTE: rcond mightn't be very reliable.
//! The default value for the parameter ts, namely \f$ 10\cdot eps,\f$ with <I> eps </I> being the machine precision is taken from [RIPPA, see. full ref. below]
template<class BM, class UL>
class SymmetricSystemSolver<BM, 'c', UL> {
public:
  typedef typename BM::value_type value_type;
  static value_type rcond;
  static inline void compute(
			BM& A,
			BM& B,
			const typename BM::value_type& ts = 10 * std::numeric_limits<
					typename BM::value_type>::epsilon()) {
		//! use adaptor
		typedef Norm<Constants<value_type>::NORMRCOND, value_type>
				RCondNormType;

		boost::numeric::ublas::symmetric_adaptor<BM, UL> Adapt(A);
				
		//value_type rcond = -1; //up to machine precision
		value_type anorm = RCondNormType::norm(Adapt);

		boost::numeric::bindings::lapack::potrf(Adapt);
		boost::numeric::bindings::lapack::pocon(Adapt, anorm, rcond);//!calc. rcond
		boost::numeric::bindings::lapack::potrs(Adapt, B);

		//!robustification: see RIPPA [1], p. 206., mid of page
		//!if the reciprocal cond. number <I>rcond</I> is near 1.0 the matrix is well-conditioned. Else if \f$ rcond \approx 0.0\f$ then the matrix is ill-conditioned. Despite its computational efficiency, rcond isn't much reliable.
		//std::cout << "rcond = "<<rcond << "    eps = "<< std::numeric_limits<value_type>::epsilon()<< std::endl;
	
	}
};

template<class BM, class UL>
typename SymmetricSystemSolver<BM, 'c', UL>::value_type SymmetricSystemSolver<BM, 'c', UL>::rcond=0;

/**
 * \brief Our specific function \f$ f(c) := \frac{1}{N+1}\| E_j\|, \ E_j = [e_{ij}]_{i = 1}^k, j \textrm{fixed}.\f$
 *
 * \tparam IP interpolation object type
 *
 * The cost function is meant as a generalization of an approach which is described in ref. [1]
 *
 * Reference:
 *
 * [1] [S. RIPPA, "An algorithm for selecting a good value for the parameter <I> c </I> in radial basis function interpolation", Advances in Computational Mathematics 11 (1999) 193-210]
 */
template<class IP>
class CostFunction4GoodC {
public:
	typedef typename IP::value_type value_type;
	typedef typename BoostMatrixTraits<value_type>::BoostMatrixType
			BoostMatrixType;
	typedef typename BoostMatrixTraits<value_type>::BoostVecType BoostVecType;

	//!Select norm of your choice
	//!'1', '2' and 'i' ('I') for the \f$ l_1, l_2 \f$ and \f$ l_{\infty}\f$ norm, respectively

	typedef Norm<Constants<value_type>::NORMBRENT, value_type> NormType;

	//must be '1'or 'i'('I')
	typedef Norm<Constants<value_type>::NORMRCOND, value_type> RCondNormType;

	CostFunction4GoodC() {
	}// default

	CostFunction4GoodC(const IP& interpol) :
		interpol_(interpol) {
		//must be formed only once!
		Chained_.resize(interpol.getNumOfPoints(),
				interpol.getOutDim() + interpol.getNumOfPoints());
		concatenate(Chained_, interpol.getData(),
				identity_full<BoostMatrixType> (interpol.getNumOfPoints()));
		//now Chained_ = [F|I]
	}

	value_type operator()(const value_type& c) {
		//! Independently from whether or not the interpolation system has already been solved, the following member function returns \f$ A(c)\f$. Note that it has invoked anew each time a new \f$ c \f$ is computed
		BoostMatrixType A = interpol_.getInterpMatrix(c);

		C_ = Chained_; //since C_ is overwritten each time the system is solved

		//!solve \f$ n \times n \f$ system
		SymmetricSystemSolver<BoostMatrixType,
				Constants<value_type>::SYMMSYSTEMSOLVER,
				boost::numeric::ublas::upper>::compute(A, C_);

		value_type rcond = SymmetricSystemSolver<BoostMatrixType,
							 Constants<value_type>::SYMMSYSTEMSOLVER,
							 boost::numeric::ublas::upper>::rcond;
		//A.size1() and F.size1() are equal
		v_.resize(A.size1());

		value_type s = value_type(); //return value

		unsigned dF2 = interpol_.getOutDim(), dF1 = interpol_.getNumOfPoints();

		for (unsigned j = 0; j < dF2; ++j) {
		  for (unsigned i = 0; i < dF1; ++i) {
		    //avoid division by a value close to zero
		    assert(std::abs(C_(i,i+dF2)) >= Constants<value_type>::APPROXZERO);
		    
		    v_(i) = C_(i, j) / C_(i, i + dF2);
		  }
		  
		  s += NormType::norm(v_);
		}
		assert(dF2 > 0);
		
		/*
		 * Robustification, if rcond get's to small, add inverse to objective
		 * small means here smaller than 1e-10
		 */
		value_type addToObj= 0.;

		// if (rcond < (1e5*std::numeric_limits<value_type>::epsilon())){
		//   addToObj = 1/rcond - 1/(1e5*std::numeric_limits<value_type>::epsilon());
		//   std::cout << "toObj " << addToObj << std::endl;
		// }

		 // addToObj = 1/rcond;
		 // return s / (dF2) + 1e-10*addToObj; 

		 addToObj = 10*std::numeric_limits<value_type>::epsilon()/rcond;
		 return s / (dF2) + addToObj; 
	
	}

private:
	const IP& interpol_; //const reference
	BoostMatrixType C_, Chained_;
	BoostVecType v_;
};

/**
 * \brief Initially bracketing a minimum
 *
 * Ref.: [PRESS et al., "Numerical Recipes in C", §10, p. 400]
 */
template<class T>
class InitialBracketingOfMinimum {
public:

	InitialBracketingOfMinimum(const T& ax = T(), const T& bx = T()) :
		ax_(ax), bx_(bx) {
		assert(!compFloat<T>(ax,bx,Constants<T>::ZEPS)); //must be distinct
	}

	const T& ax() const {
		return ax_;
	}
	const T& bx() const {
		return bx_;
	}

	// ! Taken from Press et all
	//sixtuple<T,T,T,T,T,T>
	template<class FUN>
	triple<T, T, T> mnbrak(FUN& func) {
		T cx, fa, fb, fc, ulim, u, r, q, fu, dum, my1, my2, soso;

		fa = func(ax_);
		fb = func(bx_);

		(fb > fa) ? shift(dum, ax_, bx_, dum) : shift(dum, fb, fa, dum);

		cx = bx_ + Constants<T>::GOLD * (bx_ - ax_);

		fc = func(cx);

		while (fb > fc) {
			r = (bx_ - ax_) * (fb - fc);

			q = (bx_ - cx) * (fb - fa);
			soso = Constants<T>::TINY;
			u = bx_ - ((bx_ - cx) * q - (bx_ - ax_) * r) / (2. * sign(
					std::max(std::abs(q - r), soso), q - r));
			ulim = bx_ + Constants<T>::GLIMIT * (cx - bx_);

			if ((bx_ - u) * (u - cx) > 0.0) {
				fu = func(u);

				if (fu < fc) {
					ax_ = bx_;
					bx_ = u;
					fa = fb;
					fb = fu;
					//return 0; //got a minimum between b and c
					break;

				} else if (fu > fb) {
					cx = u;
					fc = fu;
					//return 0; //got a minimum between a and u
					break;
				}

				u = cx + Constants<T>::GOLD * (cx - bx_);

				fu = func(u);

			} else if ((cx - u) * (u - ulim) > 0.0) {
				fu = func(u);
				if (fu < fc) {
					my1 = cx + Constants<T>::GOLD * (cx - bx_);
					my2 = func(u);
					shift(bx_, cx, u, my1);
					shift(fb, fc, fu, my2);
				}

			} else if ((u - ulim) * (ulim - cx) >= 0.0) {

				u = ulim;
				fu = func(u);

			} else {

				u = cx + Constants<T>::GOLD * (cx - bx_);
				fu = func(u);
			}

			shift(ax_, bx_, cx, u);
			shift(fa, fb, fc, fu);
		} //end while

		//sixtuple<T,T,T,T,T,T> six(ax_,bx_,cx,fa,fb,fc);
		triple<T, T, T> three(ax_, bx_, cx);
		return three; //six;
	}

private:
	T ax_, bx_; //! bracket the minimum
};

/**
 * \brief Decide whether you want to return the arg min value (which, in our case, is \f$ c \f$) or \f$ f(c) \f$ which is the minimum function value.
 *
 * NOTE: We need the arg min
 */
template<class C, class D, bool B> class ArgMinFunMinSelector;

//!partial specialisation
template<class C, class D>
class ArgMinFunMinSelector<C, D, true> {
public:
	static inline const C& val(const C& c, const D& fc) {
		return c;
	}
};

template<class C, class D>
class ArgMinFunMinSelector<C, D, false> {
public:
	static inline const D& val(const C& c, const D& fc) {
		return fc;
	}
};

/**
 * \brief Brent's Method -- a derivative-free optimization algorithm for a function in one variable, \f$ f:\mathbf{R} \longrightarrow \mathbf{R}. \f$
 *
 * \tparam A Anything with which the template-template class F is templatized
 * \tparam F Functor
 * \tparam ARGMIN decides whether argmin is returned or min f(c) (default: true, i.e. argmin (the shape parameter in our case) is returned
 *
 * REF.: 1.) [PRESS et al., "Numerical Recipes in C", §10, p. 404]
 *       2.) [BRENT, "Algorithms for Minimization without Derivatives", § 5]
 */
template<class A, template<class S> class F = CostFunction4GoodC, bool ARGMIN =
		true>
class BrentsMethod {
public:
	typedef F<A> FunType;
	typedef typename FunType::value_type value_type;

	BrentsMethod() :
		f_() {
	} //default constr.

	BrentsMethod(const A& ip) :
		f_(ip) {
	}

	value_type minNeu(const value_type& start, const value_type& end,
			const int bits) {
		std::pair<value_type, value_type> res =
				boost::math::tools::brent_find_minima(f_, start, end, bits);
		return res.first;
	}
	//! corresponds to function 'brent(...)' in ref. 1.)
	value_type minimum(const value_type& start, const value_type& final,
			const value_type& tol) {

		assert(tol > value_type());

		//!bracket the minimum of f_ in the first place
		//! NOTE: inital bracketing of a minimum should <B>always</B> be done!
		InitialBracketingOfMinimum<value_type> bracketmini(start, final);
		//sixtuple<value_type,value_type,value_type,value_type,value_type,value_type> six = bracket.mnbrak(f_);
		triple<value_type, value_type, value_type> brk = bracketmini.mnbrak(f_);

		//given; these 3 values bracket the minimum of the function f_
		value_type ax = brk.first, bx = brk.second, cx = brk.third;

#if SHOW_ME_SOME_OUTPUT_PAL	
		//ax < bx < cx
		std::cout << "Initial bracketing yields {ax, bx, cx} = " << brk
		<< std::endl;
#endif

		value_type xmin; //abscissa of the minimum

		//!code taken and changed from ref. 1.)
		unsigned iter;
		value_type a, b, d, etemp, fu, fv, fw, fx, p, q, r, tol1, tol2, u, v,
				w, x, xm;

		d = value_type();
		value_type e = 0.0;

		a = (ax < cx ? ax : cx);

		b = (ax > cx ? ax : cx);

		x = w = v = bx;
		fw = fv = fx = f_(x);

		for (iter = 1; iter <= Constants<value_type>::ITMAX; ++iter) {
			xm = 0.5 * (a + b);
			tol2 = 2.0 * (tol1 = tol * std::abs(x)
					+ Constants<value_type>::ZEPS);

			if (std::abs(x - xm) <= (tol2 - 0.5 * (b - a))) {
				xmin = x;

#if SHOW_ME_SOME_OUTPUT_PAL
				std::cout << "Min f(c) = " << fx << " detected after " << iter
				<< " iterations at c = " << xmin << " with solver '"<< Constants<value_type>::SYMMSYSTEMSOLVER << "'."<< std::endl;
#endif
				//return fx;                            //minimum
				return ArgMinFunMinSelector<value_type, value_type, ARGMIN>::val(
						xmin, fx);
			}
			//Construct a trial parabolic fit.
			if (std::abs(e) > tol1) {
				r = (x - w) * (fx - fv);
				q = (x - v) * (fx - fw);
				p = (x - v) * q - (x - w) * r;
				q = 2.0 * (q - r);
				if (q > 0.0)
					p = -p;
				q = std::abs(q);

				etemp = e;
				e = d;
				if (std::abs(p) >= std::abs(0.5 * q * etemp) || p <= q
						* (a - x) || p >= q * (b - x))
					d = Constants<value_type>::CGOLD * (e = (x >= xm ? a - x
							: b - x));
				//The above conditions determine the acceptability of the parabolic fit. Here we take the golden section step into the larger of the two segments.
				else {

					d = p / q;
					u = x + d;
					if (u - a < tol2 || b - u < tol2)
						d = sign(tol1, xm - x);
				}
			} else {
				d = Constants<value_type>::CGOLD * (e = (x >= xm ? a - x : b
						- x));
			}
			u = (fabs(d) >= tol1 ? x + d : x + sign(tol1, d));
			fu = f_(u);

			if (fu <= fx) {

				if (u >= x)
					a = x;
				else
					b = x;

				shift(v, w, x, u);
				shift(fv, fw, fx, fu);
			} else {
				if (u < x)
					a = u;
				else
					b = u;
				if (fu <= fw || compFloat<value_type> (w, x,
						Constants<value_type>::ZEPS)) {
					v = w;
					w = u;
					fv = fw;
					fw = fu;
				} else if (fu <= fv || compFloat<value_type> (v, x,
						Constants<value_type>::ZEPS) || compFloat<value_type> (
						v, w, Constants<value_type>::ZEPS)) {
					v = u;
					fv = fu;
				}
			}

		}

		if (iter == Constants<value_type>::ITMAX) {
			printf("%c[%d;%d;%dm", 0x1B, 1, 31, 40);
			std::cout
					<< "**** ERROR: Too many iterations in Brent's method. \n     Try increasing 'Constants<value_type>::ITMAX' for the time being..."
					<< std::endl;
			printf("%c[%dm", 0x1B, 0);
		}

		xmin = x; //abscissa at which minimum is achieved

#if SHOW_ME_SOME_OUTPUT_PAL
		std::cout << "Min f(c) = " << fx << " detected after " << iter
		<< " iterations at c = " << xmin << " with solver '"<< Constants<value_type>::SYMMSYSTEMSOLVER<< "'."<< std::endl;
#endif
		//return fx;                            //minimum
		return ArgMinFunMinSelector<value_type, value_type, ARGMIN>::val(xmin,
				fx);

	}

private:
	FunType f_;
};

} //end namespace

#endif
