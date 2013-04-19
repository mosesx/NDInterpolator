#ifndef SETTINGS_FOR_CALCULATING_A_GOOD_C_H_
#define SETTINGS_FOR_CALCULATING_A_GOOD_C_H_

namespace NDInterpolator{

  /**
   * \brief ==== TODO: Define some <I>typified </I> global constants here =====
   * \tparam T value_type of numeric constant (i.e. float, double, long double,...)
   *
   * Note: use these instead of untypified <TT> #define GOLD = 1.618... </TT> directives.
   *   =========================================================================
   */
  template<class T>
  class Constants {
  public:
    
    //!used for IntialBracketing and Brent's method
    static const T GOLD;
    static const T GLIMIT;
    static const T TINY;
    static const unsigned ITMAX = 100;
    static const T CGOLD;
    static const T ZEPS;
    
    static const T APPROXZERO; //value that is considered zero
    
    //!norm for CostFunction4GoodC
    static const char NORMBRENT = '1';  //u might try '1' and 'i' as well
    //!norm needed for estimating <I> rcond</I>
    static const char NORMRCOND = '1';  //choose between '1' and 'i' only
    
    //! 'p' = p.d. system, 'g' = general nxn system with pivoting, 's' = symm. with pivoting,  'c' = p.d. system with calculation of the reciprocal condition number
    static const char SYMMSYSTEMSOLVER = 'c';
    
    //reverse condition number tolerance
    static const T RCONDTOL;
    
  };
  
  template <class T>
  const T Constants<T>::GOLD=1.6180339887; //!\f$\phi = \frac{1 +sqrt{5}}{2}\f$

  template <class T>
  const T Constants<T>::GLIMIT = 100.0;

  template <class T>
  const T Constants<T>::TINY = 1.0e-20;
  
  template <class T>
  const T Constants<T>::CGOLD = 0.3819660;

  template <class T>
  const T Constants<T>::ZEPS = 1.0e-10;

  template <class T>
  const T Constants<T>::APPROXZERO = 1.e-20; //value that is considered zero

  template <class T>
  const T Constants<T>::RCONDTOL = 1 * 1e-14;


} //end namespace NDInterpolator

#endif
