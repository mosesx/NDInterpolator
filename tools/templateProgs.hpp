#ifndef MARC_S_TMPS_TO_INCREASE_PERFORMANCE_H
#define MARC_S_TMPS_TO_INCREASE_PERFORMANCE_H

#include <cassert>

namespace TMP{
  
 /**
   * \brief Note that for \f$ a \in \mathbf{K}\f$ and for \f$ n \in \mathbf{N}\f$, we have \f$ a^{-n} = \frac{1}{a^n}.\f$
   */ 
 template<bool B, class T> class InvExp;

  template<class T>
  class InvExp<true,T>{
  public:
    static inline T invexp(const T& e){
      assert(std::abs(e) > T());

      return 1./e;
    }
  };

  template<class T>
  class InvExp<false,T>{
  public:
    static inline T invexp(const T& e){
      return e;
    }
  };

  /**
   * \brief Compute the <EM>natural</EM> power of a given value at <B>compile time</B> by using a simple but dashing TMP, i.e. calculate
   * \f[ p = x^n, \quad n \in \mathbf{N}, x \in \mathbf{K}\f] by compile time recurrence:
   *
   * \f{eqnarray*}{
      x &:=& x\cdot x^{n-1}, \\
      x^0 &=& 1,
      \f}
   *
   * USAGE: Suppose you want to compute \f$ \prod_{i=1}^{5} -1.765, \f$ at compile time, just type  
   * \code
      std::cout << TMP::ntimesx<5>(-1.765) << std::endl;
      //a complex example
      std::complex<double> z(-3.,4);
      std::cout << TMP::ntimesx<3>(z) << std::endl;
   * \endcode
   * 
   * RESULT: for the 1st example, the following scenario occurs during comilation:
   *
   *TMP::ntimesx<5>  = -1.765*TMP::ntimesx<4>(-1.765)
   *                 = -1.765*(-1.765)*TMP::ntimesx<3>(-1.765)
   *                 = -1.765*(-1.765)*(-1.765)*TMP::ntimesx<2>(-1.765)
   *                 = -1.765*(-1.765)*(-1.765)*(-1.765)*TMP::ntimesx<1>(-1.765)
   *                 = (-1.765)^4 *TMP::ntimesx<0>(-1.765)
   *                 = (-1.765)^5 * 1 = -17.1287 
   */
  template<int N, class X>  
  class NTimesX{
  public:
    static inline X raise_to_the_power_of_N(const X& x){
      return x*NTimesX<N-1,X>::raise_to_the_power_of_N(x);
    }
  };

  
  //!end of recursion: \f$ x^0 = 1\f$
  template<class X>
  class NTimesX<0,X>{
  public:
    static inline X raise_to_the_power_of_N(const X& x){
      return X(1);
    }
  };
 

  //!convenient function for easy usage, see above. Works well with negative N as well
  template<int N, class X>
  inline X ntimesx(const X& x){
    //return NTimesX<N,X>::raise_to_the_power_of_N(x);
    return InvExp<(N < 0),X>::invexp(NTimesX<((N < 0) ? -N : N),X>::raise_to_the_power_of_N(x));
  }
  
} //end namespace 

#endif 
