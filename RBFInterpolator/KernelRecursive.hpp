#ifndef RECURSIVE_SCALAR_RADIAL_DERIVATIVES_H
#define RECURSIVE_SCALAR_RADIAL_DERIVATIVES_H

#include <iostream>
#include <cmath>
#include "../tools/utility.hpp"
#include "../tools/templateProgs.hpp"


namespace NDInterpolator{ 

  /**
   * \brief Definition: The <b>scaled radial kernel</b> is defined by \f[ K_c(x,y) := \phi(r^2/(2c^2)) := f(r^2/(2c^2)) := f(s) \f]. 
   *
   * The function implements \f$ s\f$ i.e. the argument of \f$ f(s) := f(\|x-y\|_2^2/(2c^2)), \ r:= \|x-y\|_2, x, y \in \mathbf{R}^n \f$
   */
  template<class T>
  inline T kernel_s(const T& r, const T & c){
    assert(std::abs(c) >= 1.e-13);  //devision by ~ zero is not permitted
    return 0.5*TMP::ntimesx<2>(r/c);
  }


  /**
   * \brief Recurrence formula for the \f$ k\f$-th derivative of  \f$ f_{\beta}(s)\f$ w.r.t. \f$s\f$, calculated by me, is given via
   * \f{eqnarray*}{ f^{(k)}_{\beta}(s) &=& (\beta -2(k-1))f^{(k-1)}_{\beta}(s),\\
   * f^{(1)}_{\beta}(s) &=& \beta(\sqrt(1+2s))^{\beta-2k}, \\ f^{(0)}_{\beta}(s) &=& (\sqrt(1+2s))^{\beta}. \f}
   *
   * NOTE: Applied \f$(\cdot)^{\beta/2} = (\sqrt(\cdot))^{\beta}\f$ to get rid of slow \code std::pow(.,.) \endcode. 
   *
   * This is needed to calculate the derivatives of the general multiquadric (includes inverse multiquadric as well for \f$ \beta < 0\f$).
   *
   * cf. [SCHABACK,"Programming Hints for Kernel-Based Methods", 2010] 
   */
  template<int K, int BETA, class T, int UP = K>
  class F_B_2{
  public:
    static inline T decrease(const T& r, const T& c){
      //std::cout<<"beta - 2*(K-1) = "<< (BETA-2*(K-1))  <<std::endl;
      return (BETA-2*(K-1))*F_B_2<K-1,BETA,T,UP>::decrease(r,c);
    }
  };

  //!partial specialisation: end of recursion, i.e. \f$ \beta f_{\beta + 2k}(s)  = \beta(1+2s)^{\beta/2 -k}, \f$ where \f$k \f$ denotes the order of the derivative
  template<int BETA, class T, int UP>
  class F_B_2<1,BETA,T,UP>{
  public:
    static inline T decrease(const T& r, const T& c){
      return  BETA*TMP::ntimesx<(BETA-2*UP)>(std::sqrt(1+ 2*kernel_s(r,c)));
    }
  };

  //!partial specialisation: no derivative -- just return \f$ f_{\beta}(s) := (1+2s)^{\beta/2}\f$
  template<int BETA, class T, int UP>
  class F_B_2<0,BETA,T,UP>{
  public:
    static inline T decrease(const T& r, const T& c){
      return TMP::ntimesx<BETA>(std::sqrt(1 + 2*kernel_s(r,c)));
    }
  };
  
  /**
   * \brief Convenient function which implements derivative of order \f$ k\f$ w.r.t. \f$ s \f$ for (inverse) multiquadrics; i.e. we get \f[ \partial^k_{s^k} f_{\beta}(s) = \left( \prod_{i=0}^{k-1}(\beta - 2i)\right)\cdot f_{\beta-2k}(s), \quad \beta \in \mathbf{Z} \f]
   * \tparam K order of derivative w.r.t. s
   * \tparam BETA \f$ \beta \f$ (usually: -1 for inverse multiquadric rbf)
   * \tparam T precision
   */
  template<int K, int BETA, class T>  
  inline T mq_derivative(const T& r, const T& c){
    assert(K > -1); //only definded for derivatives of positive order
    return F_B_2<K,BETA,T,K>::decrease(r,c);
  }
  



}//end of namespace 

#endif 
