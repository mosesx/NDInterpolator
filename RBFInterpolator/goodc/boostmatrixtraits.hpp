#ifndef BOOST_MATRIX_TRAITS_H
#define BOOST_MATRIX_TRAITS_H

namespace NDInterpolator{
  
  /**
   * \brief Define your Boost types here once and for all!
   */
  template<class T>
    class BoostMatrixTraits{
  public:
    typedef T value_type;
    typedef boost::numeric::ublas::column_major LayoutType;
    typedef boost::numeric::ublas::matrix<T,LayoutType> BoostMatrixType;
  
    typedef  boost::numeric::ublas::vector<T> BoostVecType;
  };
  
}

#endif 
