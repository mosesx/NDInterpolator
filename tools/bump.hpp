#ifndef BUMP_H_
#define BUMP_H_

#include <cmath>

#include "boost/numeric/ublas/matrix.hpp"
#include "boost/numeric/ublas/vector.hpp"
#include "boost/numeric/ublas/matrix_expression.hpp"

namespace NDInterpolator {
  namespace ublas = boost::numeric::ublas;

  template<class value_type>
  class Bump {

  public:
    Bump(const ublas::matrix<value_type, ublas::column_major>& BB) : BB(BB){
      alpha = .9;
      a = (-3.141592653589793/2)/(-1+alpha);
      b = (-3.141592653589793/2*alpha)/(-1+alpha);     
    };

    ublas::vector<value_type> evalGrid(const ublas::matrix<value_type, ublas::column_major>& grid) const {
      ublas::vector<value_type> ret(grid.size2());

      for (size_t k = 0; k < grid.size2(); k++){
	ret(k) = ndGbumpScaled(ublas::column(grid,k));
      }
      return ret;
    }

    ublas::vector<value_type> d_evalGrid(const ublas::matrix<value_type, 
							     ublas::column_major>& grid,
					 const size_t d) const {
      ublas::vector<value_type> ret(grid.size2());

      for (size_t k = 0; k < grid.size2(); k++){
	ret(k) = d_ndGbumpScaled(ublas::column(grid,k),d);
      }
      return ret;
    }

    void applyToData(const ublas::matrix<value_type,
		     ublas::column_major>& grid,
		     ublas::matrix<value_type,
		     ublas::column_major>& data) const {

      ublas::vector<value_type> bFun = evalGrid(grid);
      for (size_t k = 0; k < data.size2(); k++) {
	ublas::column(data,k) = ublas::column(data,k) * bFun(k);
      }
    }

    void applyToDiffData(const ublas::matrix<value_type,
					     ublas::column_major>& grid,
			 const ublas::matrix<value_type,
					     ublas::column_major>& data,
			 std::vector<ublas::matrix<value_type, 
						   ublas::column_major> >& diffData) {

      ublas::vector<value_type> bFun = evalGrid(grid);

      size_t inDim = grid.size1();
      size_t numOfPoints = grid.size2();
      ublas::matrix<value_type> d_bFun(inDim,numOfPoints);
      // eval on all grid points and for all dimensions
      for (size_t k = 0; k < inDim; k++) {
	ublas::row(d_bFun,k) = d_evalGrid(grid,k);
      }
      for (size_t k = 0; k < numOfPoints; k++) {
	diffData[k] = diffData[k] * bFun(k) + 
	  ublas::outer_prod(ublas::column(data,k),
			    ublas::column(d_bFun,k));	
      }
    }

    void applyToDiffAndData(const ublas::matrix<value_type,
					     ublas::column_major>& grid,
			    ublas::matrix<value_type,
					     ublas::column_major>& data,
			    std::vector<ublas::matrix<value_type, 
						   ublas::column_major> >& diffData) {
      // apply to diff data first to avoid overwriting data because original is needed
      applyToDiffData(grid, data, diffData);
      applyToData(grid,data);
    }

  private:
    ublas::matrix<value_type, ublas::column_major> BB;

    value_type alpha;
    
    value_type a;
    value_type b;

    /**
     * Value of 1-d bump function on interval [-1,1]
     * @param x Inpoint.
     * @return Function value.
     */
    value_type gbump(const value_type x) const {
    if (x <= -1 || x >= 1) {
	return 0;
      }
      if (-1 < x & x < -alpha){
	return exp(-tan(a*x + b) * tan(a*x + b));
      }
      if (-alpha <= x & x <= alpha) {
	return 1.;
      }
      if (alpha < x & x < 1.){
	return exp(-tan(a*x - b) * tan(a*x - b));
      }
    }
    
    /**
     * Value of 1-d bump function on interval [BB(i,0), BB(i, 1)]
     * @param x Inpoint.
     * @param d Which spatial dimension.
     * @return Function value.
     */
    value_type gbumpScaled(const value_type x, const size_t d) const{
      return gbump((-2*x + BB(d,0) + BB(d,1))/(BB(d,0)-BB(d,1)));
    }
    
    /**
     * Value of n-d bump function on bounding box.
     * @param inPoint Well.
     * @return Function value.
     */ 
    value_type ndGbumpScaled(const ublas::vector<value_type>& inPoint) const{
      value_type ret = 1;
      for (size_t k = 0; k < inPoint.size(); k++) {
	ret *= gbumpScaled(inPoint(k),k);
      }
      return ret;
    }

    /**
     * Value of derivative of 1-d bump function on interval [-1,1]
     * @param x Inpoint.
     * @return Function value.
     */
    value_type d_gbump(const value_type x) const {
      if (x <= -1 || x >= 1 || (-alpha <= x & x <= alpha)) {
	return 0;
      }
      if (-1 < x & x < -alpha) {
	return -2*a*exp(-tan(a*x + b) * tan(a*x + b)) * 
	  (1/cos(a*x + b)) * (1/cos(a*x + b)) * tan(a*x + b);
      }
      if (alpha < x & x < 1.){
	return -2*a*exp(-tan(a*x - b) * tan(a*x - b)) * 
	  (1/cos(a*x - b)) * (1/cos(a*x - b)) * tan(a*x - b);
      }
    }

    /**
     * Value of 1-d derivative bump function on interval [BB(i,0), BB(i, 1)]
     * @param x Inpoint.
     * @param d Which spatial dimension?
     * @return Function value.
     */
    value_type d_gbumpScaled(const value_type x, const size_t d) const{
      return -2/(BB(d,0)-BB(d,1))*d_gbump((-2*x + BB(d,0) + BB(d,1))/(BB(d,0)-BB(d,1)));
    }

    /**
     * Value of derivative of n-d bump function on bounding box.
     * @param inPoint Well.
     * @param d Direction.
     * @return Function value.
     */ 
    value_type d_ndGbumpScaled(const ublas::vector<value_type>& inPoint, const size_t d) const{
      value_type ret = 1;
      for (size_t k = 0; k < inPoint.size(); k++) {
	if (k == d) {
	  ret *= d_gbumpScaled(inPoint(k),k);
	}
	else {
	  ret *= gbumpScaled(inPoint(k),k);
	}
      }
      return ret;
    }


  };
} // namespace

#endif // BUMP_H_
