/**
 * MultiRBFInterpolatorPUH.h
 *
 *  Created on: \date Dec 16, 2010
 *      Author: \author mrehberg
 */

#ifndef MULTIRBFINTERPOLATORPUH_H_
#define MULTIRBFINTERPOLATORPUH_H_

// std includes
#include <cassert>
#include <algorithm>
#include <vector>
#include <cstddef>
#include <limits>
#include <cmath>

// boost includes
#include "boost/multi_array.hpp"
#include "boost/numeric/ublas/vector_proxy.hpp"
#include "boost/math/special_functions/pow.hpp"

// NDInterpolator includes
#include "../MultiNDInterpolator.hpp"
#include "../tools/utility.hpp"
#include "../tools/bump.hpp"
#include "MultiRBFInterpolatorH.hpp"
#include "BoxWeight.hpp"

namespace NDInterpolator {

  /** \brief Multi dim output Hermite interpolator based on a partition of unity approach for domain
   * space separation.
   * \tparam RTYPE Type of radial basis function to be used, e.g. NDInterpolator::GaussianRBF<double>.
   * \tparam inDimT Dimension of the domain space (input dimension).
   * \tparam outDimT Dimension of the range space (output dimension).
   */
  template<class RTYPE, std::size_t inDimT, std::size_t outDimT>
  class MultiRBFInterpolatorPUH : public MultiNDInterpolator {
  public:

    MultiRBFInterpolatorPUH();

    /*
     * instead of using various template arguments, we just take the value_type from the
     * corresponding radial basis function. Make sure that the rbf class possesses something
     * like 'typedef T value_type'
     */
    typedef typename RTYPE::value_type value_type;

    MultiRBFInterpolatorPUH(const ublas::matrix<value_type, ublas::column_major>& grid,
			    const ublas::matrix<value_type, ublas::column_major>& data,
			    const std::vector<ublas::matrix<value_type,
			    ublas::column_major> >& diffData,
			    const value_type scale = 1., const unsigned K = 100,
			    const value_type alpha = .33, const bool useBump = false);

    /**
     * Assign values to every member and reinitialize the interpolator object.
     * @param grid Set of center points.
     * @param data Matrix of data points, each column is associated to one column in grid.
     * @param diffData Vector of jacobians. Each entry in the std::vector is associated
     * to one center point and is a matrix of size outDim x outDim representing the jacobian in
     * that point.
     * @param scale Scaling factor for the underlying RBF function.
     * @param K Minimum number of points per patch.
     * @param alpha Overlap of neighboring patches, 0< alpha < 0.5.
     */
    void assign(const ublas::matrix<value_type, ublas::column_major>& grid,
		const ublas::matrix<value_type, ublas::column_major>& data,
		const std::vector<ublas::matrix<value_type,
						ublas::column_major> >& diffData,
		const value_type& scale = 1., const unsigned& K = 100,
		const value_type& alpha = .33);

    MultiRBFInterpolatorPUH& operator=(const MultiRBFInterpolatorPUH& rhs) {
      
      if (this == &rhs)
	return *this;
      
      // super class members
      inDim = rhs.inDim;
      outDim = rhs.outDim;
      numOfPoints = rhs.numOfPoints;      
      
      // easy copyable class members
      scale = rhs.scale;
      omega = rhs.omega;
      BB = rhs.BB; 
      h = rhs.h;
      K = rhs.K;
      alpha = rhs.alpha; 
      o = rhs.o;
      M = rhs.M;
      useBump = rhs.useBump;

      jacGlob = rhs.jacGlob;
      retRBFGlob = rhs.retRBFGlob;
      retGlob = rhs.retGlob;
      diffRetGlob = rhs.diffRetGlob;
      diffWeightGlob = rhs.diffWeightGlob;
      diffWeightSumGlob = rhs.diffWeightSumGlob;
      ELGlob = rhs.ELGlob;
      ERGlob = rhs.ERGlob;
      isBorderGlob = rhs.isBorderGlob;
      retWeightGlob = rhs.retWeightGlob;
      retWeightSumGlob = rhs.retWeightSumGlob;
      IGlob = rhs.IGlob;
      
      // copy interpolator objects, need to resize interpol
      ublas::vector<unsigned, ublas::bounded_array<unsigned, inDimT> > shape(inDimT);
      for (unsigned i = 0; i < inDimT; i++)
	shape(i) = M;
      interpol.resize(shape);
      for (unsigned i = 0; i < std::pow(M, inDimT); i++) {
	interpol(ind2sub(i)) = rhs.interpol(ind2sub(i));
      }
      return *this;
    }

    /**
     * Try to calculate the optimal scaling parameter for the interpolator.
     * The calculated scale is used for all computations after a call to this method. 
     * For each Patch a optimization is performed independently.
     */
    void optimizeScale() {
      unsigned nTiles = static_cast<unsigned> (boost::math::pow<inDimT>(M));
      for (unsigned i = 0; i < nTiles; i++) {
       	interpol(ind2sub(i)).optimizeScale();
      }
    }

    /**
     * Try to calculate the optimal scaling parameter for the interpolator.
     * The calculated scale is used for all computations after a call to this method. 
     * For each Patch a optimization is performed independently.
     */
    void optimizeScale(const double left,const double right) {
      unsigned nTiles = static_cast<unsigned> (boost::math::pow<inDimT>(M));
      for (unsigned i = 0; i < nTiles; i++) {
       	interpol(ind2sub(i)).optimizeScale(left, right);
      }
    }
    
    /**
     * Set the scaling parameter for the RBF to a new value and re-init. After a call,
     * the interpolation object is ready to be used again.
     * @param scale New scale
     */
    void setNewScale(const double scale){
      unsigned nTiles = static_cast<unsigned> (boost::math::pow<inDimT>(M));
      for (unsigned i = 0; i < nTiles; i++) {
	interpol(ind2sub(i)).setNewScale(scale);
      }
    }

    /**
     * Prints the scale of all patches to std::cout. 
     */
    void printScale(std::ostream& outStream=std::cout){
      unsigned nTiles = static_cast<unsigned> (boost::math::pow<inDimT>(M));
      for (unsigned i = 0; i < nTiles; i++) {
	outStream << interpol(ind2sub(i)).getScale() << "\t" <<
	  interpol(ind2sub(i)).getNumOfPoints() << std::endl;
      }
    }
    
    ublas::vector<value_type>
      eval(const ublas::vector<value_type>& inPoint);

    ublas::vector<value_type>
      evalDiff(const ublas::vector<value_type>& inPoint,
	       const unsigned int alpha);

    ublas::vector<value_type>
      evalDiff2(const ublas::vector<value_type>& inPoint,
		const unsigned alpha);

    ublas::vector<value_type> evalDiffMixed(
					    const ublas::vector<value_type>& inPoint, const unsigned alpha1,
					    const unsigned alpha2);

    ublas::vector<value_type>
      evalGrad(const ublas::vector<value_type>& inPoint,
	       const unsigned alpha);

    ublas::matrix<value_type, ublas::column_major> evalJac(
							   const ublas::vector<value_type>& inPoint);


    std::vector<ublas::matrix<value_type, ublas::column_major> > evalHess(
									  const ublas::vector<value_type>& inPoint);

    virtual ~MultiRBFInterpolatorPUH();

  private:
    value_type scale;

    // weighting function object
    BoxWeight omega;
    // storage for interpolators on each patch
    boost::multi_array<MultiRBFInterpolatorH<RTYPE> , inDimT> interpol;

    // bounding box
    ublas::matrix<value_type, ublas::column_major> BB;
    // intervalls in each dimension
    ublas::vector<value_type> h;
    // number of Points per Patch
    unsigned K;
    // overlap as fraction of h
    value_type alpha;
    // overlap in each dimension
    ublas::vector<value_type> o;
    // number of patches in each dimension
    unsigned M;
    // use bump?
    bool useBump;

    // helper variables for evaluation
    ublas::matrix<typename RTYPE::value_type, ublas::column_major> jacGlob;
    ublas::vector<value_type, ublas::bounded_array<value_type, outDimT> > retRBFGlob;
    ublas::vector<value_type, ublas::bounded_array<value_type, outDimT> > retGlob;
    ublas::matrix<value_type, ublas::column_major> diffRetGlob;
    ublas::vector<value_type, ublas::bounded_array<value_type, inDimT> > diffWeightGlob;
    ublas::vector<value_type, ublas::bounded_array<value_type, inDimT> > diffWeightSumGlob;
    ublas::vector<value_type, ublas::bounded_array<value_type, inDimT> > ELGlob;
    ublas::vector<value_type, ublas::bounded_array<value_type, inDimT> > ERGlob;
    bool isBorderGlob;
    value_type retWeightGlob;
    value_type retWeightSumGlob;
    // Index vector for patches
    std::vector<ublas::vector<unsigned> > IGlob;

    // initialize the interpolation object
    void init(ublas::matrix<value_type, ublas::column_major> grid,
	      ublas::matrix<value_type, ublas::column_major> data,
	      std::vector<ublas::matrix<value_type, ublas::column_major> > diffData);
    // compute patch index from inpoint
    std::vector<ublas::vector<unsigned> >
    getIndex(const ublas::vector<value_type>& inPoint) const;
    // recursive function to combine all patch numbers that are computed in getIndex()
    void
    combine(ublas::vector<unsigned, ublas::bounded_array<unsigned,
							 inDimT> >& ind,
	    std::vector<ublas::vector<unsigned> >& Ind,
	    std::vector<ublas::vector<unsigned, ublas::bounded_array<
	      unsigned, 2> > >& I, unsigned d, unsigned& l) const;
    // compute inDim dimensional patch index from linear index
    ublas::vector<unsigned, ublas::bounded_array<unsigned, inDimT> > 
    ind2sub(const unsigned ind) const;
    
#ifndef _WITHOUT_SERIALIZATION_
    // serialization.
    friend class boost::serialization::access;
    template<class Archive>
      void save(Archive & ar, const unsigned int version) const {

      ar & boost::serialization::base_object<MultiNDInterpolator>(*this);

      ar & scale;
      ar & omega;
      ar & BB;
      ar & h;
      ar & K;
      ar & alpha;
      ar & o;
      ar & M;
      ar & useBump;

      ar & jacGlob;
      ar & retRBFGlob;
      ar & retGlob;
      ar & diffRetGlob;
      ar & diffWeightGlob;
      ar & diffWeightSumGlob;
      ar & ELGlob;
      ar & ERGlob;
      ar & isBorderGlob;
      ar & retWeightGlob;
      ar & retWeightSumGlob;
      ar & IGlob;      

      // no serialization for multi_array --> serialize just the content
      for (unsigned i = 0; i < std::pow(M, inDimT); i++) {
	ar & interpol(ind2sub(i));
      }
    }

    template<class Archive>
      void load(Archive & ar, const unsigned int version) {

      ar & boost::serialization::base_object<MultiNDInterpolator>(*this);

      ar & scale;
      ar & omega;
      ar & BB;
      ar & h;
      ar & K;
      ar & alpha;
      ar & o;
      ar & M;
      ar & useBump;

      ar & jacGlob;
      ar & retRBFGlob;
      ar & retGlob;
      ar & diffRetGlob;
      ar & diffWeightGlob;
      ar & diffWeightSumGlob;
      ar & ELGlob;
      ar & ERGlob;
      ar & isBorderGlob;
      ar & retWeightGlob;
      ar & retWeightSumGlob;
      ar & IGlob;

      // no serialization for multi_array --> to load reshape interpol and put in interpolators
      ublas::vector<unsigned, ublas::bounded_array<unsigned, inDimT> > shape(inDimT);
      for (unsigned i = 0; i < inDimT; i++)
	shape(i) = M;
      interpol.resize(shape);
      for (unsigned i = 0; i < std::pow(M, inDimT); i++) {
	ar & interpol(ind2sub(i));
      }
    }

    template<class Archive>
      void serialize(Archive & ar, const unsigned int file_version) {
      boost::serialization::split_member(ar, *this, file_version);
    }
#endif

    bool checkBB(const ublas::vector<value_type> inPoint) {
      bool ret=true;
      for (size_t i = 0; i< inDimT; ++i){
	ret &= inPoint(i) > BB(i,0) && inPoint(i) < BB(i,1);
      }
      return ret;
    }
  };

  template<class RTYPE, std::size_t inDimT, std::size_t outDimT>
    MultiRBFInterpolatorPUH<RTYPE, inDimT, outDimT>::MultiRBFInterpolatorPUH() {
    inDim = 0.;
    outDim = 0.;
    numOfPoints = 0.;
  }

  template<class RTYPE, std::size_t inDimT, std::size_t outDimT>
  MultiRBFInterpolatorPUH<RTYPE, inDimT, outDimT>
  ::MultiRBFInterpolatorPUH(const ublas::matrix<typename RTYPE::value_type, 
						ublas::column_major>& grid,
			    const ublas::matrix<typename RTYPE::value_type, ublas::column_major>& data,
			    const std::vector<ublas::matrix<typename RTYPE::value_type,
							    ublas::column_major> >& diffData,
			    const typename RTYPE::value_type scale, const unsigned K,
			    const typename RTYPE::value_type alpha, const bool useBump) :
      scale(scale), K(K), alpha(alpha), useBump(useBump) {
    // initialize base members
    inDim = static_cast<unsigned>(grid.size1());
    outDim = static_cast<unsigned>(data.size1());
    numOfPoints = static_cast<unsigned>(grid.size2());
    // input check, assume that grid has the right size
    if (data.size2() != numOfPoints)
      throw std::length_error(
			      "MultiRBFInterpolatorPUH::Constructor: Number of interpolation points (grid.size2()) and number of data points (data.size()) do not match.");

    if (diffData.size() != numOfPoints)
      throw std::length_error(
			      "MultiRBFInterpolatorH::Constructor: Number of jacobians (diffData.size()) does not match number of centers (numOfPoints).");

    for (unsigned i = 0; i < numOfPoints; i++) {
      if (diffData[i].size1() != outDimT)
	throw std::length_error(
				"MultiRBFInterpolatorH::Constructor: Number of rows of jacobian	<< i << does not match outDim.");
      if (diffData[i].size2() != inDimT)
	throw std::length_error(
				"MultiRBFInterpolatorH::Constructor: Number of columns of jacobian << i << does not match inDim");
    }

    assert(scale>0);
    assert(alpha>0 && alpha<.5);

    // resize some objects needed for evaluation
    retRBFGlob.resize(outDimT);
    retGlob.resize(outDimT);
    diffRetGlob.resize(outDimT, inDimT);
    diffWeightGlob.resize(inDimT);
    diffWeightSumGlob.resize(inDimT);
    ELGlob.resize(inDimT);
    ERGlob.resize(inDimT);
    jacGlob.resize(outDimT, inDimT);

    // initialize the interpolation object
    init(grid, data, diffData);
  }

  /**
   * \brief an alternative assignment member
   */
  template<class RTYPE, std::size_t inDimT, std::size_t outDimT>
    inline void MultiRBFInterpolatorPUH<RTYPE, inDimT, outDimT>::assign(
									const ublas::matrix<typename RTYPE::value_type, ublas::column_major>& g,
									const ublas::matrix<typename RTYPE::value_type, ublas::column_major>& d,
									const std::vector<ublas::matrix<typename RTYPE::value_type,
									ublas::column_major> >& dD,
									const typename RTYPE::value_type& s, const unsigned& K_,
									const typename RTYPE::value_type& alpha_) {

    inDim = g.size1();
    outDim = d.size1();
    numOfPoints = g.size2();
    scale = s;
    K = K_;
    alpha = alpha_;

    // input check, assume that grid has the right size
    if (d.size2() != numOfPoints)
      std::cerr << "ERROR: MultiRBFInterpolatorH::Constructor: Number of interpolation points (grid.size2()) " 
		<< "and number of data points (data.size()) do not match." << std::endl;

    if (dD.size() != numOfPoints)
      std::cerr << "ERROR: MultiRBFInterpolatorH::Constructor: Number of jacobians " 
		<< "(diffData.size()) does not match number of centers (numOfPoints)." << std::endl;

    for (unsigned i = 0; i < numOfPoints; i++) {
      if (dD[i].size1() != outDim)
	std::cerr << "ERROR: MultiRBFInterpolatorH::Constructor: Number of rows of jacobian "
		  << i << " does not match outDim." << std::endl;
      if (dD[i].size2() != inDimT)
	std::cerr << "ERROR: MultiRBFInterpolatorH::Constructor: Number of columns of jacobian "
		  << i << " does not match inDim" << std::endl;
    }

    // init the shit out of it.
    assert(scale>0);
    assert(alpha>0 && alpha<.5);

    // resize some objects needed for evaluation
    retRBFGlob.resize(outDimT);
    retGlob.resize(outDimT);
    diffRetGlob.resize(outDimT, inDimT);
    diffWeightGlob.resize(inDimT);
    diffWeightSumGlob.resize(inDimT);
    ELGlob.resize(inDimT);
    ERGlob.resize(inDimT);
    jacGlob.resize(outDimT, inDimT);

    init(g, d, dD);
  }

  template<class RTYPE, std::size_t inDimT, std::size_t outDimT>
    inline ublas::vector<typename RTYPE::value_type> MultiRBFInterpolatorPUH<RTYPE,
    inDimT, outDimT>::eval(const ublas::vector<typename RTYPE::value_type>& inPoint) {

    // check if point is inside boundary if Debug mode is on.
#ifndef NDEBUG
    if (!checkBB(inPoint)) {
      std::cerr << "ERROR: NDInterpolator::eval(): inPoint outside of bounding box:" << std::endl;
      std::cerr << "ERROR: inPoint: " << inPoint << std::endl;
      
      for (unsigned i = 0; i < outDimT; i++)
	retGlob(i) = 0.;

      return retGlob;
    }
#endif

    for (unsigned i = 0; i < outDimT; i++)
      retGlob(i) = 0.;
    
    retWeightGlob = 0;
    retWeightSumGlob = 0;
    isBorderGlob = false;
    
    std::vector<ublas::vector<unsigned> > I = getIndex(inPoint);
    for (unsigned i = 0; i < I.size(); i++) {
      // compute edges of patch
      for (unsigned j = 0; j < inDimT; j++) {
    	ELGlob(j) = BB(j, 0) - o(j) + I[i](j) * (h(j) - alpha * h(j));
    	ERGlob(j) = BB(j, 0) - o(j) + h(j) + I[i](j)
    	  * (h(j) - alpha * h(j));
    	if (compFloat<typename RTYPE::value_type> (ELGlob(j), inPoint(j))
    	    || compFloat<typename RTYPE::value_type> (ERGlob(j),
    						      inPoint(j)))
    	  isBorderGlob = true;
      }
      if (!isBorderGlob) {
    	retWeightGlob = omega.eval(inPoint, ELGlob, ERGlob);
    	retWeightSumGlob += retWeightGlob;
    	retGlob += interpol(I[i]).eval(inPoint) * retWeightGlob;
      } else
    	isBorderGlob = false;
    }
    
    return retGlob / retWeightSumGlob;
  }


  template<class RTYPE, std::size_t inDimT, std::size_t outDimT>
    inline ublas::vector<typename RTYPE::value_type> MultiRBFInterpolatorPUH<RTYPE,
    inDimT, outDimT>::evalDiff(
			       const ublas::vector<typename RTYPE::value_type>& inPoint,
			       const unsigned alphaDiff) {

    ublas::vector<typename RTYPE::value_type, ublas::bounded_array<
      typename RTYPE::value_type, outDimT> > retRBF(outDimT);
    ublas::vector<typename RTYPE::value_type, ublas::bounded_array<
      typename RTYPE::value_type, outDimT> > ret(outDimT);
    ublas::vector<typename RTYPE::value_type, ublas::bounded_array<
      typename RTYPE::value_type, outDimT> > diffRet(outDimT);

#ifndef NDEBUG
    if (!checkBB(inPoint)) {
      std::cerr << "ERROR: NDInterpolator::eval(): inPoint outside of bounding box:" << std::endl;
      std::cerr << "ERROR: inPoint: " << inPoint << std::endl;
      
      for (unsigned i = 0; i < outDimT; i++) {
	ret(i) = 0;
      }
      return ret;
    }
#endif

    for (unsigned i = 0; i < outDimT; i++) {
      ret(i) = 0;
      diffRet(i) = 0;
    }

    typename RTYPE::value_type retWeight = 0.;
    typename RTYPE::value_type retWeightSum = 0;

    typename RTYPE::value_type diffWeight = 0.;
    typename RTYPE::value_type diffWeightSum = 0;

    std::vector<ublas::vector<unsigned> > I = getIndex(inPoint);
    ublas::vector<typename RTYPE::value_type, ublas::bounded_array<
      typename RTYPE::value_type, inDimT> > EL(inDimT);
    ublas::vector<typename RTYPE::value_type, ublas::bounded_array<
      typename RTYPE::value_type, inDimT> > ER(inDimT);
    bool isBorder;

    for (unsigned i = 0; i < I.size(); i++) {
      // compute edges of patch
      for (unsigned j = 0; j < inDimT; j++) {
	EL(j) = BB(j, 0) - o(j) + I[i](j) * (h(j) - alpha * h(j));
	ER(j) = BB(j, 0) - o(j) + h(j) + I[i](j) * (h(j) - alpha * h(j));
	if (compFloat<typename RTYPE::value_type> (EL(j), inPoint(j))
	    || compFloat<typename RTYPE::value_type> (ER(j), inPoint(j)))
	  isBorder = true;
      }
      if (!isBorder) {
	retWeight = omega.eval(inPoint, EL, ER);
	diffWeight = omega.evalDiff(inPoint, EL, ER, alphaDiff);

	diffWeightSum += diffWeight;
	retWeightSum += retWeight;

	retRBF = interpol(I[i]).eval(inPoint);

	ret += retRBF * retWeight;
	diffRet += interpol(I[i]).evalDiff(inPoint, alphaDiff)
	  * retWeight + retRBF * diffWeight;
      } else
	isBorder = false;
    }

    return (-diffWeightSum * ret / retWeightSum + diffRet) / retWeightSum;
  }


  template<class RTYPE, std::size_t inDimT, std::size_t outDimT>
    inline ublas::vector<typename RTYPE::value_type> MultiRBFInterpolatorPUH<RTYPE,
    inDimT, outDimT>::evalDiff2(
				const ublas::vector<typename RTYPE::value_type>& inPoint,
				const unsigned alphaDiff) {

    ublas::vector<typename RTYPE::value_type> retRBF(outDimT);
    ublas::vector<typename RTYPE::value_type> diffRBF(outDimT);

    ublas::vector<typename RTYPE::value_type> ret(outDimT);
    ublas::vector<typename RTYPE::value_type> diffRet(outDimT);
    ublas::vector<typename RTYPE::value_type> diff2Ret(outDimT);
    // initialize because they are used in sums
    for (unsigned i = 0; i < outDimT; i++) {
      ret(i) = 0;
      diffRet(i) = 0;
      diff2Ret(i) = 0;
    }

    typename RTYPE::value_type retWeight = 0.;
    typename RTYPE::value_type retWeightSum = 0;

    typename RTYPE::value_type diffWeight = 0.;
    typename RTYPE::value_type diffWeightSum = 0;

    typename RTYPE::value_type diff2Weight = 0.;
    typename RTYPE::value_type diff2WeightSum = 0.;

    bool isBorder;

    std::vector<ublas::vector<unsigned> > I = getIndex(inPoint);
    ublas::vector<typename RTYPE::value_type> EL(inDimT);
    ublas::vector<typename RTYPE::value_type> ER(inDimT);

    for (unsigned i = 0; i < I.size(); i++) {
      // compute edges of patch
      for (unsigned j = 0; j < inDimT; j++) {
	EL(j) = BB(j, 0) - o(j) + I[i](j) * (h(j) - alpha * h(j));
	ER(j) = BB(j, 0) - o(j) + h(j) + I[i](j) * (h(j) - alpha * h(j));
	if (compFloat<typename RTYPE::value_type> (EL(j), inPoint(j))
	    || compFloat<typename RTYPE::value_type> (ER(j), inPoint(j)))
	  isBorder = true;
      }
      if (!isBorder) {
	retWeight = omega.eval(inPoint, EL, ER);
	diffWeight = omega.evalDiff(inPoint, EL, ER, alphaDiff);
	diff2Weight = omega.evalDiff2(inPoint, EL, ER, alphaDiff);

	retWeightSum += retWeight;
	diffWeightSum += diffWeight;
	diff2WeightSum += diff2Weight;

	retRBF = interpol(I[i]).eval(inPoint);
	diffRBF = interpol(I[i]).evalDiff(inPoint, alphaDiff);

	ret += retRBF * retWeight;
	diffRet += diffRBF * retWeight + retRBF * diffWeight;
	diff2Ret += interpol(I[i]).evalDiff2(inPoint, alphaDiff)
	  * retWeight + 2 * diffRBF * diffWeight + retRBF
	  * diff2Weight;
      } else
	isBorder = false;
    }

    return ((diffWeightSum * (-2 * diffRet + 2 * ret * diffWeightSum
			      / retWeightSum) - ret * diff2WeightSum) / retWeightSum + diff2Ret)
      / retWeightSum;
  }

  template<class RTYPE, std::size_t inDimT, std::size_t outDimT>
    inline ublas::vector<typename RTYPE::value_type> MultiRBFInterpolatorPUH<RTYPE,
    inDimT, outDimT>::evalDiffMixed(
				    const ublas::vector<typename RTYPE::value_type>& inPoint,
				    const unsigned alpha1, const unsigned alpha2) {

    ublas::vector<typename RTYPE::value_type> retRBF(outDimT);
    ublas::vector<typename RTYPE::value_type> diffRBF1(outDimT);
    ublas::vector<typename RTYPE::value_type> diffRBF2(outDimT);

    ublas::vector<typename RTYPE::value_type> ret(outDimT);
    ublas::vector<typename RTYPE::value_type> diffRet1(outDimT);
    ublas::vector<typename RTYPE::value_type> diffRet2(outDimT);
    ublas::vector<typename RTYPE::value_type> diffMRet(outDimT);
    // initizalze because used in sums
    for (unsigned i = 0; i < outDimT; i++) {
      ret(i) = 0.;
      diffRet1(i) = 0.;
      diffRet2(i) = 0.;
      diffMRet(i) = 0.;
    }

    typename RTYPE::value_type retWeight = 0.;
    typename RTYPE::value_type retWeightSum = 0.;

    typename RTYPE::value_type diffWeight1 = 0.;
    typename RTYPE::value_type diffWeightSum1 = 0;

    typename RTYPE::value_type diffWeight2 = 0.;
    typename RTYPE::value_type diffWeightSum2 = 0.;

    typename RTYPE::value_type diffMWeight = 0.;
    typename RTYPE::value_type diffMWeightSum = 0.;

    bool isBorder;

    std::vector<ublas::vector<unsigned> > I = getIndex(inPoint);
    ublas::vector<typename RTYPE::value_type> EL(inDimT);
    ublas::vector<typename RTYPE::value_type> ER(inDimT);

    for (unsigned i = 0; i < I.size(); i++) {
      // compute edges of patch
      for (unsigned j = 0; j < inDimT; j++) {
	EL(j) = BB(j, 0) - o(j) + I[i](j) * (h(j) - alpha * h(j));
	ER(j) = BB(j, 0) - o(j) + h(j) + I[i](j) * (h(j) - alpha * h(j));
	if (compFloat<typename RTYPE::value_type> (EL(j), inPoint(j))
	    || compFloat<typename RTYPE::value_type> (ER(j), inPoint(j)))
	  isBorder = true;
      }
      if (!isBorder) {
	retWeight = omega.eval(inPoint, EL, ER);
	diffWeight1 = omega.evalDiff(inPoint, EL, ER, alpha1);
	diffWeight2 = omega.evalDiff(inPoint, EL, ER, alpha2);
	diffMWeight = omega.evalDiffMixed(inPoint, EL, ER, alpha1, alpha2);

	retWeightSum += retWeight;
	diffWeightSum1 += diffWeight1;
	diffWeightSum2 += diffWeight2;
	diffMWeightSum += diffMWeight;

	retRBF = interpol(I[i]).eval(inPoint);
	diffRBF1 = interpol(I[i]).evalDiff(inPoint, alpha1);
	diffRBF2 = interpol(I[i]).evalDiff(inPoint, alpha2);

	ret += retRBF * retWeight;
	diffRet1 += diffRBF1 * retWeight + retRBF * diffWeight1;
	diffRet2 += diffRBF2 * retWeight + retRBF * diffWeight2;
	diffMRet += diffRBF1 * diffWeight2 + diffRBF2 * diffWeight1
	  + interpol(I[i]).evalDiffMixed(inPoint, alpha1, alpha2)
	  * retWeight + retRBF * diffMWeight;
      } else
	isBorder = false;
    }

    return (((2 * ret * diffWeightSum1 * diffWeightSum2) / retWeightSum
	     - diffWeightSum2 * diffRet1 - diffWeightSum1 * diffRet2 + diffMRet
	     * retWeightSum - ret * diffMWeightSum) / retWeightSum)
      / retWeightSum;
  }

  template<class RTYPE, std::size_t inDimT, std::size_t outDimT>
    inline ublas::vector<typename RTYPE::value_type> MultiRBFInterpolatorPUH<RTYPE,
    inDimT, outDimT>::evalGrad(
			       const ublas::vector<typename RTYPE::value_type>& inPoint,
			       const unsigned al) {

    assert(al > 0 && al < outDimT);
    return ublas::row(evalJac(inPoint), al - 1);
  }

  template<class RTYPE, std::size_t inDimT, std::size_t outDimT>
    inline ublas::matrix<typename RTYPE::value_type, ublas::column_major> MultiRBFInterpolatorPUH<
    RTYPE, inDimT, outDimT>::evalJac(
				     const ublas::vector<typename RTYPE::value_type>& inPoint) {

#ifndef NDEBUG
    if (!checkBB(inPoint)) {
      std::cerr << "ERROR: NDInterpolator::eval(): inPoint outside of bounding box:" << std::endl;
      std::cerr << "ERROR: inPoint: " << inPoint << std::endl;
      
      for (unsigned i = 0; i < outDimT; i++) {
	for (unsigned j = 0; j < inDimT; j++) {
	  diffRetGlob(i, j) = 0;
	}
      }
      return diffRetGlob;
    }
#endif

    // used in sum
    //@todo diffWeightSumGlob initialization
    for (unsigned i = 0; i < outDimT; i++) {
      retGlob(i) = 0;
      for (unsigned j = 0; j < inDimT; j++) {
	diffRetGlob(i, j) = 0;
	diffWeightSumGlob(j) = 0;
      }
    }
    isBorderGlob = false;
    retWeightGlob = 0;
    retWeightSumGlob = 0;

    // for combined jacobian and point evaluation temp vars are needed :(
    //	ublas::vector<typename RTYPE::value_type, ublas::bounded_array<
    //			typename RTYPE::value_type, outDimT> > retRBFLoc(outDimT);
    ublas::vector<typename RTYPE::value_type> retRBFLoc(outDimT);
    ublas::matrix<typename RTYPE::value_type, ublas::column_major> diffRetLoc(
									      outDimT, inDimT);

    // in which patches does inpoint lie
    std::vector<ublas::vector<unsigned> > I = getIndex(inPoint);

    // evaluate in patches and weight with weighting function and derivatives
    for (unsigned i = 0; i < I.size(); i++) {
      // compute edges of patch
      for (unsigned j = 0; j < inDimT; j++) {
	ELGlob(j) = BB(j, 0) - o(j) + I[i](j) * (h(j) - alpha * h(j));
	ERGlob(j) = BB(j, 0) - o(j) + h(j) + I[i](j)
	  * (h(j) - alpha * h(j));
	if (compFloat<typename RTYPE::value_type> (ELGlob(j), inPoint(j))
	    || compFloat<typename RTYPE::value_type> (ERGlob(j),
						      inPoint(j)))
	  isBorderGlob = true;
      }
      if (!isBorderGlob) {

	retWeightGlob = omega.eval(inPoint, ELGlob, ERGlob);
	retWeightSumGlob += retWeightGlob;

	// for performance evaluate for outpoint and jac in one function call
	interpol(I[i]).evalFuncAndJac(inPoint, retRBFLoc, diffRetLoc);

	retGlob = retGlob + retRBFLoc * retWeightGlob;
	diffRetGlob = diffRetGlob + diffRetLoc * retWeightGlob;
	for (unsigned k = 0; k < inDimT; k++) {
	  diffWeightGlob(k) = omega.evalDiff(inPoint, ELGlob, ERGlob,
					     k + 1);
	  diffWeightSumGlob(k) += diffWeightGlob(k);
	  for (unsigned l = 0; l < outDimT; l++) {
	    diffRetGlob(l, k) += diffWeightGlob(k) * retRBFLoc(l);
	  }
	}

      } else
	isBorderGlob = false;
    }

    for (unsigned k = 0; k < outDimT; k++) {
      retGlob(k) /= retWeightSumGlob;
      for (unsigned l = 0; l < inDimT; l++) {
	diffRetGlob(k, l) += -diffWeightSumGlob(l) * retGlob(k);
	diffRetGlob(k, l) /= retWeightSumGlob;
      }
    }
    return diffRetGlob;
  }


  template<class RTYPE, std::size_t inDimT, std::size_t outDimT>
    inline std::vector<ublas::matrix<typename RTYPE::value_type,
    ublas::column_major> > MultiRBFInterpolatorPUH<RTYPE, inDimT, outDimT>::evalHess(
										     const ublas::vector<typename RTYPE::value_type>& inPoint) {

    std::vector<ublas::matrix<typename RTYPE::value_type, ublas::column_major> >
      ret(outDimT);
    for (unsigned i = 0; i < outDimT; i++) {
      ret[i].resize(inDimT, inDimT);
    }
    ublas::vector<typename RTYPE::value_type> retVec(outDimT);

    for (unsigned i = 0; i < inDimT; i++) {
      for (unsigned j = i + 1; j < inDimT; j++) {
	retVec = this->evalDiffMixed(inPoint, i + 1, j + 1);
	for (unsigned k = 0; k < outDimT; k++)
	  ret[k](i, j) = retVec(k);
      }
      retVec = this->evalDiff2(inPoint, i + 1);
      for (unsigned k = 0; k < outDimT; k++)
	ret[k](i, i) = retVec(k);
    }

    return ret;
  }

  template<class RTYPE, std::size_t inDimT, std::size_t outDimT>
    void MultiRBFInterpolatorPUH<RTYPE, inDimT, outDimT>::init(
							       ublas::matrix<typename RTYPE::value_type, ublas::column_major> grid,
							       ublas::matrix<typename RTYPE::value_type, ublas::column_major> data,
							       std::vector<ublas::matrix<typename RTYPE::value_type,
							       ublas::column_major> > diffData) {

    /*
     * basic steps of the initialisation:
     * 1. compute bounding box
     * 2. Smooth data (optional)
     * 3. compute all kind of paramters:
     * 		- delta_i diameter in dimension i
     * 		- MSnake number of patches if directly adjacent
     * 		- o_i overlap in dimension i
     * 		- M number of patches with overlap
     * 4. run trough points and determine indices of patches they belong to
     * 5. initialize RBF interpolators for each patch in multiarray *
     */
    // 1. and 2. some parameters that already can be used in the loop
    unsigned MTilde = static_cast<unsigned>(std::floor(std::pow(numOfPoints / K, 1. / inDimT)));
    if (MTilde == 0)
      MTilde = 1;

    M = static_cast<unsigned>(std::ceil((MTilde - alpha) / (1. - alpha)));
    // shape of multi_array, needed later, but can be initialized in the loop
    ublas::vector<unsigned, ublas::bounded_array<unsigned, inDimT> > shape(
									   inDimT);
    // vector of row indices for grid, needed later but can be initialized in the loop
    ublas::vector<unsigned, ublas::bounded_array<unsigned, inDimT> > gridRows(
									      inDimT);
    // Bounding Box
    BB.resize(inDimT, 2);
    h.resize(inDimT);
    o.resize(inDimT);
    /* Enlarge the bounding box by a little to avoid floating point weirdness later, e.g.
     * differences between BB and points are calculated and rounding errors might lead to
     * the situation (already in init) that a point seems to be outside of the BB although 
     * it is right on the border
     */
    for (unsigned i = 0; i < inDimT; i++) {
      BB(i, 0) = *std::min_element(ublas::row(grid, i).begin(),
				   ublas::row(grid, i).end());
      BB(i, 0) = std::nextafter(BB(i,0),
			       (double) -std::numeric_limits<typename RTYPE::value_type>::infinity());
      
      BB(i, 1) = *std::max_element(ublas::row(grid, i).begin(),
				   ublas::row(grid, i).end());
      BB(i, 1) = std::nextafter(BB(i,1),
			       (double) std::numeric_limits<typename RTYPE::value_type>::infinity());
	// if just one patch, then the border of the patch is widened a little to prevent
      // points lying on the edge of the patch
      if (M == 1) {
	h(i) = 1.1 * (BB(i, 1) - BB(i, 0)) / MTilde;
      } else {
	h(i) = (BB(i, 1) - BB(i, 0)) / MTilde;
      }
      o(i) = (M * (1 - alpha) - MTilde + alpha) * h(i) / 2.;
      if (o(i)<0)
	o(i)=0;

      // use for loop to fill shape vector for later
      shape(i) = M;
      // use for loop to fill row indices
      gridRows(i) = i;
    }

    if (useBump) {
      std::cout << "NDInterpolator::WARNING: Using Bump Function" << std::endl;
      Bump<typename RTYPE::value_type> bump(BB);
      bump.applyToDiffAndData(grid,data,diffData);
    }
    // 3. get indices of patches for each grid point
    //temporary storage for patch indices as output of getIndex()
    std::vector<ublas::vector<unsigned> > I(inDimT);

    // multiDim Storage for column indices
    boost::multi_array < ublas::vector<unsigned>, inDimT > Columns(shape);
    size_t tempSize = 0;
    for (unsigned i = 0; i < numOfPoints; i++) {
      I = getIndex(ublas::column(grid, i));
      for (unsigned j = 0; j < I.size(); j++) {
	// add i to the patches, which indices are collected in I
	tempSize = Columns(I[j]).size();
	Columns(I[j]).resize(tempSize + 1, true);
	Columns(I[j])(tempSize) = i; // 0 based indexing --> current index is tempSize
      }
    }

    // 4. initialize Interpolators
    // vector of row indices for data
    ublas::vector<unsigned, ublas::bounded_array<unsigned, outDimT> > dataRows(outDimT);
    for (unsigned i = 0; i < outDimT; i++)
      dataRows(i) = i;
    // index for dataRows
    ublas::indirect_array < ublas::vector<unsigned> > dataR(outDimT, dataRows);
    // index for gridRows
    ublas::indirect_array < ublas::vector<unsigned> > gridR(inDimT, gridRows);

    // resize interpol multi_array
    interpol.resize(shape);
    unsigned nPatches = static_cast<unsigned> (boost::math::pow<inDimT>(M));
    for (unsigned i = 0; i < nPatches; i++) {

      ublas::indirect_array < ublas::vector<unsigned> > C(
							  Columns(ind2sub(i)).size(), Columns(ind2sub(i)));
      // put interpolator in patches
      // attack here --> can't use the same R for
      // grid and data
      interpol(ind2sub(i)) = MultiRBFInterpolatorH<RTYPE> (
							   ublas::matrix_indirect<ublas::matrix<
							   typename RTYPE::value_type, ublas::column_major>,
							   ublas::indirect_array<ublas::vector<unsigned> > >(grid,
													     gridR, C),
							   ublas::matrix_indirect<ublas::matrix<
							   typename RTYPE::value_type, ublas::column_major>,
							   ublas::indirect_array<ublas::vector<unsigned> > >(data,
													     dataR, C), stdVectorIndirect(diffData, C), scale);
    }
  }

  /*
   * Compute the index of all patches a point lies in.
   */
  template<class RTYPE, std::size_t inDimT, std::size_t outDimT>
    inline std::vector<ublas::vector<unsigned> > MultiRBFInterpolatorPUH<RTYPE,
    inDimT, outDimT>::getIndex(
			       const ublas::vector<typename RTYPE::value_type>& inPoint) const {

    // initialize some temporary variables
    std::vector<ublas::vector<unsigned, ublas::bounded_array<unsigned, 2> > >
      ret(inDimT);
    // temporary vector of patch numbers
    ublas::vector<unsigned, ublas::bounded_array<unsigned, 2> > tempPatch(2);
    // temporary fractioned patch number --> converted to unsigned
    typename RTYPE::value_type tempL;
    typename RTYPE::value_type tempR;
    // counter --> how many patch indices do we have
    unsigned l = 1;

    for (unsigned i = 0; i < inDimT; i++) {
      /*
       * clean output
       * case 1:
       * if < 0 or > M-1, tempPatch(0) can't be smaller than 0
       * and tempPatch(1) can't be larger than M
       * case 2:
       * both entries are the same
       */
      // left edge
      tempL = (o(i) - BB(i, 0) + inPoint(i)) / (h(i) - alpha * h(i));
      if (tempL > M - 1)
	tempPatch(0) = M - 1;
      else
	tempPatch(0) = static_cast<unsigned> (std::floor(tempL));

      // right edge
      tempR = (o(i) - BB(i, 0) - h(i) + inPoint(i)) / (h(i) - alpha * h(i));
      if (tempR < 0)
	tempPatch(1) = 0;
      else
	tempPatch(1) = static_cast<unsigned> (std::ceil(tempR));

      // test for same element.
      if (tempPatch(0) == tempPatch(1)) {
	tempPatch.resize(1, true);
	ret[i] = tempPatch;
	// resize to have right size again
	tempPatch.resize(2, false);
      } else {
	ret[i] = tempPatch;
	// update counter only if two patches are added
	l = 2 * l;
      }
    }

    // prepare to call combine
    ublas::vector<unsigned, ublas::bounded_array<unsigned, inDimT> >
      ind(inDimT);
    ublas::vector<unsigned, ublas::bounded_array<unsigned, inDimT> >& indRef =
      ind;
    std::vector<ublas::vector<unsigned> > Ind(l);
    // reset l
    l = 0;
    unsigned& lRef = l;
    std::vector<ublas::vector<unsigned> > &IndRef = Ind;
    std::vector<ublas::vector<unsigned, ublas::bounded_array<unsigned, 2> > >
      &IRef = ret;
    // recursively generate indices
    combine(indRef, IndRef, IRef, 0, lRef);

    return Ind;
  }

  /*
   * recursively combine all individual patches
   */
  template<class RTYPE, std::size_t inDimT, std::size_t outDimT>
    inline void MultiRBFInterpolatorPUH<RTYPE, inDimT, outDimT>::combine(
									 ublas::vector<unsigned, ublas::bounded_array<unsigned, inDimT> >& ind,
									 std::vector<ublas::vector<unsigned> >& Ind,
									 std::vector<ublas::vector<unsigned, ublas::bounded_array<unsigned, 2> > >& I,
									 unsigned d, unsigned& l) const {
    /*
     * recursively generate all combinations.
     * for each dimension travel along the patches (i runs max to 1)
     * recursion ends if last dimension is reached --> put index vector into large storage for index
     * vectors.
     * d starts at 0 !!
     * l starts at 0 !! (overall counter)
     */
    for (unsigned i = 0; i < I[d].size(); i++) {
      ind(d) = I[d](i);
      if (d < inDimT - 1)
	combine(ind, Ind, I, d + 1, l);
      else {
	Ind[l] = ind;
	l++;
      }
    }
  }

  /*
   * compute multi_array index vector from linear index
   */
  template<class RTYPE, std::size_t inDimT, std::size_t outDimT>
    inline ublas::vector<unsigned, ublas::bounded_array<unsigned, inDimT> > MultiRBFInterpolatorPUH<
    RTYPE, inDimT, outDimT>::ind2sub(unsigned ind) const {
    ublas::vector<unsigned, ublas::bounded_array<unsigned, inDimT> >
      ret(inDimT);
    //unsigned MExp = std::pow(M, inDimT - 1);
    unsigned MExp = static_cast<unsigned>(boost::math::pow<inDimT-1>(M));
    for (int i = inDimT - 1; i >= 0; i--) {
      ret(i) = (ind - (ind % MExp)) / MExp;
      ind = ind - MExp * ret(i);
      MExp /= M;
    }
    return ret;
  }

  template<class RTYPE, std::size_t inDimT, std::size_t outDimT>
    MultiRBFInterpolatorPUH<RTYPE, inDimT, outDimT>::~MultiRBFInterpolatorPUH() {
  }
  
}

#endif /* MULTIRBFINTERPOLATORPUH_H_ */
