/**
 * RBFInterpolatorPUH.h
 *
 *  Created on: \date Dec 15, 2010
 *      Author: \author mrehberg
 */

#ifndef RBFINTERPOLATORPUH_H_
#define RBFINTERPOLATORPUH_H_

// std includes
#include <cassert>
#include <algorithm>
#include <vector>
#include <cstddef>
// boost includes
#include "boost/multi_array.hpp"

#include "boost/numeric/ublas/vector_proxy.hpp"
// NDInterpolator includes
#include "../NDInterpolator.hpp"
#include "RBFInterpolatorH.hpp"
#include "BoxWeight.hpp"

namespace NDInterpolator {
/** \brief Hermite RBF interpolator implementing a partition of unity approach for domain space
 * separation.
 * \tparam RTYPE Type of radial basis function to be used, e.g. NDInterpolator::GaussianRBF<double>
 * \tparam inDimT Dimension of the domain space (input dimension).
 */
template<class RTYPE, std::size_t inDimT>
class RBFInterpolatorPUH: virtual public NDInterpolator {

public:
	typedef typename RTYPE::value_type value_type;

	RBFInterpolatorPUH();
	RBFInterpolatorPUH(
			const ublas::matrix<value_type, ublas::column_major>& grid,
			const ublas::vector<value_type>& data,
			const ublas::matrix<value_type, ublas::column_major>& diffData,
			const value_type scale = 1., const unsigned K = 100,
			const value_type alpha = .33);

	/**
	 * Try to calculate the optimal scaling parameter for the interpolator.
	 * @param tol Termination tolerance of the optimization algorithm. The calculated scale
	 * is used for all computations after a call to this method. For each Patch a optimization is
	 * performed independently.
	 */
	void optimizeScale(const double tol = 1e-12) {
		for (unsigned i = 0; i < std::pow(M, inDimT); i++) {
			interpol(ind2sub(i)).optimizeScale(tol);
		}
	}

	value_type eval(const ublas::vector<value_type>& inPoint) const;

	value_type evalDiff(const ublas::vector<value_type>& inPoint,
			const unsigned int alpha) const;

	value_type
			evalDiff2(const ublas::vector<value_type>& inPoint,
					const unsigned alpha) const;

	value_type evalDiffMixed(const ublas::vector<value_type>& inPoint,
			const unsigned alpha1, const unsigned alpha2) const;

	ublas::vector<value_type>
	evalGrad(const ublas::vector<value_type>& inPoint) const;

	ublas::matrix<value_type>
	evalHess(const ublas::vector<value_type>& inPoint) const;

	virtual ~RBFInterpolatorPUH();

private:
	value_type scale;

	// weighting function object
	BoxWeight omega;
	// storage for interpolators on each patch
	boost::multi_array<RBFInterpolatorH<RTYPE> , inDimT> interpol;

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

	// initialize the interpolation object
	void init(ublas::matrix<value_type, ublas::column_major> grid,
			ublas::vector<value_type> data,
			ublas::matrix<value_type, ublas::column_major> diffData);
	// compute patch index from inpoint
	std::vector<ublas::vector<unsigned> >
	getIndex(const ublas::vector<value_type>& inPoint) const;
	// recursive function to combine all patch numbers that are computed in getIndex()
	void
	combine(
			ublas::vector<unsigned>& ind,
			std::vector<ublas::vector<unsigned> >& Ind,
			std::vector<ublas::vector<unsigned, ublas::bounded_array<unsigned,
					2> > >& I, unsigned d, unsigned& l) const;
	// compute inDim dimensional patch index from linear index
	ublas::vector<unsigned, ublas::bounded_array<unsigned, inDimT> > ind2sub(
			const
			unsigned ind) const;

#ifndef _WITHOUT_SERIALIZATION_
	// serialization.
	friend class boost::serialization::access;
	template<class Archive>
	void save(Archive & ar, const unsigned int version) const {

		ar & boost::serialization::base_object<NDInterpolator>(*this);

		ar & scale;
		ar & omega;
		ar & BB;
		ar & h;
		ar & K;
		ar & alpha;
		ar & o;
		ar & M;
		// no serialization for multi_array --> serialize just the content
		for (unsigned i = 0; i < std::pow(M, inDimT); i++) {
			ar & interpol(ind2sub(i));
		}
	}

	template<class Archive>
	void load(Archive & ar, const unsigned int version) {

		ar & boost::serialization::base_object<NDInterpolator>(*this);

		ar & scale;
		ar & omega;
		ar & BB;
		ar & h;
		ar & K;
		ar & alpha;
		ar & o;
		ar & M;
		// no serialization for multi_array --> to load reshape interpol and put in interpolators
		ublas::vector<unsigned, ublas::bounded_array<unsigned, inDimT> > shape(
				inDimT);
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
};

template<class RTYPE, std::size_t inDimT>
RBFInterpolatorPUH<RTYPE, inDimT>::RBFInterpolatorPUH() {
	inDim = 0;
	numOfPoints = 0;
}

template<class RTYPE, std::size_t inDimT>
RBFInterpolatorPUH<RTYPE, inDimT>::RBFInterpolatorPUH(
		const ublas::matrix<typename RTYPE::value_type, ublas::column_major>& grid,
		const ublas::vector<typename RTYPE::value_type>& data,
		const ublas::matrix<typename RTYPE::value_type, ublas::column_major>& diffData,
		const typename RTYPE::value_type scale, const unsigned K,
		const typename RTYPE::value_type alpha) :
	scale(scale), K(K), alpha(alpha) {
	// initialize base members
	inDim = grid.size1();
	numOfPoints = grid.size2();
	// input check, assume that grid has the right size
	if (data.size() != numOfPoints)
		throw std::length_error(
				"RBFInterpolatorPUH::Constructor: Number of interpolation points (grid.size2()) and number of data points (data.size()) do not match.");

	if (diffData.size1() != inDim || diffData.size2() != numOfPoints)
		throw std::length_error(
				"RBFInterpolatorPUH::Constructor: Number of columns of diffData does not match number of interpolation points in grid or number of rows of diffData does not match inDim");

	assert(scale>0);
	assert(alpha>0 && alpha<.5);

	// initialize the interpolation object
	init(grid, data, diffData);
}

template<class RTYPE, std::size_t inDimT>
inline typename RTYPE::value_type RBFInterpolatorPUH<RTYPE, inDimT>::eval(
		const ublas::vector<typename RTYPE::value_type>& inPoint) const {
	typename RTYPE::value_type ret = 0.;
	typename RTYPE::value_type retWeight = 0.;
	typename RTYPE::value_type retWeightSum = 0;
	std::vector<ublas::vector<unsigned> > I = getIndex(inPoint);
	ublas::vector<typename RTYPE::value_type> EL(inDimT);
	ublas::vector<typename RTYPE::value_type> ER(inDimT);
	bool isBorder = false;
	for (unsigned i = 0; i < I.size(); i++) {
		// compute edges of patch
		for (unsigned j = 0; j < inDimT; j++) {
			EL(j) = BB(j, 0) - o(j) + I[i](j) * (h(j) - alpha * h(j));
			ER(j) = BB(j, 0) - o(j) + h(j) + I[i](j) * (h(j) - alpha * h(j));
			if (EL(j) == inPoint(j) || ER(j) == inPoint(j))
				isBorder = true;
		}
		if (!isBorder) {
			retWeight = omega.eval(inPoint, EL, ER);
			retWeightSum += retWeight;
			ret += interpol(I[i]).eval(inPoint) * retWeight;
		} else
			isBorder = false;
	}

	return ret / retWeightSum;
}

template<class RTYPE, std::size_t inDimT>
inline typename RTYPE::value_type RBFInterpolatorPUH<RTYPE, inDimT>::evalDiff(
		const ublas::vector<typename RTYPE::value_type>& inPoint,
		const unsigned alphaDiff) const {

	typename RTYPE::value_type retRBF = 0.;

	typename RTYPE::value_type ret = 0.;
	typename RTYPE::value_type diffRet = 0.;
	typename RTYPE::value_type retWeight = 0.;
	typename RTYPE::value_type retWeightSum = 0;

	typename RTYPE::value_type diffWeight = 0.;
	typename RTYPE::value_type diffWeightSum = 0;

	std::vector<ublas::vector<unsigned> > I = getIndex(inPoint);
	ublas::vector<typename RTYPE::value_type> EL(inDimT);
	ublas::vector<typename RTYPE::value_type> ER(inDimT);
	bool isBorder;

	for (unsigned i = 0; i < I.size(); i++) {
		// compute edges of patch
		for (unsigned j = 0; j < inDimT; j++) {
			EL(j) = BB(j, 0) - o(j) + I[i](j) * (h(j) - alpha * h(j));
			ER(j) = BB(j, 0) - o(j) + h(j) + I[i](j) * (h(j) - alpha * h(j));
			if (EL(j) == inPoint(j) || ER(j) == inPoint(j))
				isBorder = true;
		}
		if (!isBorder) {
			retWeight = omega.eval(inPoint, EL, ER);
			diffWeight = omega.evalDiff(inPoint, EL, ER, alphaDiff);

			diffWeightSum += diffWeight;
			retWeightSum += retWeight;

			retRBF = interpol(I[i]).eval(inPoint);

			ret += retRBF * retWeight;
			diffRet += interpol(I[i]).evalDiff(inPoint, alphaDiff) * retWeight
					+ retRBF * diffWeight;
		} else
			isBorder = false;
	}

	return (-diffWeightSum * ret / retWeightSum + diffRet) / retWeightSum;
}

template<class RTYPE, std::size_t inDimT>
inline typename RTYPE::value_type RBFInterpolatorPUH<RTYPE, inDimT>::evalDiff2(
		const ublas::vector<typename RTYPE::value_type>& inPoint,
		const unsigned alphaDiff) const {

	typename RTYPE::value_type retRBF = 0.;
	typename RTYPE::value_type diffRBF = 0.;

	typename RTYPE::value_type ret = 0.;
	typename RTYPE::value_type diffRet = 0.;
	typename RTYPE::value_type diff2Ret = 0.;

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
			if (EL(j) == inPoint(j) || ER(j) == inPoint(j))
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

template<class RTYPE, std::size_t inDimT>
inline typename RTYPE::value_type RBFInterpolatorPUH<RTYPE, inDimT>::evalDiffMixed(
		const ublas::vector<typename RTYPE::value_type>& inPoint,
		const unsigned alpha1, const unsigned alpha2) const {

	typename RTYPE::value_type retRBF = 0.;
	typename RTYPE::value_type diffRBF1 = 0.;
	typename RTYPE::value_type diffRBF2 = 0.;

	typename RTYPE::value_type ret = 0.;
	typename RTYPE::value_type diffRet1 = 0.;
	typename RTYPE::value_type diffRet2 = 0.;
	typename RTYPE::value_type diffMRet = 0.;

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
			if (EL(j) == inPoint(j) || ER(j) == inPoint(j))
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

template<class RTYPE, std::size_t inDimT>
inline ublas::vector<typename RTYPE::value_type> RBFInterpolatorPUH<RTYPE,
		inDimT>::evalGrad(
		const ublas::vector<typename RTYPE::value_type>& inPoint) const {
	ublas::vector<typename RTYPE::value_type> ret(inDim);
	for (unsigned i = 0; i < inDim; i++) {
		ret(i) = this->evalDiff(inPoint, i + 1);
	}
	return ret;
}

template<class RTYPE, std::size_t inDimT>
inline ublas::matrix<typename RTYPE::value_type> RBFInterpolatorPUH<RTYPE,
		inDimT>::evalHess(
		const ublas::vector<typename RTYPE::value_type>& inPoint) const {
	ublas::matrix<typename RTYPE::value_type, ublas::column_major> ret(inDim,
			inDim);
	ublas::symmetric_adaptor < ublas::matrix<typename RTYPE::value_type,
			ublas::column_major>, ublas::upper > retAdapt(ret);
	for (unsigned i = 0; i < inDim; i++) {
		for (unsigned j = i + 1; j < inDim; j++) {
			retAdapt(i, j) = this->evalDiffMixed(inPoint, i + 1, j + 1);
		}
		retAdapt(i, i) = this->evalDiff2(inPoint, i + 1);
	}
	return ret;
}

template<class RTYPE, std::size_t inDimT>
void RBFInterpolatorPUH<RTYPE, inDimT>::init(
		ublas::matrix<typename RTYPE::value_type, ublas::column_major> grid,
		ublas::vector<typename RTYPE::value_type> data,
		ublas::matrix<typename RTYPE::value_type, ublas::column_major> diffData) {
	/*
	 * basic steps of the initialisation:
	 * 1. compute bounding box
	 * 2. compute all kind of paramters:
	 * 		- delta_i diameter in dimension i
	 * 		- MSnake number of patches if directly adjacent
	 * 		- o_i overlap in dimension i
	 * 		- M number of patches with overlap
	 * 3. run trough points and determine indices of patches they belong to
	 * 4. initialize RBF interpolators for each patch in multiarray *
	 */

	// 1. and 2. some parameters that already can be used in the loop
	unsigned MTilde = std::floor(std::pow(numOfPoints / K, 1. / inDim));
	M = std::ceil((MTilde - alpha) / (1. - alpha));
	// shape of multi_array, needed later, but can be initialized in the loop
	ublas::vector<unsigned, ublas::bounded_array<unsigned, inDimT> > shape(
			inDimT);
	// vector of row indices, needed later but can be initialized in the loop
	ublas::vector<unsigned, ublas::bounded_array<unsigned, inDimT> > rows(
			inDimT);
	// Bounding Box
	BB.resize(inDim, 2);
	h.resize(inDim);
	o.resize(inDim);
	for (unsigned i = 0; i < inDim; i++) {
		BB(i, 0) = *std::min_element(ublas::row(grid, i).begin(),
				ublas::row(grid, i).end());
		BB(i, 1) = *std::max_element(ublas::row(grid, i).begin(),
				ublas::row(grid, i).end());
		h(i) = (BB(i, 1) - BB(i, 0)) / MTilde;
		o(i) = (M * (1 - alpha) - MTilde + alpha) * h(i) / 2.;
		if (o(i)<0)
		  o(i)=0;
		// use for loop to fill shape vector for later
		shape(i) = M;
		// use for loop to fill row indices
		rows(i) = i;
	}

	// 3. get indices of patches for each grid point
	//temporary storage for patch indices as output of getIndex()
	std::vector<ublas::vector<unsigned> > I(inDim);

	// multiDim Storage for column indices
	boost::multi_array < ublas::vector<unsigned>, inDimT > Columns(shape);
	unsigned tempSize = 0;
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
	//RBFInterpolator tempInterpol();
	// index for rows
	ublas::indirect_array < ublas::vector<unsigned> > R(inDimT, rows);
	// resize interpol multi_array
	interpol.resize(shape);
	for (unsigned i = 0; i < std::pow(M, inDimT); i++) {
		ublas::indirect_array < ublas::vector<unsigned> > C(
				Columns(ind2sub(i)).size(), Columns(ind2sub(i)));
		// put interpolator in patches
		interpol(ind2sub(i)) = RBFInterpolatorH<RTYPE> (
				ublas::matrix_indirect<ublas::matrix<
						typename RTYPE::value_type, ublas::column_major>,
						ublas::indirect_array<ublas::vector<unsigned> > >(grid,
						R, C),
				ublas::vector_indirect<
						ublas::vector<typename RTYPE::value_type>,
						ublas::indirect_array<ublas::vector<unsigned> > >(data,
						C),
				ublas::matrix_indirect<ublas::matrix<
						typename RTYPE::value_type, ublas::column_major>,
						ublas::indirect_array<ublas::vector<unsigned> > >(
						diffData, R, C), scale);
	}
}

/*
 * Compute the index of all patches a point lies in.
 */
template<class RTYPE, std::size_t inDimT>
inline std::vector<ublas::vector<unsigned> > RBFInterpolatorPUH<RTYPE, inDimT>::getIndex(
		const ublas::vector<typename RTYPE::value_type>& inPoint) const {

	// initialize some temporary variables
	std::vector<ublas::vector<unsigned, ublas::bounded_array<unsigned, 2> > >
			ret(inDim);
	// temporary vector of patch numbers
	ublas::vector<unsigned, ublas::bounded_array<unsigned, 2> > tempPatch(2);
	// temporary fractioned patch number --> converted to unsigned
	typename RTYPE::value_type tempL;
	typename RTYPE::value_type tempR;
	// counter --> how many patch indices do we have
	unsigned l = 1;

	for (unsigned i = 0; i < inDim; i++) {
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
			tempPatch(0) = (unsigned) std::floor(tempL);

		// right edge
		tempR = (o(i) - BB(i, 0) - h(i) + inPoint(i)) / (h(i) - alpha * h(i));
		if (tempR < 0)
			tempPatch(1) = 0;
		else
			tempPatch(1) = (unsigned) std::ceil(tempR);

		// test if inPoint falls exactly on the edge of one patch
		// test for same element.
		if (tempPatch(0) == tempPatch(1)) {
			tempPatch.resize(1, true);
			ret[i] = tempPatch;
			// resize to have right size
			tempPatch.resize(2, false);
		} else {
			ret[i] = tempPatch;
			// update counter only if two patches are added
			l = 2 * l;
		}
	}

	// prepare to call combine
	ublas::vector<unsigned> ind(inDim);
	ublas::vector<unsigned>& indRef = ind;
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
template<class RTYPE, std::size_t inDimT>
inline void RBFInterpolatorPUH<RTYPE, inDimT>::combine(
		ublas::vector<unsigned>& ind,
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
		if (d < inDim - 1)
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
template<class RTYPE, std::size_t inDimT>
inline ublas::vector<unsigned, ublas::bounded_array<unsigned, inDimT> > RBFInterpolatorPUH<
		RTYPE, inDimT>::ind2sub(unsigned ind) const {
	ublas::vector<unsigned, ublas::bounded_array<unsigned, inDimT> >
			ret(inDimT);
	unsigned MExp = std::pow(M, inDimT - 1);
	for (int i = inDimT - 1; i >= 0; i--) {
		ret(i) = (ind - (ind % MExp)) / MExp;
		ind = ind - MExp * ret(i);
		MExp /= M;
	}
	return ret;
}

template<class RTYPE, std::size_t inDimT>
RBFInterpolatorPUH<RTYPE, inDimT>::~RBFInterpolatorPUH() {
}

}

#endif /* RBFINTERPOLATORPUH_H_ */
