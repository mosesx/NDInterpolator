//============================================================================
// Name        : Test.cpp
// Author      : Marcel Rehberg
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <string>
#include <vector>
#include <cmath>

#include "MultiRBFInterpolators.hpp"
#include "tools/interpCppADOperation.hpp"
#include "interpCppADSettings.hpp"

#include "exampleDefs.hpp"

int main() {
  namespace ublas = boost::numeric::ublas;

  ublas::vector<double> lcI(2);
  ublas::vector<double> rcI(2);
  ublas::vector<unsigned> numOfPointsI(2);
  lcI(0) = -3.14;
  lcI(1) = -3.14;
  rcI(0) = 3.14;
  rcI(1) = 3.14;
  numOfPointsI(0) = 10;
  numOfPointsI(1) = 10;
  ublas::matrix<double, ublas::column_major> grid =
    NDInterpolator::genRecGrid(lcI, rcI, numOfPointsI);

  // create interpolation data
  // Attention: different data structure for multi-dim output interpolators although
  // the function only has one output
  unsigned N = numOfPointsI(0) * numOfPointsI(1);
  ublas::matrix<double, ublas::column_major> diffData(2, 2);
  ublas::matrix<double, ublas::column_major> data(2, N);
  std::vector<ublas::matrix<double, ublas::column_major> > diffDataMulti(N);
  for (unsigned i = 0; i < N; i++) {
    data(0, i) = rosenbrock(ublas::column(grid, i));
    data(1, i) = sinCos(ublas::column(grid, i));

    diffData(0, 0) = diffRosenbrock(ublas::column(grid, i), 1);
    diffData(0, 1) = diffRosenbrock(ublas::column(grid, i), 2);
    diffData(1, 0) = diffSinCos(ublas::column(grid, i), 1);
    diffData(1, 1) = diffSinCos(ublas::column(grid, i), 2);
    diffDataMulti[i] = diffData;
  }


  interpType multiPUHermiteRBF(grid, data, diffDataMulti, 2., 20, .33) ;

  multiPUHermiteRBF.optimizeScale();
  interpType* interpRef = &multiPUHermiteRBF;
  interpolators.push_back(interpRef);
  
  CppAD::vector<CppAD::AD<double> > X(2), ax(2), ay(2);
  X[0]=0; X[1]=0;
  CppAD::Independent(X);
  ax[0]=X[0]; ax[1]=X[1];
  rbfInterpolator(0,ax,ay);
  
  CppAD::ADFun<double> f;
  f.Dependent(X,ay);

  // test forward mode, ATTENTION: direct evaluation results are not supposed to be equal
  ublas::vector<double> inPoint(2); inPoint(0)=0; inPoint(1)=0;
  CppAD::vector<double> x(2),w(2); x[0]=0; x[1]=0;
  std::cout << "\n\nTest forward mode" << std::endl;
  std::cout <<"\nforward evaluation:\t"<<f.Forward(0,x) << "\t\tdirect evaluation:\t"<<
    interpolators[0]->eval(inPoint) << std::endl;

  CppAD::vector<double> dx(2); dx[0]=1; dx[1]=0;
  CppAD::vector<double> dx2(2); dx2[0]=1; dx2[1]=0;
  // d/dx1^2
  dx[0]=1; dx[1]=0; dx2[0]=0; dx2[1]=0;
  std::cout <<"forward d/dx1:\t\t"<< f.Forward(1,dx) << "\t\tdirect evaluation:\t" <<
    interpolators[0]->evalDiff(inPoint,1) << std::endl;
  std::cout <<"forward d^2/dx1^2:\t"<< f.Forward(2,dx2) << "\tdirect evaluation:\t" <<
    interpolators[0]->evalDiff2(inPoint,1) << std::endl;
  // d/dx1dx2
  dx[0]=1; dx[1]=1; dx2[0]=0; dx2[1]=0;
  f.Forward(1,dx);
  std::cout <<"forward d/dx1x2:\t"<< f.Forward(2,dx2) << "\t\tdirect evaluation:\t" <<
    interpolators[0]->evalDiffMixed(inPoint,1,2) << std::endl;
  // d/dx2^2
  dx[0]=0; dx[1]=1; dx2[0]=0; dx2[1]=0;
  std::cout <<"forward d/dx2:\t\t"<< f.Forward(1,dx) << "\tdirect evaluation:\t" <<
    interpolators[0]->evalDiff(inPoint,2) << std::endl;
  std::cout <<"forward d^2/dx2^2:\t"<< f.Forward(2,dx2) << "\t\tdirect evaluation:\t" <<
    interpolators[0]->evalDiff2(inPoint,2) << std::endl;

  std::cout << "\n\nTest reverse mode" << std::endl;
  w[0]=1; w[1]=0;
  std::cout << "\nreverse (df1/dx1, df1/dx2):\t" << f.Reverse(1,w) << "\tdirect evaluation:\t" <<
    ublas::row(interpolators[0]->evalJac(inPoint),0) << std::endl;
  w[0]=0; w[1]=1;
  std::cout << "reverse (df2/dx1, df2/dx2):\t" << f.Reverse(1,w) << "\tdirect evaluation:\t" <<
    ublas::row(interpolators[0]->evalJac(inPoint),1) << std::endl;

  w[0]=1; w[1]=0; dx[0]=1; dx[1]=0;
  f.Forward(1,dx);
  std::cout << "\nreverse (df1/dx1, df1^2/dx1^2, df1/dx2, df1^2/dx1dx2):\t\t\t" <<
    f.Reverse(2,w) <<
    "\ndirect evaluation (df1/dx1, df1/dx2), (df1^2/dx1^2, df1^2/dx1dx2)\t" <<
    ublas::row(interpolators[0]->evalJac(inPoint),0) << "\t" <<
    ublas::row((interpolators[0]->evalHess(inPoint))[0],0)<< std::endl;

  w[0]=1; w[1]=0; dx[0]=0; dx[1]=1;
  f.Forward(1,dx);
  std::cout << "\nreverse (df1/dx1, df1^2/dx1dx2, df1/dx2, df1^2/dx2^2):\t\t\t" <<
    f.Reverse(2,w) <<
    "\ndirect evaluation (df1/dx1, df1/dx2), (df1^2/dx1dx2, df1^2/dx2^2)\t" <<
    ublas::row(interpolators[0]->evalJac(inPoint),0) << "\t" <<
    ublas::column((interpolators[0]->evalHess(inPoint))[0],1)<< std::endl;

  w[0]=0; w[1]=1; dx[0]=1; dx[1]=0;
  f.Forward(1,dx);
  std::cout << "\nreverse (df2/dx1, df2^2/dx1^2, df2/dx2, df2^2/dx2dx1):\t\t\t" << f.Reverse(2,w)
  	    << "\ndirect evaluation (df2/dx1, df2/dx2), (df2^2/dx1^2, df2^2/dx2dx1)\t" <<
    ublas::row(interpolators[0]->evalJac(inPoint),1) << "\t" <<
    ublas::row((interpolators[0]->evalHess(inPoint))[1],0)<< std::endl;

  w[0]=0; w[1]=1; dx[0]=0; dx[1]=1;
  f.Forward(1,dx);
  std::cout << "\nreverse (df2/dx1, df2^2/dx1dx2, df2/dx2, df2^2/dx2^2):\t\t\t" << f.Reverse(2,w)
  	    << "\ndirect evaluation (df2/dx1, df2/dx2), (df2^2/dx1dx2, df2^2/dx2^2)\t" <<
    ublas::row(interpolators[0]->evalJac(inPoint),1) << "\t" <<
    ublas::column((interpolators[0]->evalHess(inPoint))[1],1)<< std::endl;


  // sparsity Check
  std::cout <<"\n\nTest sparsity" << std::endl;
  static const size_t n=2;
  static const size_t m=2;
  CppAD::vector<bool> r(n * n);
  size_t i, j;
  for(i = 0; i < n; i++) {
    for(j = 0; j < n; j++)
      r[ i * n + j ] = (i == j);
  }
  CppAD::vector<bool> s(m * n);

  s = f.ForSparseJac(n, r);
  std::cout <<"Forward Jacobian sparsity (unit r)\t" << s << std::endl;

  for(i = 0; i < n; i++) {
    for(j = 0; j < n; j++)
      r[ i * n + j ] = (i < j);
  }
  std::cout <<"\nForward Jacobian sparsity input R: \t"<< r << std::endl;
  s = f.ForSparseJac(n, r);
  std::cout <<"Forward Jacobian sparsity\t\t" << s << std::endl;

  // jacobian reverse
  for(i = 0; i < m; i++) {
    for(j = 0; j < m; j++)
      s[ i * m + j ] = (i == j);
  }
  r = f.RevSparseJac(m, s);
  std::cout <<"\nReverse Jacobian sparsity (unit s)\t" << r << std::endl;

  for(i = 0; i < m; i++) {
    for(j = 0; j < m; j++)
      s[ i * m + j ] = (i < j);
  }
  std::cout <<"\nReverse Jacibian sparsity inputS: \t"<< s << std::endl;
  r = f.RevSparseJac(m, s);
  std::cout <<"Reverse Jacobian sparsity\t\t" << r << std::endl;

  // hessian reverse
  // first r should be identity matrix and used in call to forSparseJac
  for(i = 0; i < n; i++) {
    for(j = 0; j < n; j++)
      r[ i * n + j ] = (i == j);
  }
  f.ForSparseJac(n,r);
  s.resize(m);
  s[0]=true; s[1]=false;
  CppAD::vector<bool> h(n*n);
  h=f.RevSparseHes(n,s);
  std::cout <<"\n\nReverse Hessian sparsity (s=[1,0])\t" << h << std::endl;
  s[0]=false; s[1]=true;
  h=f.RevSparseHes(n,s);
  std::cout <<"Reverse Hessian sparsity (s=[0,1])\t" << h << std::endl;
  std::cout << "\n" << std::endl;

  CppAD::user_atomic<double>::clear();

  return 0;
}

