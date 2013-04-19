//============================================================================
// Name        : example.cpp
// Author      : Marcel Rehberg
// For building and execution instructions see doc/html/index.html
//============================================================================

#include <iostream>

#include "boost/timer/timer.hpp"
// defines matrix/vector i/o functions and other utilities
#include "RBFInterpolators.hpp"
#include "MultiRBFInterpolators.hpp"

#include "exampleDefs.hpp"



int main(int argc, char *argv[]) {

  // ublas types are included via NDInterpolator.h
  namespace ublas = boost::numeric::ublas;

  // create interpolation grid by specifying edges and number of points in each dimension.
  ublas::vector<double> lcI(1);
  ublas::vector<double> rcI(1);
  ublas::vector<unsigned> numOfPointsI(1);
  double domain = 3.33;
  lcI(0) = domain;
  rcI(0) = -domain;

  
  // schleife über numOfPoints
  ublas::matrix<double, ublas::column_major> grid;
  unsigned N, NEval;
  ublas::matrix<double,ublas::column_major> data(1,1);
  std::vector<ublas::matrix<double, ublas::column_major> > diffData(1);
  ublas::vector<double> lcE(1);
  ublas::vector<double> rcE(1);
  ublas::vector<unsigned> numOfPointsE(1);
  numOfPointsE(0) = 500;  
  NEval = numOfPointsE(0);

  ublas::matrix<double, ublas::column_major> eval;
    // storage for true result values obtained by evaluating the function on the evauluation grid
  ublas::vector<double> res(NEval);
  ublas::vector<double> dres(NEval);
  // storage for interpolation results up to derivatives of second order
  ublas::vector<double> resI(NEval);

  // create evaluation grid
  lcE(0) = .9*domain;
  rcE(0) = -.9*domain;

  
  eval = NDInterpolator::genRecGrid(lcE, rcE, numOfPointsE);

  typedef NDInterpolator::MultiRBFInterpolatorPUH<NDInterpolator::GaussianRBF<double>, 1, 1 > interpType;
  interpType lagrangeRBF;

  ublas::vector<double> error;
  ublas::vector<double> P;
  ublas::vector<double> alpha;
  ublas::vector<double> timeCreate;
  ublas::vector<double> timeEval;
  boost::timer::cpu_timer t;
  for (unsigned p = 0; p<120; p++){
    
    numOfPointsI(0) = p+4;
    error.resize(error.size()+1);
    P.resize(P.size()+1);
    P(p) = numOfPointsI(0);
    alpha.resize(alpha.size()+1);
    timeCreate.resize(timeCreate.size()+1);
    timeEval.resize(timeEval.size()+1);

    grid = NDInterpolator::genRecGrid(lcI, rcI, numOfPointsI);
    
    // create interpolation data
    // Attention: different data structure for multi-dim output interpolators although
    // the function only has one output
    N = numOfPointsI(0);
    data.resize(1,N);
    diffData.resize(N);
    for (unsigned i = 0; i < N; i++) {
      data(0,i) = peaks1d(ublas::column(grid, i));
      diffData[i].resize(1,1);
      diffData[i](0,0) = dpeaks1d(ublas::column(grid,i));
      //      std::cout << N << "\t" << data(0,i) << std::endl;
    }
    
    // create simple interpolator objects and use the grid and data created
    // scale parameters are just guesses

    t.start();
    lagrangeRBF = interpType(grid, data, diffData, 10,20,.1);
    lagrangeRBF.optimizeScale();
    //    lagrangeRBF.printScale();
    t.stop();
    timeCreate(p) = atof(boost::timer::format(t.elapsed(),16,"%w").c_str());

    alpha(p) = 1;
    
    for (unsigned i = 0; i < NEval; i++) {
      res(i) = peaks1d(ublas::column(eval, i));
      dres(i) = dpeaks1d(ublas::column(eval,i));
    }
    t.start();
    for (unsigned i = 0; i < NEval; i++) {
      resI(i) = lagrangeRBF.eval(ublas::column(eval, i))(0);
    }
    t.stop();
    timeEval(p) = atof(boost::timer::format(t.elapsed(),16,"%w").c_str());
    
    error(p) = ublas::norm_inf(res-resI);
  }
  // write error to file
  NDInterpolator::writeVector(P,"numOfPointsSinShapeOptPU.dat");
  NDInterpolator::writeVector(error,"errorSinShapeOptPUH.dat");
  NDInterpolator::writeVector(alpha,"alpha.dat");
  NDInterpolator::writeVector(timeCreate,"timeCreatePUH.dat");
  NDInterpolator::writeVector(timeEval,"timeEvalPUH.dat");

  

  NDInterpolator::writeMatrix(eval,"PUHGrid.dat");
  NDInterpolator::writeVector(res,"resPUH.dat");
  NDInterpolator::writeVector(dres,"dresPUH.dat");
  NDInterpolator::writeVector(resI,"resIPUH.dat");
  // // plots for interesting output
  // numOfPointsI(0) = 45;
  // grid = NDInterpolator::genRecGrid(lcI, rcI, numOfPointsI);
  
  // // create interpolation data
  // // Attention: different data structure for multi-dim output interpolators although
  // // the function only has one output
  // N = numOfPointsI(0);
  // data.resize(N);
  // for (unsigned i = 0; i < N; i++) {
  //   data(i) = peaks1d(ublas::column(grid, i));
  // }
  
  // // create simple interpolator objects and use the grid and data created
  // // scale parameters are just guesses
  
  // lagrangeRBF = interpType(grid, data, 10);
  // lagrangeRBF.optimizeScale();
  
  // for (unsigned i = 0; i < NEval; i++) {
  //   res(i) = peaks1d(ublas::column(eval, i));
  //   resI(i) = lagrangeRBF.eval(ublas::column(eval, i));
  // }
  // NDInterpolator::writeVector(res,"resPU.dat");
  // NDInterpolator::writeVector(resI,"interpPU.dat");
  // NDInterpolator::writeMatrix(grid,"gridPU.dat");
  // NDInterpolator::writeMatrix(eval,"gridInterpPU.dat");
  
  return 0;
}
















