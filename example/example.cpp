//============================================================================
// Name        : example.cpp
// Author      : Marcel Rehberg
// For building and execution instructions see doc/html/index.html
//============================================================================

#include <iostream>

#include "boost/progress.hpp"
// defines matrix/vector i/o functions and other utilities
#include "RBFInterpolators.hpp"
#include "MultiRBFInterpolators.hpp"

#include "exampleDefs.hpp"



int main(int argc, char *argv[]) {

  // ublas types are included via NDInterpolator.h
  namespace ublas = boost::numeric::ublas;

  // create interpolation grid by specifying edges and number of points in each dimension.
  ublas::vector<double> lcI(2);
  ublas::vector<double> rcI(2);
  ublas::vector<unsigned> numOfPointsI(2);
  lcI(0) = -1.5;
  lcI(1) = -0.5;
  rcI(0) = 2;
  rcI(1) = 3;
  
  if (argc>2) {
    numOfPointsI(0)=(unsigned) atoi(argv[2]);
    numOfPointsI(1)=(unsigned) atoi(argv[2]);
  }
  else {
    numOfPointsI(0) = 10;
    numOfPointsI(1) = 10;
  }
  
  ublas::matrix<double, ublas::column_major> grid =
    NDInterpolator::genRecGrid(lcI, rcI, numOfPointsI);

  // create interpolation data
  // Attention: different data structure for multi-dim output interpolators although
  // the function only has one output
  unsigned N = numOfPointsI(0) * numOfPointsI(1);
  ublas::vector<double> data(N);
  ublas::matrix<double, ublas::column_major> diffData(2, N);
  ublas::matrix<double, ublas::column_major> dataMulti(1, N);
  std::vector<ublas::matrix<double, ublas::column_major> > diffDataMulti(N);
  for (unsigned i = 0; i < N; i++) {
    data(i) = rosenbrock(ublas::column(grid, i));
    dataMulti(0, i) = data(i);

    diffData(0, i) = diffRosenbrock(ublas::column(grid, i), 1);
    diffData(1, i) = diffRosenbrock(ublas::column(grid, i), 2);
    diffDataMulti[i] = ublas::trans(
				    ublas::subrange(diffData, 0, 2, i, i + 1));
  }
  
  // create evaluation grid
  ublas::vector<double> lcE(2);
  ublas::vector<double> rcE(2);
  ublas::vector<unsigned> numOfPointsE(2);
  lcE(0) = -1.4;
  lcE(1) = -0.4;
  rcE(0) = 1.9;
  rcE(1) = 2.9;
  numOfPointsE(0) = 35;
  numOfPointsE(1) = 35;
  ublas::matrix<double, ublas::column_major> eval =
    NDInterpolator::genRecGrid(lcE, rcE, numOfPointsE);

  unsigned NEval = numOfPointsE(0) * numOfPointsE(1);

  // storage for true result values obtained by evaluating the function on the evauluation grid
  ublas::vector<double> res(NEval);
  ublas::vector<double> resDiff(NEval);
  ublas::vector<double> resDiff2(NEval);
  ublas::vector<double> resDiffM(NEval);

  // storage for interpolation results up to derivatives of second order
  ublas::vector<double> resI(NEval);
  ublas::vector<double> resIMulti(NEval);

  ublas::vector<double> resDiffI(NEval);
  ublas::vector<double> resDiffIMulti(NEval);

  ublas::vector<double> resDiff2I(NEval);
  ublas::vector<double> resDiff2IMulti(NEval);

  ublas::vector<double> resDiffMI(NEval);
  ublas::vector<double> resDiffMIMulti(NEval);

  //----------------------------------------------------------------------------
  // Everything prepared to create and evaluate the interpolator objects
  //----------------------------------------------------------------------------

  // create simple interpolator objects and use the grid and data created
  // scale parameters are just guesses
  // NDInterpolator::RBFInterpolator<NDInterpolator::GaussianRBF<double> >
  //   lagrangeRBF(grid, data, 2.);
  // lagrangeRBF.optimizeScale();

  // NDInterpolator::RBFInterpolatorH<NDInterpolator::GaussianRBF<double> >
  //   hermiteRBF(grid, data, diffData, 2.);
  // hermiteRBF.optimizeScale();

  // NDInterpolator::RBFInterpolatorPU<NDInterpolator::GaussianRBF<double>, 2>
  //   PULagrangeRBF(grid, data, 2., 50, 0.33);
  // PULagrangeRBF.optimizeScale();

  // NDInterpolator::RBFInterpolatorPUH<NDInterpolator::GaussianRBF<double>, 2>
  //   PUHermiteRBF(grid, data, diffData, 2., 50, 0.33);
  // PUHermiteRBF.optimizeScale();

  // // create multi-dim out interpolator objects
  // // Attention: uses different data structures than simple interpolator
  // NDInterpolator::MultiRBFInterpolator<NDInterpolator::GaussianRBF<double> >
  //   multiLagrangeRBF(grid, dataMulti, 2.);
  // multiLagrangeRBF.optimizeScale();

  // NDInterpolator::MultiRBFInterpolatorH<NDInterpolator::GaussianRBF<double> >
  //   multiHermiteRBF(grid, dataMulti, diffDataMulti, 2.);
  // multiHermiteRBF.optimizeScale();

  // NDInterpolator::MultiRBFInterpolatorPU<NDInterpolator::GaussianRBF<double>, 2, 1> multiPULagrangeRBF(grid, dataMulti, 2., 50, .33);
  // multiPULagrangeRBF.optimizeScale();
  
  NDInterpolator::MultiRBFInterpolatorPUH<
    NDInterpolator::GaussianRBF<double>, 2, 1> multiPUHermiteRBF(grid, dataMulti, 
								 diffDataMulti, 2., 50, .2);
  if (argc > 1) {
    if (atof(argv[1])>0){
      multiPUHermiteRBF.setNewScale(atof(argv[1]));
    }
    else{
      multiPUHermiteRBF.optimizeScale();
      multiPUHermiteRBF.printScale();
    }
  }
  else {
    multiPUHermiteRBF.optimizeScale();
    multiPUHermiteRBF.printScale();
  }

  //============================================================================================
  // EXCHANGE interpolators here
  // evaluate interpolation object
  // Uses (Multi)NDInterpolator to be able to reference the different interpolators
  //===========================================================================================
  // typedef NDInterpolator::RBFInterpolator<NDInterpolator::GaussianRBF<double> > interpType;
  // typedef NDInterpolator::RBFInterpolatorH<NDInterpolator::GaussianRBF<double> > interpType;
  // typedef NDInterpolator::RBFInterpolatorPU<NDInterpolator::GaussianRBF<double>, 2> interpType;
  typedef NDInterpolator::RBFInterpolatorPUH<NDInterpolator::GaussianRBF<double>, 2> interpType;

  // typedef NDInterpolator::MultiRBFInterpolator<NDInterpolator::GaussianRBF<double> > interpTypeMulti;
  // typedef NDInterpolator::MultiRBFInterpolatorH<NDInterpolator::GaussianRBF<double> > interpTypeMulti;
  // typedef NDInterpolator::MultiRBFInterpolatorPU<NDInterpolator::GaussianRBF<double>,2, 1> interpTypeMulti;
  typedef NDInterpolator::MultiRBFInterpolatorPUH<NDInterpolator::GaussianRBF<double>,2, 1> interpTypeMulti;

  //  interpType& interpRef = PUHermiteRBF;
  interpTypeMulti& interpRefMulti = multiPUHermiteRBF;
  //===========================================================================================

  // evaluation for all points in evaluation grid
  {    boost::progress_timer t;
  for (unsigned l=0; l<1; l++){
    for (unsigned i = 0; i < NEval; i++) {
      res(i) = rosenbrock(ublas::column(eval, i));
      // resI(i) = interpRef.eval(ublas::column(eval, i));
      resIMulti(i) = interpRefMulti.eval(ublas::column(eval, i))(0);
      
      //interpRefMulti.eval(ublas::column(eval, i));
      //interpRefMulti.evalDiff(ublas::column(eval, i), 1);
      
      //    resDiff(i) = diffRosenbrock(ublas::column(eval, i), 1);
      //resDiffI(i) = interpRef.evalDiff(ublas::column(eval, i), 1);
      // resDiffIMulti(i)
      //   = interpRefMulti.evalDiff(ublas::column(eval, i), 1)(0);
      
      // resDiff2(i) = diff2Rosenbrock(ublas::column(eval, i), 1, 1);
      // //resDiff2I(i) = interpRef.evalDiff2(ublas::column(eval, i), 1);
      // resDiff2IMulti(i)
      //   = interpRefMulti.evalDiff2(ublas::column(eval, i), 1)(0);
      
      // resDiffM(i) = diff2Rosenbrock(ublas::column(eval, i), 1, 2);
      // //resDiffMI(i) = interpRef.evalDiffMixed(ublas::column(eval, i), 2, 1);
      // resDiffMIMulti(i) = interpRefMulti.evalDiffMixed(
      // 						     ublas::column(eval, i), 2, 1)(0);
    }
  }
  }
  // output absolute error per point (1 dim output Interpolators
  // std::cout << "\n\n" << std::endl;
  // std::cout << "Evaluation of 1-dim output Interpolators \n \n" << std::endl;
  // std::cout << "\t mean absolute error function evaluation: \t\t"
  // 	    << ublas::norm_1(res - resI) / NEval << std::endl;
  // std::cout << "\t max absolute error function evaluation: \t\t"
  // 	    << ublas::norm_inf(res - resI) << "\n" << std::endl;

  // std::cout << "\t mean absolute error derivative evaluation: \t\t"
  // 	    << ublas::norm_1(resDiff - resDiffI) / NEval << std::endl;
  // std::cout << "\t max absolute error derivative evaluation: \t\t"
  // 	    << ublas::norm_inf(resDiff - resDiffI) << "\n" << std::endl;

  // std::cout << "\t mean absolute error 2nd derivative evaluation: \t"
  // 	    << ublas::norm_1(resDiff2 - resDiff2I) / NEval << std::endl;
  // std::cout << "\t max absolute error 2nd derivative evaluation: \t\t"
  // 	    << ublas::norm_inf(resDiff2 - resDiff2I) << "\n" << std::endl;

  // std::cout << "\t mean absolute error mixed derivative evaluation: \t"
  // 	    << ublas::norm_1(resDiffM - resDiffMI) / NEval << std::endl;
  // std::cout << "\t max absolute error mixed derivative evaluation: \t"
  // 	    << ublas::norm_inf(resDiffM - resDiffMI) << std::endl;

  // output absolute error per point (multi Interpolators)
  std::cout << "\n\n" << std::endl;
  std::cout << "Evaluation of multi-dim output Interpolators \n \n"
  	    << std::endl;
  std::cout << "\t mean absolute error function evaluation: \t\t"
  	    << ublas::norm_1(res - resIMulti) / NEval << std::endl;
  std::cout << "\t max absolute error function evaluation: \t\t"
  	    << ublas::norm_inf(res - resIMulti) << "\n" << std::endl;

  // std::cout << "\t mean absolute error derivative evaluation: \t\t"
  // 	    << ublas::norm_1(resDiff - resDiffIMulti) / NEval << std::endl;
  // std::cout << "\t max absolute error derivative evaluation: \t\t"
  // 	    << ublas::norm_inf(resDiff - resDiffIMulti) << "\n" << std::endl;

  // std::cout << "\t mean absolute error 2nd derivative evaluation: \t"
  // 	    << ublas::norm_1(resDiff2 - resDiff2IMulti) / NEval << std::endl;
  // std::cout << "\t max absolute error 2nd derivative evaluation: \t\t"
  // 	    << ublas::norm_inf(resDiff2 - resDiff2IMulti) << "\n" << std::endl;

  // std::cout << "\t mean absolute error mixed derivative evaluation: \t"
  // 	    << ublas::norm_1(resDiffM - resDiffMIMulti) / NEval << std::endl;
  // std::cout << "\t max absolute error mixed derivative evaluation: \t"
  // 	    << ublas::norm_inf(resDiffM - resDiffMIMulti) << "\n" << std::endl;


  // // write grid to file
  NDInterpolator::writeMatrix(grid,"grid.dat");
  NDInterpolator::writeMatrix(eval,"eval.dat");

  // // write results to file
  NDInterpolator::writeMatrix(dataMulti,"dataMulti.dat");
  // NDInterpolator::writeVector(res,"res.dat");
  NDInterpolator::writeVector(resIMulti,"resIMulti.dat");
  // NDInterpolator::writeVector(resDiffIMulti,"resDiffIMulti.dat");
  

  return 0;
}

