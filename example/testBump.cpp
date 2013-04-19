#include <vector>
#include "../tools/utility.hpp"
#include "../tools/bump.hpp"

#include "exampleDefs.hpp"

int main(int argc, char *argv[]) {
  namespace ublas = boost::numeric::ublas;

  ublas::matrix<double, ublas::column_major> BB(2,2);
  BB(0,0) = -1; BB(0,1) = 2; BB(1,0) = 0; BB(1,1) = 2;

  ublas::vector<double> lu(2);
  ublas::vector<double> ro(2); 
  ublas::vector<size_t> numOfPoints(2);

  lu(0) = -1.5; lu(1) = -.5; ro(0) = 2; ro(1) = 3;
  numOfPoints(0) = 100; numOfPoints(1) = 100; 
  ublas::matrix<double, ublas::column_major> grid = NDInterpolator::genRecGrid(lu,ro,numOfPoints);


  NDInterpolator::Bump<double> tBump(BB);
  NDInterpolator::writeVector(tBump.evalGrid(grid),"bumbTest.dat");
  NDInterpolator::writeVector(tBump.d_evalGrid(grid,1),"d_bumbTest.dat");
  NDInterpolator::writeMatrix(grid,"bumbTestGrid.dat");

  size_t N = numOfPoints(0) * numOfPoints(1);
  ublas::matrix<double, ublas::column_major> dataMulti(1, N);
  std::vector<ublas::matrix<double, ublas::column_major> > diffDataMulti(N);
  for (unsigned i = 0; i < N; i++) {
    dataMulti(0, i) = rosenbrock(ublas::column(grid, i));

    // diffData(0, i) = diffRosenbrock(ublas::column(grid, i), 1);
    // diffData(1, i) = diffRosenbrock(ublas::column(grid, i), 2);
    diffDataMulti[i].resize(1,2);
    diffDataMulti[i](0,0) = diffRosenbrock(ublas::column(grid,i),1);
    diffDataMulti[i](0,1) = diffRosenbrock(ublas::column(grid,i),2);
  }

  NDInterpolator::writeMatrix(dataMulti,"bumpTestDataB.dat");
  ublas::vector<double> outVec(N);
  for (unsigned i = 0; i < N; i++) {
    outVec(i) = diffDataMulti[i](0,1);
  }
  NDInterpolator::writeVector(outVec,"bumpTestDiffB.dat");

  tBump.applyToDiffAndData(grid,dataMulti,diffDataMulti);

  for (unsigned i = 0; i < N; i++) {
    outVec(i) = diffDataMulti[i](0,1);
  }
  NDInterpolator::writeMatrix(dataMulti,"bumpTestData.dat");
  NDInterpolator::writeVector(outVec,"bumpTestDiff.dat");
}
