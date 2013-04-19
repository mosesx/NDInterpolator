#ifndef _INTERPCPPADSETTINGS_
#define _INTERPCPPADSETTINGS_

typedef NDInterpolator::MultiRBFInterpolatorPUH<
  NDInterpolator::GaussianRBF<double>, 2, 2> interpType;

#define _INTERPCPPADVECTORTYPE_ CppAD::vector

#endif
