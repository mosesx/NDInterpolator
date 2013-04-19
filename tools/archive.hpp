/**
 * archive.h
 *	\brief include boost::serialization, in order to have one file to go to for changing the archive
 *	\brief type
 *  \date Feb 15, 2011
 *  \author mrehberg
 */

// do not include if serialization support is not wanted by user
#ifndef _WITHOUT_SERIALIZATION_

#ifndef ARCHIVE_H_
#define ARCHIVE_H_

// archive types to save and read
#include "boost/archive/text_oarchive.hpp"
#include "boost/archive/text_iarchive.hpp"

// safe derived class serialization
#include "boost/serialization/base_object.hpp"

// include vector to be able to deal with diffData Types
#include "boost/serialization/vector.hpp"

// save and load in different functions for PU interpolators
#include "boost/serialization/split_member.hpp"
#endif /* ARCHIVE_H_ */
#endif /* _WITHOUT_SERIALIZATION_ */
