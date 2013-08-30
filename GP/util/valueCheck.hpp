#ifndef VALUE_CHECK_HPP
#define VALUE_CHECK_HPP

#include <float.h>
#include "GP/DataTypes.hpp"

namespace GPOM {

	#ifndef isNaN
	#define isNaN(x) ((x)!=(x)) 
	#endif

	//# define isNaN(x)    _isnan(x)
	//# define isFinite(x) (_finite(x) != 0)
	# define isInf(x)    (_finite(x) == 0)

	const Scalar BIG_NUMBER = 1.0e10f;

}

#endif