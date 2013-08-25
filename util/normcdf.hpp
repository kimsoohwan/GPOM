#ifndef NORMAL_CUMMULATIVE_DISTRIBUTION_FUNCTION_HPP
#define NORMAL_CUMMULATIVE_DISTRIBUTION_FUNCTION_HPP

#include <cmath>
#include "GP/DataTypes.hpp"

namespace GPOM {
	Scalar normcdf(Scalar x)
	{
		// constants
		Scalar a1 =  0.254829592f;
		Scalar a2 = -0.284496736f;
		Scalar a3 =  1.421413741f;
		Scalar a4 = -1.453152027f;
		Scalar a5 =  1.061405429f;
		Scalar p  =  0.3275911f;

		// Save the sign of x
		int sign = 1;
		if (x < 0)
			sign = -1;
		x = fabs(x)/sqrt((Scalar) 2.f);

		// A&S formula 7.1.26
		Scalar t = ((Scalar) 1.f)/(((Scalar) 1.f) + p*x);
		Scalar y = ((Scalar) 1.f) - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x);

		return ((Scalar) 0.5f) * (((Scalar) 1.f) + ((Scalar) sign)*y);
	}

}
#endif
