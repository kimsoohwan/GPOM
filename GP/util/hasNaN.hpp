#ifndef HAS_NAN_HPP
#define HAS_NAN_HPP

#include <iostream>
#include "GP/DataTypes.hpp"

namespace GPOM {

#ifndef isNaN
#define isNaN(x) ((x)!=(x)) 
#endif

	bool hasNaN(MatrixConstPtr pX)
	{
		bool fHasNaN = false;
		for(int row = 0; row < pX->rows(); row++)
		{
			for(int col = 0; col < pX->cols(); col++)
			{
				if(isNaN((*pX)(row, col)))
				{
					std::cout << row << ", " << col << " : nan!" << std::endl;
					fHasNaN = true;
				}
			}
		}

		return fHasNaN;
	}

	void testHasNaN(MatrixConstPtr pX)
	{
		std::cout << "testing ... " << std::endl;
		if(hasNaN(pX))		std::cout << " done: has NaN !!!" << std::endl;
		else							std::cout << " done: good." << std::endl;
	}
}

#endif