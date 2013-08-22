#ifndef INTEGER_TO_STRING_HPP
#define INTEGER_TO_STRING_HPP

#include <sstream>

namespace GPOM {

	std::string to_string(const int i)
	{
		std::stringstream ss;
		std::string s;
		ss << i;
		s = ss.str();

		return s;
	}
}

#endif