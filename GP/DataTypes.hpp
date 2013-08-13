#ifndef DATA_TYPES_HPP
#define DATA_TYPES_HPP

#include <boost/shared_ptr.hpp>

#include <Eigen/Core>
//#include <Eigen/Sparse>

namespace GPOM{
	class PointMatrixDirection{
	public:
		static const bool fRowWisePointsMatrix = false;
	};

	// Scalar
	//typedef	float																									Scalar;
	typedef	double																								Scalar;

	// Matrix
	typedef	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>		Matrix;
	typedef	boost::shared_ptr<Matrix>															MatrixPtr;
	typedef	boost::shared_ptr<const Matrix>													MatrixConstPtr;

	// Vector
	typedef	Eigen::Matrix<Scalar, Eigen::Dynamic, 1>								Vector;
	typedef	boost::shared_ptr<Vector>															VectorPtr;
	typedef	boost::shared_ptr<const Vector>												VectorConstPtr;

	// CholeskyFactor
	typedef	Eigen::LLT<Matrix>																		CholeskyFactor;
	typedef	boost::shared_ptr<CholeskyFactor>											CholeskyFactorPtr;
	typedef	boost::shared_ptr<const CholeskyFactor>								CholeskyFactorConstPtr;

	// EPS
	const Scalar EPSILON		= 1e-16;
	const Scalar SQRT3			= 1.732050807568877;
}

#endif