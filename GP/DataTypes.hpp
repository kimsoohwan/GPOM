#ifndef DATA_TYPES_HPP
#define DATA_TYPES_HPP

#include <boost/shared_ptr.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>
//#include <Eigen/IterativeLinearSolvers>

namespace GPOM{
	class PointMatrixDirection
	{
		public:
			static const bool fRowWisePointsMatrix = false;
	};

	// Scalar
	typedef	float																									Scalar;
	//typedef	double																								Scalar;

	// Matrix
	typedef	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>		Matrix;
	typedef	boost::shared_ptr<Matrix>															MatrixPtr;
	typedef	boost::shared_ptr<const Matrix>													MatrixConstPtr;

	// Sparse Matrix
	typedef	Eigen::SparseMatrix<Scalar>														SparseMatrix;
	typedef	boost::shared_ptr<SparseMatrix>												SparseMatrixPtr;
	typedef	boost::shared_ptr<const SparseMatrix>									SparseMatrixConstPtr;

	// Vector
	typedef	Eigen::Matrix<Scalar, Eigen::Dynamic, 1>								Vector;
	typedef	boost::shared_ptr<Vector>															VectorPtr;
	typedef	boost::shared_ptr<const Vector>												VectorConstPtr;

	// CholeskyFactor
	typedef	Eigen::LLT<Matrix>												CholeskyFactor;
	typedef	Eigen::ConjugateGradient<Matrix, Eigen::Upper>					CGCholeskyFactor;
	//typedef	Eigen::SimplicialLDLT<SparseMatrix, Eigen::Upper>			CholeskyFactor;
	//typedef	Eigen::ConjugateGradient<SparseMatrix, Eigen::Upper>		CGCholeskyFactor;

	// EPS
	const Scalar EPSILON		= 1e-16f;
	const Scalar SQRT3			= 1.732050807568877f;
	const Scalar PI					= 3.141592653589793f;
	const Scalar TWO_PI		= 6.283185307179586f;
	const Scalar PI_CUBED	= 31.006276680299816f;
}

#endif