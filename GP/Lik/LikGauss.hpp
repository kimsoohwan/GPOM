#ifndef LIKELIHOOD_FUNCTION_GAUSSIAN_HPP
#define LIKELIHOOD_FUNCTION_GAUSSIAN_HPP

#include <cmath>

#include "GP/DataTypes.hpp"

namespace GP{

	class LikGauss
	{
	public:
		// hyperparameters
		typedef	Eigen::Matrix<Scalar, 1, 1>									Hyp;
		typedef	boost::shared_ptr<Hyp>								HypPtr;
		typedef	boost::shared_ptr<const Hyp>					HypConstPtr;

	public:
		// constructor
		LikGauss()
		{
		}

		// destructor
		virtual ~LikGauss()
		{
		}

		// diagonal vector
		VectorPtr operator()(MatrixConstPtr pX, HypConstPtr pLogHyp, const int pdIndex = -1) const
		{
			VectorPtr pD(new Vector(pX->rows()));
			if(pdIndex == -1)			pD->fill(exp((Scalar) 2.f * (*pLogHyp)(0)));
			else									pD->fill(((Scalar) 2.f) * exp((Scalar) 2.f * (*pLogHyp)(0)));
			return pD;
		}

		//// diagonal matrix
		//MatrixPtr operator()(MatrixConstPtr pX, HypConstPtr pLogHyp, const int pdIndex = -1) const
		//{
		//	MatrixPtr pD(new Matrix(pX->rows(), pX->rows()));
		//	pD->setZero();
		//	if(pdIndex == -1)		pD->diagonal().fill(exp((Scalar) 2.f * (*pLogHyp)(0)));
		//	else								pD->diagonal().fill(((Scalar) 2.f) * exp((Scalar) 2.f * (*pLogHyp)(0)));
		//	return pD;
		//}
	};
}

#endif