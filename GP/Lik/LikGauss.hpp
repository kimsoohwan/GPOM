#ifndef LIKELIHOOD_FUNCTION_GAUSSIAN_HPP
#define LIKELIHOOD_FUNCTION_GAUSSIAN_HPP

#include <cmath>

#include "GP/util/TrainingInputSetter.hpp"

namespace GPOM{

	class LikGauss : public TrainingInputSetter
	{
	public:
		// hyperparameters
		typedef	Eigen::Matrix<Scalar, 1, 1>							Hyp;

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
		VectorPtr operator()(const Hyp &logHyp, const int pdIndex = -1) const
		{
			assert(pdIndex < 1);

			// number of training data
			const int n = getN();
			VectorPtr pD(new Vector(n));

			// derivatives w.r.t sn
			if(pdIndex == 0)				pD->fill(((Scalar) 2.f) * exp((Scalar) 2.f * logHyp(0)));

			// likelihood
			else									pD->fill(exp((Scalar) 2.f * logHyp(0)));

			return pD;
		}

		//// diagonal matrix
		//MatrixPtr operator()(MatrixConstPtr pX, const Hyp &logHyp, const int pdIndex = -1) const
		//{
		//	// number of training data
		//	const int n = getN();
		//	MatrixPtr pD(new Matrix(n, n));
		//	pD->setZero();

		//	// derivatives w.r.t sn
		//	if(pdIndex == 0)			pD->diagonal().fill(((Scalar) 2.f) * exp((Scalar) 2.f * logHyp(0)));

		//	// likelihood
		//	else								pD->diagonal().fill(exp((Scalar) 2.f * logHyp(0)));

		//	return pD;
		//}
	};

}

#endif