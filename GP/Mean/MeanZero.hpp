#ifndef MEAN_FUNCTION_ZERO_HPP
#define MEAN_FUNCTION_ZERO_HPP

#include "GP/util/TrainingInputSetter.hpp"

namespace GPOM{

	class MeanZero : public TrainingInputSetter
	{
	public:
		// hyperparameters
		typedef	Eigen::Matrix<Scalar, 0, 1>							Hyp;
		typedef	boost::shared_ptr<Hyp>								HypPtr;
		typedef	boost::shared_ptr<const Hyp>					HypConstPtr;

	public:
		// constructor
		MeanZero() { }

		// destructor
		virtual ~MeanZero() { }

		// mean and derivatives
		VectorPtr operator()(HypConstPtr pLogHyp, const int pdIndex = -1) const
		{
			// number of training data
			const int n = getN();
			VectorPtr mu(new Vector(n));
			mu->setZero();
			return mu;
		}

		// Ms
		VectorPtr Ms(MatrixConstPtr pXs, HypConstPtr pLogHyp) const
		{
			// number of training data
			const int m = PointMatrixDirection::fRowWisePointsMatrix ? pXs->rows() : pXs->cols();
			VectorPtr mu(new Vector(m));
			mu->setZero();
			return mu;
		}
	};

}

#endif