#ifndef MEAN_FUNCTION_ZERO_HPP
#define MEAN_FUNCTION_ZERO_HPP

#include "GP/DataTypes.hpp"

namespace GPOM{

	class MeanZero
	{
	public:
		// hyperparameters
		typedef	Eigen::Matrix<Scalar, 0, 1>									Hyp;
		typedef	boost::shared_ptr<Hyp>								HypPtr;
		typedef	boost::shared_ptr<const Hyp>					HypConstPtr;

	public:
		// constructor
		MeanZero() { }

		// destructor
		virtual ~MeanZero() { }

		// mean and derivatives
		VectorPtr operator()(MatrixConstPtr pX, HypConstPtr pLogHyp, const int pdIndex = -1) const
		{
			// number of training data
			const int n  = PointMatrixDirection::fRowWisePointsMatrix ? pX->rows()   : pX->cols();

			VectorPtr mu(new Vector(n));
			mu->setZero();
			return mu;
		}
	};
}

#endif