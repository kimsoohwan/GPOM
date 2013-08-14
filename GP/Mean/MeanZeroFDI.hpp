#ifndef MEAN_FUNCTION_ZERO_BETWEEN_FUNCTION_VALUE_DERIVATIVE_AND_INTEGRAL_HPP
#define MEAN_FUNCTION_ZERO_BETWEEN_FUNCTION_VALUE_DERIVATIVE_AND_INTEGRAL_HPP

#include "GP/util/TrainingInputSetterDerivatives.hpp"
#include "GP/Mean/MeanZero.hpp"

namespace GPOM{

	class MeanZeroFDI : public MeanZero, public TrainingInputSetterDerivatives
	{
		public:
			// constructor
			MeanZeroFDI() { }

			// destructor
			virtual ~MeanZeroFDI() { }

		// setter
		private:
			bool setTrainingInputs(MatrixConstPtr pX) { }
		public:
			bool setTrainingInputs(MatrixConstPtr pXd, MatrixConstPtr pX)
			{
				// pX: function observations
				// pXd: function and derivative observations

				assert(PointMatrixDirection::fRowWisePointsMatrix ? pX->cols() == pXd->cols() : pX->rows() == pXd->rows());

				// check if the training inputs are the same
				bool fDifferentX = TrainingInputSetter::setTrainingInputs(pX);
				bool fDifferentXd = TrainingInputSetterDerivatives::setTrainingInputs(pXd);
				if(!fDifferentX && !fDifferentXd)		return false;

				return true;
			}

			// getter
			virtual int	getN() const { return m_nd*(m_d+1)+m_n; }
	};

}

#endif