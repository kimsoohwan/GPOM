#ifndef INF_EXACT_UNSTABLE_BUT_FAST_BETWEEN_FUNCTION_VALUE_DERIVATIVE_AND_INTEGRAL_HPP
#define INF_EXACT_UNSTABLE_BUT_FAST_BETWEEN_FUNCTION_VALUE_DERIVATIVE_AND_INTEGRAL_HPP

#include "GP/util/TrainingInputSetterDerivatives.hpp"
#include "GP/Inf/InfExactUnstableButFast.hpp"

namespace GPOM{

	template<class MeanFunc, class CovFunc, class LikFunc>
	class InfExactUnstableButFastFDI : public InfExactUnstableButFast<MeanFunc, CovFunc, LikFunc>, public TrainingInputSetterDerivatives
	{
		public:
			// constructor
			InfExactUnstableButFastFDI() { }

			// destructor
			virtual ~InfExactUnstableButFastFDI() { }

			// setter
		private:
			bool setTrainingInputs(MatrixConstPtr pX) { }
			bool setTrainingData(MatrixConstPtr pX, VectorConstPtr pY) { }
		public:
			bool setTrainingData(MatrixConstPtr pXd, MatrixConstPtr pX, VectorConstPtr pYall)
			{
				// pX: function observations
				// pXd: function and derivative observations

				// check if the training inputs are the same
				bool fDifferentF = TrainingDataSetter::setTrainingData(pX, pYall);
				bool fDifferentD = TrainingInputSetterDerivatives::setTrainingInputs(pXd);
				if(!fDifferentF && !fDifferentD)		return false;

				m_MeanFunc.setTrainingInputs(pXd, pX);
				m_CovFunc.setTrainingInputs(pXd, pX);
				m_LikFunc.setTrainingInputs(pXd, pX);

				return true;
			}

			// getter
			virtual int	getN() const { return m_nd*(m_d+1) + m_n; }

	protected:
		// training outputs
		VectorConstPtr		m_pYd;
	};

}

#endif 