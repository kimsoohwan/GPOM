#ifndef TRAINING_DATA_SETTER_DERIVATIVES_HPP
#define TRAINING_DATA_SETTER_DERIVATIVES_HPP

#include "GP/DataTypes.hpp"

namespace GPOM{

	class TrainingDataSetterDerivatives
	{
	public:
		// setter
		bool setTrainingData(MatrixConstPtr pXd, MatrixConstPtr pX, VectorConstPtr pYall)
		{
			// check if the training inputs are the same
			if(m_pXd == pXd) return false;
			m_pXd		= pXd;

			// numbers
			if(PointMatrixDirection::fRowWisePointsMatrix)
			{
				m_nd		= pXd->rows();
				m_n			= pX->rows();
				m_d			= pXd->cols();
				assert(getN() == pYall->size());
			}
			else
			{
				m_nd		= pXd->cols();
				m_n			= pX->cols();
				m_d			= pXd->rows();
				assert(getN() == pYall->size());
			}

			return true;
		}

		// getter
		int	getN() const { return m_nd*(m_d+1) + m_n; }

	protected:
		int											m_nd;			// number of derivative training inputs
		MatrixConstPtr						m_pXd;		// training inputs
	};

}

#endif