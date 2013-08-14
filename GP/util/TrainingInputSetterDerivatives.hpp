#ifndef TRAINING_INPUT_SETTER_DERIVATIVES_HPP
#define TRAINING_INPUT_SETTER_DERIVATIVES_HPP

#include "GP/DataTypes.hpp"

namespace GPOM{

	class TrainingInputSetterDerivatives
	{
	public:
		//setter 
		bool setTrainingInputs(MatrixConstPtr pXd)
		{
			// check if the training inputs are the same
			if(m_pXd == pXd) return false;
			m_pXd = pXd;

			// numbers
			m_nd = PointMatrixDirection::fRowWisePointsMatrix ? pXd->rows() : pXd->cols();

			return true;
		}

		// getter
		virtual int	getNd() const { return m_nd; }

	protected:
		int											m_nd;			// number of training inputs
		MatrixConstPtr						m_pXd;		// derivative training inputs
	};

}

#endif