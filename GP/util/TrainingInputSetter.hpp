#ifndef TRAINING_INPUT_SETTER_HPP
#define TRAINING_INPUT_SETTER_HPP

#include "GP/DataTypes.hpp"

namespace GPOM{

	class TrainingInputSetter
	{
	public:
		// setter
		bool setTrainingInputs(MatrixConstPtr pX)
		{
			// check if the training inputs are the same
			if(m_pX == pX) return false;
			m_pX = pX;

			// numbers
			if(PointMatrixDirection::fRowWisePointsMatrix)
			{
				m_n		= pX->rows();
				m_d		= pX->cols();
			}
			else
			{
				m_n		= pX->cols();
				m_d		= pX->rows();
			}

			return true;
		}

		// getter
		virtual int	getN() const { return m_n; }
		virtual int	getD() const { return m_d; }

	protected:
		int											m_n;		// number of training inputs
		int											m_d;		// dimensions
		MatrixConstPtr						m_pX;		// training inputs
	};

}

#endif