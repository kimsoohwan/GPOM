#ifndef TRAINING_DATA_SETTER_HPP
#define TRAINING_DATA_SETTER_HPP

#include "GP/DataTypes.hpp"

namespace GPOM{

	class TrainingDataSetter
	{
	public:
		// setter
		bool setTrainingData(MatrixConstPtr pX, VectorConstPtr pY)
		{
			// check if the training inputs are the same
			if(m_pX == pX && m_pY == pY) return false;
			m_pX = pX;
			m_pY = pY;

			// numbers
			if(PointMatrixDirection::fRowWisePointsMatrix)
			{
				//assert(pX->rows() == pY->size());
				m_n		= pX->rows();
				m_d		= pX->cols();
			}
			else
			{
				//assert(pX->cols() == pY->size());
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
		VectorConstPtr						m_pY;		// training outputs
	};

}

#endif