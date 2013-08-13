#ifndef COVARIANCE_FUNCTION_SQUARED_EXPONENTIAL_ISO_BASE_HPP
#define COVARIANCE_FUNCTION_SQUARED_EXPONENTIAL_ISO_BASE_HPP

#include <vector>

#include "GP/DataTypes.hpp"
#include "GP/util/sqDistances.hpp"

namespace GPOM{

class CovSEisoBase
{
	protected:
		// type
		typedef		std::vector<MatrixPtr>						DeltaList;
		typedef		const std::vector<MatrixPtr>			ConstDeltaList;

		// constructor
		CovSEisoBase() { }

		// destructor
		virtual ~CovSEisoBase() { }

	protected:
		// pre-calculate the squared distances
		bool preCalculateSqDist(MatrixConstPtr pX)
		{
			// check if the training inputs are the same
			if(m_pTrainingInputsForSqDist == pX) return false;
			m_pTrainingInputsForSqDist = pX;

			// pre-calculate the squared distances
			m_pSqDist = selfSqDistances(pX);
			return true;
		}

		// pre-calculate the delta
		bool preCalculateSqDistAndDelta(MatrixConstPtr pX)
		{
			// dimension
			const int d = PointMatrixDirection::fRowWisePointsMatrix ? pX->cols() : pX->rows();

			// pre-calculate the squared distances
			preCalculateSqDist(pX);

			// check if the training inputs are the same
			if(m_pTrainingInputsForDelta == pX) return false;
			m_pTrainingInputsForDelta = pX;

			// pre-calculate the delta
			m_pDelta.resize(d);
			for(int i = 0; i < d; i++) m_pDelta[i] = selfDelta(pX, i);
			
			return true;
		}

	protected:
		MatrixConstPtr						m_pTrainingInputsForSqDist;			// training inputs
		MatrixPtr									m_pSqDist;										// squared distances of the training inputs

		MatrixConstPtr						m_pTrainingInputsForDelta;			// training inputs
		DeltaList									m_pDelta;											// x_i - x_i' of the training inputs
};
}

#endif
