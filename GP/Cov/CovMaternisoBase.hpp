#ifndef COVARIANCE_FUNCTION_MATERN_ISO_BASE_HPP
#define COVARIANCE_FUNCTION_MATERN_ISO_BASE_HPP

#include <vector>

#include "GP/DataTypes.hpp"
#include "GP/util/sqDistances.hpp"

namespace GP{

class CovMaternisoBase
{
protected:
	// type
	typedef		std::vector<MatrixPtr>						DeltaList;
	typedef		const std::vector<MatrixPtr>			ConstDeltaList;

	// constructor
	CovMaternisoBase() { }

	// destructor
	virtual ~CovMaternisoBase() { }

	protected:
		// pre-calculate the squared distances
		bool preCalculateDist(MatrixConstPtr pX)
		{
			// check if the training inputs are the same
			if(m_pTrainingInputsForDist == pX) return false;
			m_pTrainingInputsForDist = pX;

			// pre-calculate the distances
			m_pDist = selfSqDistances(pX);						// squared distances
			m_pDist->noalias() = m_pDist->cwiseSqrt();	// distances
			return true;
		}

		// pre-calculate the delta
		bool preCalculateDistAndDelta(MatrixConstPtr pX)
		{
			// pre-calculate the squared distances
			preCalculateDist(pX);

			// check if the training inputs are the same
			if(m_pTrainingInputsForDelta == pX) return false;
			m_pTrainingInputsForDelta = pX;

			// pre-calculate the delta
			const int d = pX->cols(); // dimension
			m_pDelta.resize(d);
			for(int i = 0; i < d; i++) m_pDelta[i] = selfDelta(pX, i);
			
			return true;
		}

	protected:
		// pre-calculated matrices for speed up in training
		MatrixConstPtr						m_pTrainingInputsForDist;				// training inputs
		MatrixPtr									m_pDist;												// distances of the training inputs

		MatrixConstPtr						m_pTrainingInputsForDelta;			// training inputs
		DeltaList									m_pDelta;											// x_i - x_i' of the training inputs
};
}

#endif
