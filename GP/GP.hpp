#ifndef Gaussian_Process_HPP
#define Gaussian_Process_HPP

#include "GP/DataTypes.hpp"
#include "Trainer.hpp"

namespace GPOM{

	template <class MeanFunc, class CovFunc, class LikFunc, 
					  template <class, class, class> class InfMethod>
	class GaussianProcess
	{
	public:
		typedef	typename MeanFunc::Hyp						MeanHyp;
		typedef	typename MeanFunc::HypPtr					MeanHypPtr;
		typedef	typename MeanFunc::HypConstPtr		MeanHypConstPtr;

		typedef	typename CovFunc::Hyp							CovHyp;
		typedef	typename CovFunc::HypPtr						CovHypPtr;
		typedef	typename CovFunc::HypConstPtr			CovHypConstPtr;

		typedef	typename LikFunc::Hyp							LikHyp;
		typedef	typename LikFunc::HypPtr						LikHypPtr;
		typedef	typename LikFunc::HypConstPtr				LikHypConstPtr;
		
	public:
		// constructor
		GaussianProcess() { }

		// destructor
		virtual ~GaussianProcess() { }

		// setter
		bool setTrainingData(MatrixConstPtr pX, VectorConstPtr pY)
		{
			return m_InfMethod.setTrainingData(pX, pY);
		}

		bool setTrainingData(MatrixConstPtr pXd, MatrixConstPtr pX, VectorConstPtr pYall)
		{
			return m_InfMethod.setTrainingData(pXd, pX, pYall);
		}

		// prediction
		void predict(MeanHypConstPtr			pMeanLogHyp, 
							 CovHypConstPtr				pCovLogHyp, 
							 LikHypConstPtr				pLikCovLogHyp, 
							 MatrixConstPtr					pXs, 
							 VectorPtr							&pMu, 
							 MatrixPtr							&pSigma, 
							 bool									fVarianceVector = true)
		{
			m_InfMethod.predict(pMeanLogHyp, pCovLogHyp, pLikCovLogHyp, pXs, 
												 pMu, pSigma, fVarianceVector);
		}

		// nlZ, dnlZ
		// calculationMode
		// [0]: calculate both nlZ and pDnlZ
		// [+]: calculate nlZ only
		// [-]: calculate pDnlZ only
		void negativeLogMarginalLikelihood(MeanHypConstPtr			pMeanLogHyp, 
																		CovHypConstPtr			pCovLogHyp, 
																		LikHypConstPtr				pLikCovLogHyp, 
																		Scalar								&nlZ, 
																		VectorPtr						&pDnlZ,
																		const int							calculationMode = 0)
		{
			m_InfMethod.negativeLogMarginalLikelihood(pMeanLogHyp, pCovLogHyp, pLikCovLogHyp, 
																						   nlZ, pDnlZ,
																						   calculationMode);
		}

		// train hyperparameters
		template<class SearchStrategy, class StoppingStrategy>
		void train(MeanHypPtr				pMeanLogHyp, 
						 CovHypPtr					pCovLogHyp, 
						 LikHypPtr					pLikCovLogHyp,
						 const int						maxIter = 0,
						 const double				minValue = 1e-7)
		{
			Trainer<MeanFunc, CovFunc, LikFunc, InfMethod> trainer;
			trainer.train<SearchStrategy, StoppingStrategy>(*this, 
																								pMeanLogHyp, pCovLogHyp, pLikCovLogHyp, maxIter, minValue);
		}

	protected:
		// GP setting
		InfMethod<MeanFunc, CovFunc, LikFunc>			m_InfMethod;
	};

}

#endif 