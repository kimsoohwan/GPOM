#ifndef Gaussian_Process_HPP
#define Gaussian_Process_HPP

#include "DataTypes.hpp"
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

	// prediction
	void predict(MeanHypConstPtr			pMeanLogHyp, 
						 CovHypConstPtr				pCovLogHyp, 
						 LikHypConstPtr				pLikCovLogHyp, 
						 MatrixConstPtr					pX, 
						 VectorConstPtr					pY, 
						 MatrixConstPtr					pXs, 
						 VectorPtr							&pMu, 
						 MatrixPtr							&pSigma, 
						 bool									fVarianceVector = true)
	{
		m_InfMethod.predict(pMeanLogHyp, pCovLogHyp, pLikCovLogHyp, 
											 pX, pY, pXs, 
											 pMu, pSigma, fVarianceVector);
	}

	//// prediction
	//void predict(MeanHypConstPtr			pMeanLogHyp, 
	//					 CovHypConstPtr				pCovLogHyp, 
	//					 LikHypConstPtr				pLikCovLogHyp, 
	//					 MatrixConstPtr					pX, 
	//					 VectorConstPtr					pY, 
	//					 MatrixConstPtr					pXs, 
	//					 VectorPtr							&pMu, 
	//					 MatrixPtr							&pSigma, 
	//					 bool									fVarianceVector = true)
	//{
	//	m_InfMethod.predict(pMeanLogHyp, pCovLogHyp, pLikCovLogHyp, 
	//										 pX, pY, pXDerivatives, pYDerivatives, pXs, 
	//										 pMu, pSigma, fVarianceVector);
	//}

	// nlZ, dnlZ
	void negativeLogMarginalLikelihood(MeanHypConstPtr			pMeanLogHyp, 
																	CovHypConstPtr			pCovLogHyp, 
																	LikHypConstPtr				pLikCovLogHyp, 
																	MatrixConstPtr				pX, 
																	VectorConstPtr				pY,
																	Scalar								&nlZ, 
																	VectorPtr						&pDnlZ,
																	const int							calculationMode = 0)
	{
		m_InfMethod.negativeLogMarginalLikelihood(pMeanLogHyp, pCovLogHyp, pLikCovLogHyp, 
																					   pX, pY, 
																					   nlZ, pDnlZ,
																					   calculationMode);
	}

	//// nlZ, dnlZ
	//void negativeLogMarginalLikelihood(MeanHypConstPtr			pMeanLogHyp, 
	//																CovHypConstPtr			pCovLogHyp, 
	//																LikHypConstPtr				pLikCovLogHyp, 
	//																MatrixConstPtr				pX, 
	//																VectorConstPtr				pY,
	//																MatrixConstPtr				pXDerivatives, 
	//																VectorConstPtr				pYDerivatives, 
	//																Scalar								&nlZ, 
	//																VectorPtr						&pDnlZ,
	//																const int							calculationMode = 0)
	//{
	//	m_InfMethod.negativeLogMarginalLikelihood(pMeanLogHyp, pCovLogHyp, pLikCovLogHyp, 
	//																				   pX, pY, pXDerivatives, pYDerivatives, 
	//																				   nlZ, pDnlZ,
	//																				   calculationMode);
	//}

	// train hyperparameters
	template<class SearchStrategy, class StoppingStrategy>
	void train(MatrixConstPtr			pX, 
					 VectorConstPtr			pY,
					 MeanHypPtr				pMeanLogHyp, 
					 CovHypPtr					pCovLogHyp, 
					 LikHypPtr					pLikCovLogHyp)
	{
		Trainer<MeanFunc, CovFunc, LikFunc, InfMethod> trainer;
		trainer.train<SearchStrategy, StoppingStrategy>(*this, pX, pY,
																							pMeanLogHyp, pCovLogHyp, pLikCovLogHyp);
	}

	//// train hyperparameters
	//template<class SearchStrategy, class StoppingStrategy>
	//void train(MatrixConstPtr			pX, 
	//				 VectorConstPtr			pY,
	//				 MatrixConstPtr			pXDerivatives, 
	//				 VectorConstPtr			pYDerivatives, 
	//				 MeanHypPtr				pMeanLogHyp, 
	//				 CovHypPtr					pCovLogHyp, 
	//				 LikHypPtr					pLikCovLogHyp)
	//{
	//	Trainer<MeanFunc, CovFunc, LikFunc, InfMethod> trainer;
	//	trainer.train<SearchStrategy, StoppingStrategy>(*this, pX, pY, pXDerivatives, pYDerivatives, 
	//																						pMeanLogHyp, pCovLogHyp, pLikCovLogHyp);
	//}

protected:
	// GP setting
	InfMethod<MeanFunc, CovFunc, LikFunc>			m_InfMethod;
};

}

#endif 