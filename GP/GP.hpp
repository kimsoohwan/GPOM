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
		typedef	typename CovFunc::Hyp							CovHyp;
		typedef	typename LikFunc::Hyp							LikHyp;
		
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
		void predict(const MeanHyp				&meanLogHyp, 
							 const CovHyp					&covLogHyp, 
							 const LikHyp						&likCovLogHyp, 
							 MatrixConstPtr					pXs, 
							 VectorPtr							&pMu, 
							 MatrixPtr							&pSigma, 
							 bool									fVarianceVector = true)
		{
			std::cout << "GP::predict" << std::endl;
			m_InfMethod.predict(meanLogHyp, covLogHyp, likCovLogHyp, pXs, 
												 pMu, pSigma, fVarianceVector);
		}

		// nlZ, dnlZ
		// calculationMode
		// [0]: calculate both nlZ and pDnlZ
		// [+]: calculate nlZ only
		// [-]: calculate pDnlZ only
		void negativeLogMarginalLikelihood(const MeanHyp				&meanLogHyp, 
																		const CovHyp				&covLogHyp, 
																		const LikHyp					&likCovLogHyp, 
																		Scalar								&nlZ, 
																		VectorPtr						&pDnlZ,
																		const int							calculationMode = 0)
		{
			m_InfMethod.negativeLogMarginalLikelihood(meanLogHyp, covLogHyp, likCovLogHyp, 
																						   nlZ, pDnlZ,
																						   calculationMode);
		}

		// train hyperparameters
		template<class SearchStrategy, class StoppingStrategy>
		void train(MeanHyp						&meanLogHyp, 
						 CovHyp							&covLogHyp, 
						 LikHyp							&likCovLogHyp, 
						 const int							maxIter = 0,
						 const double					minValue = 1e-7)
		{
			Trainer<MeanFunc, CovFunc, LikFunc, InfMethod> trainer;
			trainer.train<SearchStrategy, StoppingStrategy>(*this, 
																								meanLogHyp, covLogHyp, likCovLogHyp, maxIter, minValue);
		}

	protected:
		// GP setting
		InfMethod<MeanFunc, CovFunc, LikFunc>			m_InfMethod;
	};

}

#endif 