#ifndef HYPER_PARAMETER_TRAINER_HPP
#define HYPER_PARAMETER_TRAINER_HPP

#include <climits>								// for std::numeric_limits<Scalar>::infinity()
#include <dlib/optimization.h>			// for dlib::find_min

#include "DataTypes.hpp"

namespace GP{

// Gaussian Process
template <class MeanFunc, class CovFunc, class LikFunc, 
				  template <class, class, class> class InfMethod>
class GaussianProcess;

// Search Strategy
class CG		{	public:		typedef dlib::cg_search_strategy			Type; };
class BFGS	{	public:		typedef dlib::bfgs_search_strategy			Type; };
class LBFGS	{	public:		typedef dlib::lbfgs_search_strategy		Type; };

// Stopping Strategy
class DeltaFunc			{	public:		typedef dlib::objective_delta_stop_strategy			Type; };
class GradientNorm	{	public:		typedef dlib::gradient_norm_stop_strategy			Type; };

// Trainer
template <class MeanFunc, class CovFunc, class LikFunc, 
				  template <class, class, class> class InfMethod>
class Trainer
{
// typedef
public:
	typedef	typename Trainer<MeanFunc, CovFunc, LikFunc, InfMethod>							TrainerType;
	typedef	typename GaussianProcess<MeanFunc, CovFunc, LikFunc, InfMethod>			GaussianProcessType;

	typedef	typename GaussianProcessType::MeanHyp					MeanHyp;
	typedef	typename GaussianProcessType::MeanHypPtr				MeanHypPtr;
	typedef	typename GaussianProcessType::MeanHypConstPtr		MeanHypConstPtr;

	typedef	typename GaussianProcessType::CovHyp						CovHyp;
	typedef	typename GaussianProcessType::CovHypPtr					CovHypPtr;
	typedef	typename GaussianProcessType::CovHypConstPtr		CovHypConstPtr;

	typedef	typename GaussianProcessType::LikHyp							LikHyp;
	typedef	typename GaussianProcessType::LikHypPtr					LikHypPtr;
	typedef	typename GaussianProcessType::LikHypConstPtr			LikHypConstPtr;

	typedef	dlib::matrix<Scalar, 0, 1>														DlibVector;	


// inner class
public:
	// nlZ
	class NlZ
	{
	public:
		NlZ(TrainerType &parent, GaussianProcessType &gp)
			: m_parent(parent), m_gp(gp)
		{
		}
		~NlZ() { }
		void setTrainingData(MatrixConstPtr		pX,
											 VectorConstPtr		pY)
		{
			m_pX = pX;
			m_pY = pY;
		}

		Scalar operator()(const DlibVector &hyp) const
		{
			// conversion from Dlib to Eigen vectors
			MeanHypPtr				pMeanLogHyp(new MeanHyp());
			CovHypPtr					pCovLogHyp(new CovHyp());
			LikHypPtr					pLikCovLogHyp(new LikHyp());
			m_parent.Dlib2Eigen(hyp, pMeanLogHyp, pCovLogHyp, pLikCovLogHyp);

			// calculate nlZ only
			Scalar				nlZ;
			VectorPtr		pDnlZ;
			m_gp.negativeLogMarginalLikelihood(pMeanLogHyp, pCovLogHyp,pLikCovLogHyp, 
																			   m_pX, m_pY,
																			   nlZ, 
																			   pDnlZ,
																			   1);

			return nlZ;
		}

	protected:
		TrainerType							&m_parent;
		GaussianProcessType			&m_gp;
		MatrixConstPtr						m_pX; 
		VectorConstPtr						m_pY;
	};

	// dnlZ
	class DnlZ
	{
	public:
		DnlZ(TrainerType &parent, GaussianProcessType &gp)
			: m_parent(parent), m_gp(gp)
		{
		}
		~DnlZ() { }
		void setTrainingData(MatrixConstPtr		pX,
											 VectorConstPtr		pY)
		{
			m_pX = pX;
			m_pY = pY;
		}

		DlibVector operator()(const DlibVector &hyp) const
		{
			// conversion from Dlib to Eigen vectors
			MeanHypPtr				pMeanLogHyp(new MeanHyp());
			CovHypPtr					pCovLogHyp(new CovHyp());
			LikHypPtr					pLikCovLogHyp(new LikHyp());
			m_parent.Dlib2Eigen(hyp, pMeanLogHyp, pCovLogHyp, pLikCovLogHyp);

			// calculate dnlZ only
			Scalar				nlZ;
			VectorPtr		pDnlZ;
			m_gp.negativeLogMarginalLikelihood(pMeanLogHyp, pCovLogHyp,pLikCovLogHyp, 
																			   m_pX, m_pY,
																			   nlZ,
																			   pDnlZ,
																			   -1);

			DlibVector dnlZ(pMeanLogHyp->size() + pCovLogHyp->size() + pLikCovLogHyp->size());
			m_parent.Eigen2Dlib(pDnlZ, dnlZ);
			return dnlZ;
		}

	protected:
		TrainerType							&m_parent;
		GaussianProcessType			&m_gp;
		MatrixConstPtr						m_pX; 
		VectorConstPtr						m_pY;
	};

// method
public:
	// train hyperparameters
	template<class SearchStrategy, class StoppingStrategy>
	void train(GaussianProcessType		&gp,
					 MatrixConstPtr						pX, 
					 VectorConstPtr						pY,
					 MeanHypPtr							pMeanLogHyp, 
					 CovHypPtr								pCovLogHyp, 
					 LikHypPtr								pLikCovLogHyp,
					 const int									maxIter = 0,
					 const double							minValue = 1e-7)
	{
		// maxIter
		// [+]:		max iteration criteria on
		// [0, -]:		max iteration criteria off

		// set training data
		NlZ								nlZ(*this, gp);
		DnlZ							dnlZ(*this, gp);
		nlZ.setTrainingData(pX, pY);
		dnlZ.setTrainingData(pX, pY);

		// hyperparameters
		DlibVector hyp;
		hyp.set_size(pMeanLogHyp->size() + pCovLogHyp->size() + pLikCovLogHyp->size());

		// initialization
		Eigen2Dlib(pMeanLogHyp, pCovLogHyp, pLikCovLogHyp, hyp);

		// find minimum
		if(maxIter <= 0)
		{
			dlib::find_min(SearchStrategy::Type(),
									StoppingStrategy::Type(minValue).be_verbose(),
									nlZ, 
									dnlZ,
									hyp, -std::numeric_limits<Scalar>::infinity());
		}
		else
		{
			dlib::find_min(SearchStrategy::Type(),
									StoppingStrategy::Type(minValue, maxIter).be_verbose(),
									nlZ, 
									dnlZ,
									hyp, -std::numeric_limits<Scalar>::infinity());
		}

		// conversion  from Dlib to Eigen vectors
		Dlib2Eigen(hyp, pMeanLogHyp, pCovLogHyp, pLikCovLogHyp);
	}

protected:
	// conversion between Eigen and Dlib vectors
	void Eigen2Dlib(VectorConstPtr				pVector,
								 DlibVector						&vec) const
	{
		for(int i = 0;		i < pVector->size();		i++)		vec(i, 0) = (*pVector)(i);
	}

	void Eigen2Dlib(MeanHypConstPtr				pMeanLogHyp, 
								 CovHypConstPtr					pCovLogHyp, 
								 LikHypConstPtr					pLikCovLogHyp,
								 DlibVector								&hyp) const
	{
		int j = 0; // hyperparameter index
		for(int i = 0;		i < pMeanLogHyp->size();		i++)		hyp(j++, 0) = (*pMeanLogHyp)(i);
		for(int i = 0;		i < pCovLogHyp->size();			i++)		hyp(j++, 0) = (*pCovLogHyp)(i);
		for(int i = 0;		i < pLikCovLogHyp->size();		i++)		hyp(j++, 0) = (*pLikCovLogHyp)(i);
	}

	void Dlib2Eigen(const DlibVector					&hyp,
								 MeanHypPtr							pMeanLogHyp, 
								 CovHypPtr								pCovLogHyp, 
								 LikHypPtr								pLikCovLogHyp) const
	{
		int j = 0; // hyperparameter index
		for(int i = 0;		i < pMeanLogHyp->size();		i++)		(*pMeanLogHyp)(i)		= hyp(j++, 0);
		for(int i = 0;		i < pCovLogHyp->size();			i++)		(*pCovLogHyp)(i)			= hyp(j++, 0);
		for(int i = 0;		i < pLikCovLogHyp->size();		i++)		(*pLikCovLogHyp)(i)		= hyp(j++, 0);
	}

public:
	// constructor
	Trainer() { }

	// destructor
	virtual ~Trainer() { }
};

}

#endif 