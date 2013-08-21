#ifndef HYPER_PARAMETER_TRAINER_HPP
#define HYPER_PARAMETER_TRAINER_HPP

#include <climits>								// for std::numeric_limits<Scalar>::infinity()
#include <dlib/optimization.h>			// for dlib::find_min

#include "GP/DataTypes.hpp"

namespace GPOM{

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
	typedef	typename GaussianProcessType::CovHyp						CovHyp;
	typedef	typename GaussianProcessType::LikHyp							LikHyp;

	//typedef	dlib::matrix<Scalar, 0, 1>														DlibVector;	
	typedef	dlib::matrix<double, 0, 1>														DlibVector;	


// inner class
public:
	// nlZ
	class NlZ
	{
	public:
		NlZ(GaussianProcessType &gp)
			: m_gp(gp)
		{
		}
		~NlZ() { }

		//Scalar operator()(const DlibVector &hyp) const
		double operator()(const DlibVector &hyp) const
		{
			// conversion from Dlib to Eigen vectors
			MeanHyp				meanLogHyp;
			CovHyp					covLogHyp;
			LikHyp					likCovLogHyp;
			Dlib2Eigen(hyp, meanLogHyp, covLogHyp, likCovLogHyp);

			// calculate nlZ only
			Scalar				nlZ;
			VectorPtr		pDnlZ;
			m_gp.negativeLogMarginalLikelihood(meanLogHyp, covLogHyp, likCovLogHyp, 
																			   nlZ, 
																			   pDnlZ,
																			   1);

			std::cout << "hyp.mean = " << std::endl << meanLogHyp << std::endl << std::endl;
			std::cout << "hyp.cov = " << std::endl << covLogHyp << std::endl << std::endl;
			std::cout << "hyp.lik = " << std::endl << likCovLogHyp << std::endl << std::endl;
			std::cout << "nlz = " << nlZ << std::endl;
			return nlZ;
		}

	protected:
		GaussianProcessType			&m_gp;
	};

	// dnlZ
	class DnlZ
	{
	public:
		DnlZ(GaussianProcessType &gp)
			: m_gp(gp)
		{
		}
		~DnlZ() { }

		DlibVector operator()(const DlibVector &hyp) const
		{
			// conversion from Dlib to Eigen vectors
			MeanHyp				meanLogHyp;
			CovHyp					covLogHyp;
			LikHyp					likCovLogHyp;
			Dlib2Eigen(hyp, meanLogHyp, covLogHyp, likCovLogHyp);

			// calculate dnlZ only
			Scalar				nlZ;
			VectorPtr		pDnlZ;
			m_gp.negativeLogMarginalLikelihood(meanLogHyp, covLogHyp, likCovLogHyp, 
																			   nlZ,
																			   pDnlZ,
																			   -1);

			std::cout << "hyp.mean = " << std::endl << meanLogHyp << std::endl << std::endl;
			std::cout << "hyp.cov = " << std::endl << covLogHyp << std::endl << std::endl;
			std::cout << "hyp.lik = " << std::endl << likCovLogHyp << std::endl << std::endl;
			std::cout << "dnlz = " << std::endl << *pDnlZ << std::endl << std::endl;

			DlibVector dnlZ(meanLogHyp.size() + covLogHyp.size() + likCovLogHyp.size());
			Eigen2Dlib(pDnlZ, dnlZ);
			return dnlZ;
		}

	protected:
		GaussianProcessType			&m_gp;
	};

// method
public:

	// train hyperparameters
	template<class SearchStrategy, class StoppingStrategy>
	void train(GaussianProcessType		&gp,
					 MeanHyp								&meanLogHyp, 
					 CovHyp									&covLogHyp, 
					 LikHyp									&likCovLogHyp,
					 const int									maxIter = 0,
					 const double							minValue = 1e-7)
	{
		// maxIter
		// [+]:		max iteration criteria on
		// [0, -]:		max iteration criteria off

		// set training data
		NlZ								nlZ(gp);
		DnlZ							dnlZ(gp);

		// hyperparameters
		DlibVector hyp;
		hyp.set_size(meanLogHyp.size() + covLogHyp.size() + likCovLogHyp.size());
		//std::cout << "number of hyperparameters" << std::endl;
		//std::cout << "mean: " << pMeanLogHyp->size() << std::endl;
		//std::cout << "cov: " << pCovLogHyp->size() << std::endl;
		//std::cout << "lik: " << pLikCovLogHyp->size() << std::endl;
		//std::cout << "total number of hyperparameters: " << pMeanLogHyp->size() + pCovLogHyp->size() + pLikCovLogHyp->size() << std::endl;

		// initialization
		Eigen2Dlib(meanLogHyp, covLogHyp, likCovLogHyp, hyp);

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
		Dlib2Eigen(hyp, meanLogHyp, covLogHyp, likCovLogHyp);
	}

protected:
	// conversion between Eigen and Dlib vectors
	static void Eigen2Dlib(VectorConstPtr				pVector,
										   DlibVector						&vec)
	{
		for(int i = 0;		i < pVector->size();		i++)		vec(i, 0) = (*pVector)(i);
	}

	static void Eigen2Dlib(const MeanHyp				&meanLogHyp, 
											const CovHyp				&covLogHyp, 
											const LikHyp					&likCovLogHyp,
											DlibVector						&hyp)
	{
		int j = 0; // hyperparameter index
		for(int i = 0;		i < meanLogHyp.size();		i++)		hyp(j++, 0) = meanLogHyp(i);
		for(int i = 0;		i < covLogHyp.size();			i++)		hyp(j++, 0) = covLogHyp(i);
		for(int i = 0;		i < likCovLogHyp.size();		i++)		hyp(j++, 0) = likCovLogHyp(i);
	}

	static void Dlib2Eigen(const DlibVector				&hyp,
											MeanHyp							&meanLogHyp, 
											CovHyp								&covLogHyp, 
											LikHyp								&likCovLogHyp)
	{
		int j = 0; // hyperparameter index
		for(int i = 0;		i < meanLogHyp.size();		i++)		meanLogHyp(i)		= hyp(j++, 0);
		for(int i = 0;		i < covLogHyp.size();			i++)		covLogHyp(i)			= hyp(j++, 0);
		for(int i = 0;		i < likCovLogHyp.size();		i++)		likCovLogHyp(i)		= hyp(j++, 0);
	}

public:
	// constructor
	Trainer() { }

	// destructor
	virtual ~Trainer() { }
};

}

#endif 