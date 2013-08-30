#ifndef INF_EXACT_UNSTABLE_BUT_FAST_HPP
#define INF_EXACT_UNSTABLE_BUT_FAST_HPP

#include <math.h>
#include <Eigen/Cholesky> // for LLT

#include "GP/util/valueCheck.hpp"
#include "GP/util/TrainingDataSetter.hpp"

namespace GPOM{

	template<class MeanFunc, class CovFunc, class LikFunc>
	class InfExactUnstableButFast : public TrainingDataSetter
	{
	public:
		typedef	typename MeanFunc::Hyp						MeanHyp;
		typedef	typename CovFunc::Hyp							CovHyp;
		typedef	typename LikFunc::Hyp							LikHyp;

	public:
		// constructor
		InfExactUnstableButFast() { }

		// destructor
		virtual ~InfExactUnstableButFast() { }

		// setter
		bool setTrainingData(MatrixConstPtr pX, VectorConstPtr pY)
		{
			// check if the training inputs are the same
			if(!TrainingDataSetter::setTrainingData(pX, pY))		return false;

			m_MeanFunc.setTrainingInputs(pX);
			m_CovFunc.setTrainingInputs(pX);
			m_LikFunc.setTrainingInputs(pX);

			return true;
		}

		// prediction
		void predict(const MeanHyp					&meanLogHyp, 
							 const CovHyp						&covLogHyp, 
							 const LikHyp							&likCovLogHyp, 
							 MatrixConstPtr						pXs, 
							 VectorPtr								&pMu, 
							 MatrixPtr								&pSigma, 
							 const bool										fVarianceVector = true,
							 const bool								fBatchProcessing = true)
		{
			//std::cout << "InfExactUnstableButFast::predict" << std::endl;

			// number of data
			assert(m_MeanFunc.getN() == m_CovFunc.getN() && m_CovFunc.getN() == m_LikFunc.getN() && m_LikFunc.getN() == getN());
			const int n = getN();
			const int m = PointMatrixDirection::fRowWisePointsMatrix ? pXs->rows() : pXs->cols();

			// memory allocation
			pMu.reset(new Vector(m));
			if(fVarianceVector)		pSigma.reset(new Matrix(m, 1));	// variance vector (mx1)
			else					pSigma.reset(new Matrix(m, m));	// covariance matrix (mxm)

			// calculate L and alpha
			// Kn = K + D = LL'
			// alpha = inv(K + D)*(y-m)
			calculateLandAlpha(meanLogHyp, covLogHyp, likCovLogHyp);

			// too many test points: batch
			if(fVarianceVector && fBatchProcessing)
			{
				const int mPerBatch = 10000;	// max number of test points per batch
				int mss = 0;					// number of test points per batch
				const int d = PointMatrixDirection::fRowWisePointsMatrix ? pXs->cols() : pXs->rows();
				MatrixPtr pXss;					// part of test points

				// batch processing
				int from	= 0;
				int to		= -1;
				while(to < m)
				{
					// range
					from = to + 1;
					to = (m < from + mPerBatch - 1) ? m : from + mPerBatch - 1;
					std::cout << "predict from " << from << " to " << to << std::endl;

					// allocation
					if(mss != to - from + 1)
					{
						mss = to - from + 1;
						if(PointMatrixDirection::fRowWisePointsMatrix)	pXss.reset(new Matrix(mss, d));
						else											pXss.reset(new Matrix(d, mss));
					}

					// copy
					if(PointMatrixDirection::fRowWisePointsMatrix)	(*pXss) = pXs->block(from, 0, mss, d);
					else											(*pXss) = pXs->block(0, from, d, mss);
				
					// Ks, Kss
					MatrixConstPtr pKs		= m_CovFunc.Ks(pXss, covLogHyp);					// nxm
					MatrixConstPtr pKss		= m_CovFunc.Kss(pXss, covLogHyp, fVarianceVector);	// Vector (mx1) or Matrix (mxm)

					// predictive mean
					// mu = ms + Ks' * inv(Kn) * (y-m)
					//       = ms + Ks' * alpha
					pMu->segment(from, mss).noalias() = *(m_MeanFunc.Ms(pXss, meanLogHyp)) + (pKs->transpose()) * (*m_pAlpha);
					std::cout << "mean" << std::endl;

					// predictive variance
					// Sigma = kss - ks' * inv(Kn) * ks
					//       = kss - sum(ks .* (inv(Kn) * ks), 1)
					//pSigma->block(from, 0, mss, 1).noalias() = (*pKss) - pKs->cwiseProduct(m_L.solve(*pKs)).colwise().sum().transpose();
					pSigma->block(from, 0, mss, 1).noalias() = (*pKss) - pKs->cwiseProduct((*m_pInvKn) * (*pKs)).colwise().sum().transpose();
					std::cout << "variance" << std::endl;
				}
			}
			else
			{
				// Ks, Kss
				MatrixConstPtr pKs		= m_CovFunc.Ks(pXs, covLogHyp);					// nxm
				MatrixConstPtr pKss		= m_CovFunc.Kss(pXs, covLogHyp, fVarianceVector);	// Vector (mx1) or Matrix (mxm)

				// predictive mean
				// mu = ms + Ks' * inv(Kn) * (y-m)
				//       = ms + Ks' * alpha
				pMu->noalias() = *(m_MeanFunc.Ms(pXs, meanLogHyp)) + (pKs->transpose()) * (*m_pAlpha);

				// predictive variance
				if(fVarianceVector)
				{
					// sigma2 = kss - v' * v
					//(*pSigma).noalias() = (*pKss) - pKs->cwiseProduct(m_L.solve(*pKs)).colwise().sum().transpose();
					(*pSigma).noalias() = (*pKss) - pKs->cwiseProduct((*m_pInvKn) * (*pKs)).colwise().sum().transpose();
				}
				else
				{
					// Sigma = Kss - Ks' * inv(Kn) * Ks
					//(*pSigma).noalias() = (*pKss) - pKs->transpose() * (m_L.solve(*pKs));
					(*pSigma).noalias() = (*pKss) - pKs->transpose() * (*m_pInvKn) * (*pKs);
				}
			}
		}


		// nlZ, dnlZ
		void negativeLogMarginalLikelihood(const MeanHyp				&meanLogHyp, 
																		const CovHyp				&covLogHyp, 
																		const LikHyp					&likCovLogHyp, 
																		Scalar								&nlZ, 
																		VectorPtr						&pDnlZ,
																		const int							calculationMode = 0)
		{
			// calculationMode
			// [0]: calculate both nlZ and pDnlZ
			// [+]: calculate nlZ only
			// [-]: calculate pDnlZ only

			std::cout << "meanLogHyp = " << std::endl << meanLogHyp.array().exp() << std::endl << std::endl;
			std::cout << "covLogHyp = " << std::endl << covLogHyp.array().exp() << std::endl << std::endl;
			std::cout << "likCovLogHyp = " << std::endl << likCovLogHyp.array().exp() << std::endl << std::endl;

			// number of training data
			assert(m_MeanFunc.getN() == m_CovFunc.getN() && m_CovFunc.getN() == m_LikFunc.getN() && m_LikFunc.getN() == getN());
			const int n = getN();

			// calculate L and alpha
			// Kn = K + D = LL'
			// alpha = inv(K + sn2*I)*(y-m)
			// Note: if non positivie definite, nlZ = Inf, dnlZ = zeros
			if(!calculateLandAlpha(meanLogHyp, covLogHyp, likCovLogHyp))
			{
				std::cout << "something wrong!" << std::endl;
				if(calculationMode >= 0)	nlZ = BIG_NUMBER;
				if(calculationMode <= 0)
				{
					pDnlZ.reset(new Vector(meanLogHyp.size() + covLogHyp.size() + likCovLogHyp.size()));
					pDnlZ->setZero();
				}
				return;
			}

			// marginal likelihood
			// p(y) = N(m, Kn) = (2pi)^(-n/2) * |Kn|^(-1/2) * exp[(-1/2) * (y-m)' * inv(Kn) * (y-m)]
			// nlZ = (1/2) * (y-m)' * inv(Kn) * (y-m) + (1/2) * log |Kn| + (n/2) * log(2pi)
			if(calculationMode >= 0)
			{
				nlZ = ((Scalar) 0.5f) * ((*m_pY_M).dot(*m_pAlpha)
										 + log(static_cast<Scalar>(m_pKn->determinant())))
					  + ((Scalar) n) * ((Scalar) 0.918938533204673f); // log(2pi)/2 = 0.918938533204673;
				std::cout << "1 = " << std::endl << ((Scalar) 0.5f) * (*m_pY_M).dot(*m_pAlpha) << std::endl << std::endl;
				std::cout << "2 = " << std::endl << ((Scalar) 0.5f) * log(static_cast<Scalar>(m_pKn->determinant())) << std::endl << std::endl;
				std::cout << "3 = " << std::endl << ((Scalar) n) * ((Scalar) 0.918938533204673f) << std::endl << std::endl;
				std::cout << "nlZ = " << nlZ << std::endl;

				// NaN check
				if(isNaN(nlZ))
				{
					std::cout << "nlZ is NaN!" << std::endl;
					nlZ = BIG_NUMBER;
				}

				// Infinity check
				if(isInf(nlZ))
				{
					std::cout << "nlZ is Inf!" << std::endl;
					nlZ = BIG_NUMBER;
				}
			}

			// partial derivatives w.r.t hyperparameters
			if(calculationMode <= 0)
			{
				//std::cout << "dnlZ" << std::endl;

				// derivatives (f_j = partial f / partial x_j)
				int j = 0; // partial derivative index
				pDnlZ.reset(new Vector(meanLogHyp.size() + covLogHyp.size() + likCovLogHyp.size()));

				// (1) w.r.t the mean parameters
				// nlZ = (1/2) * (y-m)' * inv(Kn) * (y-m)
				//       = - m' * inv(Kn) * y + (1/2) m' * inv(Kn) * m
				// nlZ_i = - m_i' * inv(Kn) * y + m_i' * inv(Kn) * m
				//          = - m_i' * inv(Kn) (y - m)
				//          = - m_i' * alpha
				for(int i = 0; i < meanLogHyp.size(); i++)
				{
					(*pDnlZ)(j++) = m_MeanFunc(meanLogHyp, i)->dot(*m_pAlpha);
					//std::cout << "DnlZ[ " << j-1 << " ] =  " << (*pDnlZ)(j-1) << std::endl;
				}

				// (2) w.r.t the cov parameters
				// nlZ = (1/2) * (y-m)' * inv(Kn) * (y-m) + (1/2) * log |Kn|
				// nlZ_j = (-1/2) * (y-m)' * inv(Kn) * K_j * inv(Kn) * (y-m)	+ (1/2) * tr[inv(Kn) * K_j]
				//          = (-1/2) * alpha' * K_j * alpha								+ (1/2) * tr[inv(Kn) * K_j]
				//          = (-1/2) * tr[(alpha' * alpha) * K_j]							+ (1/2) * tr[inv(Kn) * K_j]
				//          = (1/2) tr[(inv(Kn) - alpha*alpha') * K_j]
				//          = (1/2) tr[Q * K_j]
				//
				// Q = inv(Kn) - alpha*alpha'
				Matrix Q(n, n); // nxn
				//Q.noalias() = m_L.solve(Matrix::Identity(n, n)) - (*m_pAlpha) * (m_pAlpha->transpose());
				Q.noalias() = (*m_pInvKn) - (*m_pAlpha) * (m_pAlpha->transpose());
				for(int i = 0; i < covLogHyp.size(); i++)
				{
					//(*pDnlZ)(j++) = ((Scalar) 0.5f) * (Q * (m_CovFunc(covLogHyp, i)->selfadjointView<Eigen::Upper>())).trace(); // [CAUTION] K: upper triangular matrix
					(*pDnlZ)(j++) = ((Scalar) 0.5f) * Q.cwiseProduct(*(m_CovFunc(covLogHyp, i))).sum();
					//std::cout << "DnlZ[ " << j-1 << " ] =  " << (*pDnlZ)(j-1) << std::endl;
				}

				// (3) w.r.t the cov parameters
				// nlZ = (1/2) * (y-m)' * inv(K + D) * (y-m) + (1/2) * log |K + D|
				// nlZ_j = (-1/2) * (y-m)' * inv(Kn) * D_j * inv(Kn) * (y-m)	+ (1/2) * tr[inv(Kn) * D_j]
				//          = (-1/2) * alpha' * D_j * alpha								+ (1/2) * tr[inv(Kn) * D_j]
				//          = (-1/2) * tr[(alpha' * alpha) * D_j]							+ (1/2) * tr[inv(Kn) * D_j]
				//          = (1/2) tr[(inv(Kn) - alpha*alpha') * D_j]
				//          = (1/2) tr[Q * D_j]
				for(int i = 0; i < likCovLogHyp.size(); i++)
				{
					//(*pDnlZ)(j++) = (Scalar) 0.5f * (Q * (*m_LikFunc(likCovLogHyp, i))).trace();
					//(*pDnlZ)(j++) = ((Scalar) 0.5f) * Q.cwiseProduct(*(m_LikFunc(likCovLogHyp, i))).sum(); // [CAUTION] K: upper triangular matrix
					//(*pDnlZ)(j++) = ((Scalar) 0.5f) * (Q.array() * Matrix(m_LikFunc(likCovLogHyp, i)->asDiagonal()).array()).sum(); // [CAUTION] K: upper triangular matrix
					(*pDnlZ)(j++) = ((Scalar) 0.5f) * Q.cwiseProduct(Matrix(m_LikFunc(likCovLogHyp, i)->asDiagonal())).sum(); // [CAUTION] K: upper triangular matrix
					//std::cout << "DnlZ[ " << j-1 << " ] =  " << (*pDnlZ)(j-1) << std::endl;
				}
				std::cout << "dnlZ = " << std::endl << *pDnlZ << std::endl << std::endl;
			}
		}

	protected:
		bool calculateLandAlpha(const MeanHyp				&meanLogHyp,			// [input] hyperparameters
													const CovHyp				&covLogHyp, 
													const LikHyp					&likCovLogHyp)
		{
			std::cout << "calculateLandAlpha" << std::endl;

			// number of training data
			assert(m_MeanFunc.getN() == m_CovFunc.getN() && m_CovFunc.getN() == m_LikFunc.getN() && m_LikFunc.getN() == getN());
			const int n = getN();

			// memory allocation
			m_pY_M.reset(new Vector(n));					// nx1
			m_pAlpha.reset(new Vector(n));					// nx1

			// K 
			m_pKn = m_CovFunc(covLogHyp); // [CAUTION] K: upper triangular matrix
			if(m_pKn->hasNaN())	{ std::cout << "K has NaN." << std::endl; return false; }
			if(!m_pKn->allFinite())	{ std::cout << "K has Inf." << std::endl; return false; }
			std::cout << "K" << std::endl;
			//std::cout << "K = " << std::endl << *m_pKn << std::endl << std::endl;

			// D = sn2*I
			//MatrixPtr pD = m_LikFunc(likCovLogHyp);
			VectorPtr pD = m_LikFunc(likCovLogHyp);
			if(pD->hasNaN())	{ std::cout << "D has NaN." << std::endl; return false; }
			if(!pD->allFinite())	{ std::cout << "D has Inf." << std::endl; return false; }
			std::cout << "D" << std::endl;
			////std::cout << "D = " << std::endl << *pD << std::endl << std::endl;

			// Kn = K + D
			(*m_pKn) += pD->asDiagonal();
			if(m_pKn->hasNaN())	{ std::cout << "Kn has NaN." << std::endl; return false; }
			if(!m_pKn->allFinite())	{ std::cout << "Kn has Inf." << std::endl; return false; }
			std::cout << "Kn" << std::endl;
			//std::cout << "K + sn2*I = " << std::endl << *m_pKn << std::endl << std::endl;

			// LL' = K + D
			m_L.compute(*m_pKn);	// compute the Cholesky decomposition of Kn
			switch(m_L.info())
			{
			//case Eigen::ComputationInfo::Success:
			//	{
			//		std::cout << "Success" << std::endl;
			//		break;
			//	}
			case Eigen::ComputationInfo::NumericalIssue :
				{
					std::cout << "NumericalIssue " << std::endl;
					return false;
				}
			case Eigen::ComputationInfo::NoConvergence :
				{
					std::cout << "NoConvergence " << std::endl;
					return false;
				}
			case Eigen::ComputationInfo::InvalidInput :
				{
					std::cout << "InvalidInput " << std::endl;
					return false;
				}
			}
			std::cout << "L" << std::endl;

			// inv(Kn)
			m_pInvKn.reset(new Matrix(n, n));
			m_pInvKn->noalias() = m_L.solve(Matrix::Identity(n, n));
			if(m_pInvKn->hasNaN())	{ std::cout << "inv(Kn) has NaN." << std::endl; return false; }
			if(!m_pInvKn->allFinite())	{ std::cout << "inv(Kn) has Inf." << std::endl; return false; }
			std::cout << "inv(Kn)" << std::endl;

			// y - m
			(*m_pY_M).noalias() = (*m_pY) - (*(m_MeanFunc(meanLogHyp)));
			if(m_pY_M->hasNaN())	{ std::cout << "Y_M has NaN." << std::endl; return false; }
			if(!m_pY_M->allFinite())	{ std::cout << "Y_M has Inf." << std::endl; return false; }
			std::cout << "y - m" << std::endl;
			//std::cout << "y - m = " << std::endl << *m_pY_M << std::endl << std::endl;

			// alpha = inv(K + sn2*I)*(y-m)
			//(*m_pAlpha) = m_L.solve(*m_pY_M);
			m_pAlpha->noalias() = (*m_pInvKn) * (*m_pY_M);
			if(m_pAlpha->hasNaN())	{ std::cout << "Alpha has NaN." << std::endl; return false; }
			if(!m_pAlpha->allFinite())	{ std::cout << "Alpha has Inf." << std::endl; return false; }
			std::cout << "alpha" << std::endl;
			//std::cout << "alpha = " << std::endl << *m_pAlpha << std::endl << std::endl;

			return true;
		}

	protected:
		// GP setting
		MeanFunc				m_MeanFunc;
		CovFunc					m_CovFunc;
		LikFunc					m_LikFunc;

		// L, alpha
		CGCholeskyFactor				m_L;							// LL' = D^(-1/2) * K * D^(-1/2) + I
		MatrixPtr						m_pKn;				// Kn = K + D
		MatrixPtr						m_pInvKn;			// inv(Kn) = inv(K + D)
		VectorPtr						m_pY_M;					// y-m
		VectorPtr						m_pAlpha;					// alpha = inv(Kn) * (y-m)
	};

}

#endif 