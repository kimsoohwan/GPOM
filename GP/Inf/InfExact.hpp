#ifndef INF_EXACT_HPP
#define INF_EXACT_HPP

#include <Eigen/Cholesky> // for LLT

#include "GP/util/TrainingDataSetter.hpp"

namespace GPOM{

	template<class MeanFunc, class CovFunc, class LikFunc>
	class InfExact : public TrainingDataSetter
	{
	public:
		typedef	typename MeanFunc::Hyp						MeanHyp;
		typedef	typename CovFunc::Hyp							CovHyp;
		typedef	typename LikFunc::Hyp							LikHyp;

	public:
		// constructor
		InfExact() { }

		// destructor
		virtual ~InfExact() { }

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
							 bool										fVarianceVector = true)
		{
			// number of data
			assert(m_MeanFunc.getN() == m_CovFunc.getN() && m_CovFunc.getN() == m_LikFunc.getN() && m_LikFunc.getN() == getN());
			const int n = getN();
			const int m = PointMatrixDirection::fRowWisePointsMatrix ? pXs->rows() : pXs->cols();

			// calculate L and alpha
			// Kn = K + D
			// LL' = D^(-1/2) * K * D^(-1/2) + I
			// alpha = inv(K + sn2*I)*(y-m)
			calculateLandAlpha(meanLogHyp, covLogHyp, likCovLogHyp);

			// Ks, Kss
			MatrixConstPtr pKs		= m_CovFunc.Ks(pXs, covLogHyp); // nxm
			MatrixConstPtr pKss		= m_CovFunc.Kss(pXs, covLogHyp, fVarianceVector); // Vector (mx1) or Matrix (mxm)
			//std::cout << "Ks = " << std::endl << *pKs << std::endl << std::endl;
			//std::cout << "Kss = " << std::endl << *pKss << std::endl << std::endl;

			// predictive mean
			// mu = ms + Ks' * inv(Kn) * (y-m)
			//       = ms + Ks' * alpha
			pMu.reset(new Vector(m));
			pMu->noalias() = *(m_MeanFunc.Ms(pXs, meanLogHyp)) + (pKs->transpose()) * (*m_pAlpha);
			//std::cout << "Mu = " << std::endl << *pMu << std::endl << std::endl;

			// predictive variance
			// Sigma = Kss - Ks' * inv(Kn) * Ks
			//             = Kss - Ks' * inv(D^(1/2) * (D^(-1/2) * K * D^(-1/2) + I) * D^(1/2)) * Ks
			//             = Kss - Ks' * inv(D^(1/2) * LL' * D^(1/2)) * Ks
			//             = Kss - Ks' * D^(-1/2) * inv(L') * inv(L) * D^(-1/2) * Ks
			//             = Kss - (inv(L) * D^(-1/2) * Ks)' * (inv(L) * D^(-1/2) * Ks)
			//             = Kss - V' * V

			// V = inv(L) * D^(-1/2) * Ks
			//        (nxn)  *    (nxn)    * (nxm)
			Matrix V(n, m); // nxm
			V = m_pL->matrixL().solve(m_pInvSqrtD->asDiagonal() * (*pKs));
			//std::cout << "V = " << std::endl << V << std::endl << std::endl;

			if(fVarianceVector)
			{
				// sigma2 = kss - v' * v
				pSigma.reset(new Matrix(m, 1));					// variance vector (mx1)
				(*pSigma).noalias() = (*pKss) - V.transpose().array().square().matrix().rowwise().sum();
			}
			else
			{
				// Sigma = Kss - V' *V
				pSigma.reset(new Matrix(m, m));				// covariance matrix (mxm)
				(*pSigma).noalias() = (*pKss) - V.transpose() * V;
			}
			//std::cout << "Sigma = " << std::endl << *pSigma << std::endl << std::endl;
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

			// TODO: if non positivie definite, nlZ = Inf, dnlZ = zeros

			// number of training data
			assert(m_MeanFunc.getN() == m_CovFunc.getN() && m_CovFunc.getN() == m_LikFunc.getN() && m_LikFunc.getN() == getN());
			const int n = getN();

			// calculate L and alpha
			// Kn = K + D
			// LL' = D^(-1/2) * K * D^(-1/2) + I
			// alpha = inv(K + sn2*I)*(y-m)
			calculateLandAlpha(meanLogHyp, covLogHyp, likCovLogHyp);
			//std::cout << "L = " << std::endl << Matrix(m_pL->matrixL()) << std::endl << std::endl;
			//std::cout << "D^(-1/2) = " << std::endl << *m_pInvSqrtD << std::endl << std::endl;
			//std::cout << "y - m = " << std::endl << *m_pY_M << std::endl << std::endl;
			//std::cout << "alpha = " << std::endl << *m_pAlpha << std::endl << std::endl;

			// marginal likelihood
			// p(y) = N(m, Kn) = (2pi)^(-n/2) * |Kn|^(-1/2) * exp[(-1/2) * (y-m)' * inv(Kn) * (y-m)]
			// nlZ = (1/2) * (y-m)' * inv(Kn) * (y-m) + (1/2) * log |Kn|										+ (n/2) * log(2pi)
			//       = (1/2) * (y-m)' * alpha                + (1/2) * log |D^(1/2)*L*L'*D^(1/2)|		+ (n/2) * log(2pi)
			//       = (1/2) * (y-m)' * alpha                + (1/2) * log |D^(1/2)|*|L|*|L'|*|D^(1/2)|	+ (n/2) * log(2pi)
			//       = (1/2) * (y-m)' * alpha                + log |L|			+ log |D^(1/2)|						+ (n/2) * log(2pi)
			//       = (1/2) * (y-m)' * alpha                + log |L|			- log |D^(-1/2)|					+ (n/2) * log(2pi)
			//       = (1/2) * (y-m)' * alpha                + tr[log (L)]	- tr[log(D^(-1/2))]				+ (n/2) * log(2pi)
			if(calculationMode >= 0)
			{
				//std::cout << "nlZ" << std::endl;
				//std::cout << "meanLogHyp = " << std::endl << meanLogHyp << std::endl << std::endl;
				//std::cout << "covLogHyp = " << std::endl << covLogHyp << std::endl << std::endl;
				//std::cout << "likCovLogHyp = " << std::endl << likCovLogHyp << std::endl << std::endl;

				Matrix L(m_pL->matrixL());
				nlZ = ((Scalar) 0.5f) * (*m_pY_M).dot(*m_pAlpha)
					  + L.diagonal().array().log().sum()
					  - m_pInvSqrtD->array().log().sum()
					  + ((Scalar) n) * ((Scalar) 0.918938533204673f); // log(2pi)/2 = 0.918938533204673
				//nlZ = ((Scalar) 0.5f) * ((*m_pY_M).transpose() * (*m_pAlpha)).sum()
				//	  + log(det)
				//	  + ((Scalar) n) * ((Scalar) 0.918938533204673f); // log(2pi)/2 = 0.918938533204673
				//std::cout << "1 = " << std::endl << ((Scalar) 0.5f) * (*m_pY_M).dot(*m_pAlpha) << std::endl << std::endl;
				//std::cout << "2 = " << std::endl << L.diagonal().array().log().sum() << std::endl << std::endl;
				//std::cout << "3 = " << std::endl <<  - m_pInvSqrtD->array().log().sum() << std::endl << std::endl;
				//std::cout << "4 = " << std::endl << ((Scalar) n) * ((Scalar) 0.918938533204673f) << std::endl << std::endl;
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
				//
				// Kn * inv(Kn) = I
				// => D^(1/2) * LL' * D^(1/2) * inv(Kn) = I
				// => LL' * D^(1/2) * inv(Kn) = D^(-1/2)
				// => D^(1/2) * inv(Kn) = L.solve(D^(-1/2))
				// => inv(Kn) = D^(-1/2) * L.solve(D^(-1/2))

				Matrix Q(n, n); // nxn
				Q.noalias() = m_pInvSqrtD->asDiagonal() * (m_pL->solve(Matrix(m_pInvSqrtD->asDiagonal())));
				Q -= (*m_pAlpha) * (m_pAlpha->transpose());
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
			}
		}

	protected:
		void calculateLandAlpha(const MeanHyp				&meanLogHyp,			// [input] hyperparameters
													const CovHyp				&covLogHyp, 
													const LikHyp					&likCovLogHyp)
		{
			// number of training data
			assert(m_MeanFunc.getN() == m_CovFunc.getN() && m_CovFunc.getN() == m_LikFunc.getN() && m_LikFunc.getN() == getN());
			const int n = getN();

			// memory allocation
			m_pL.reset(new CholeskyFactor());			// nxn
			//m_pInvSqrtD.reset(new Vector(n));			// nx1
			m_pY_M.reset(new Vector(n));					// nx1
			m_pAlpha.reset(new Vector(n));					// nx1

			// K 
			MatrixPtr pKn = m_CovFunc(covLogHyp); // [CAUTION] K: upper triangular matrix
			//std::cout << "K = " << std::endl << *pKn << std::endl << std::endl;

			// D = sn2*I = D^(1/2) * D^(1/2)
			//MatrixPtr pD = m_LikFunc(likCovLogHyp);
			//VectorPtr pD = m_LikFunc(likCovLogHyp);
			m_pInvSqrtD = m_LikFunc(likCovLogHyp);	// sW
			//std::cout << "D = " << std::endl << *m_pInvSqrtD << std::endl << std::endl;
			(*m_pInvSqrtD) = m_pInvSqrtD->cwiseSqrt().cwiseInverse();
			//std::cout << "D^(-1/2) = " << std::endl << *m_pInvSqrtD << std::endl << std::endl;

			// Kn = K + D
			//(*pKn) += pD->asDiagonal();
			//(*pKn) += (*pD);

			// Kn = D^(-1/2) * K * D^(-1/2) + I
			//std::cout << "K = " << std::endl << *pKn << std::endl << std::endl;
			//std::cout << "K/sn2 = " << std::endl << m_pInvSqrtD->asDiagonal() * (*pKn) * m_pInvSqrtD->asDiagonal() << std::endl << std::endl;
			(*pKn) = m_pInvSqrtD->asDiagonal() * (*pKn) * m_pInvSqrtD->asDiagonal() + Matrix(n, n).setIdentity();
			//std::cout << "K/sn2 + I = " << std::endl << *pKn << std::endl << std::endl;

			// instead of						LL' = K + D
			// for numerical stability,	LL' = D^(-1/2) * K * D^(-1/2) + I 
			m_pL->compute(*pKn);	// compute the Cholesky decomposition of Kn
			//m_pL->compute(pKn->selfadjointView<Eigen::Upper>() );
			// [CAUTION]
			// Matrix L(n, n); L = m_pL->matrixL();
			// L*L.transpose() != m_pL->matrixLLT();
			//std::cout << "L = " << std::endl << Matrix(m_pL->matrixL()) << std::endl << std::endl;

			// y - m
			(*m_pY_M).noalias() = (*m_pY) - (*(m_MeanFunc(meanLogHyp)));
			//std::cout << "y - m = " << std::endl << *m_pY_M << std::endl << std::endl;

			// alpha = inv(K + sn2*I)*(y-m)
			// => (K + sn2*I) * alpha = y - m
			// =>  D^(1/2) * (D^(-1/2) * K * D^(-1/2) + I ) * D^(1/2) * alpha = y - m
			// => (D^(-1/2) * K * D^(-1/2) + I ) * D^(1/2) * alpha = D^(-1/2) * (y - m)
			// => D^(1/2) * alpha = L.solve(D^(-1/2) * (y - m))
			// => alpha = D^(-1/2) * L.solve(D^(-1/2) * (y - m))
			//(*m_pAlpha) = m_pL->solve(*m_pY_M);
			(*m_pAlpha).noalias() = m_pInvSqrtD->asDiagonal() * (m_pL->solve(m_pInvSqrtD->asDiagonal() * (*m_pY_M)));
			//std::cout << "D^(-1/2) * (y - m) = " << std::endl << m_pInvSqrtD->asDiagonal() * (*m_pY_M) << std::endl << std::endl;
			//std::cout << "L.solve(D^(-1/2) * (y - m)) = " << std::endl << m_pL->solve(m_pInvSqrtD->asDiagonal() * (*m_pY_M)) << std::endl << std::endl;
			//std::cout << "alpha = " << std::endl << *m_pAlpha << std::endl << std::endl;
		}

	protected:
		// GP setting
		MeanFunc				m_MeanFunc;
		CovFunc					m_CovFunc;
		LikFunc					m_LikFunc;

		// L, alpha
		CholeskyFactorPtr			m_pL;							// LL' = D^(-1/2) * K * D^(-1/2) + I
		VectorPtr							m_pInvSqrtD;				// D^(-1/2)
		VectorPtr							m_pY_M;					// y-m
		VectorPtr							m_pAlpha;					// alpha = inv(Kn) * (y-m)
	};

}

#endif 