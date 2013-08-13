#ifndef INF_EXACT_UNSTABLE_HPP
#define INF_EXACT_UNSTABLE_HPP

#include "GP/DataTypes.hpp"

namespace GPOM{

template<class MeanFunc, class CovFunc, class LikFunc>
class InfExactUnstable
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
	InfExactUnstable() { }

	// destructor
	virtual ~InfExactUnstable() { }

	// prediction
	void predict(MeanHypConstPtr				pMeanLogHyp, 
						 CovHypConstPtr					pCovLogHyp, 
						 LikHypConstPtr					pLikCovLogHyp, 
						 MatrixConstPtr						pX, 
						 VectorConstPtr						pY, 
						 MatrixConstPtr						pXs, 
						 VectorPtr								&pMu, 
						 MatrixPtr								&pSigma, 
						 bool										fVarianceVector = true)
	{
		// number of data
		const int n  = PointMatrixDirection::fRowWisePointsMatrix ? pX->rows()   : pX->cols();
		const int m = PointMatrixDirection::fRowWisePointsMatrix ? pXs->rows() : pXs->cols();

		// calculate L and alpha
		// Kn = K + D = LL' 
		// alpha = inv(K + sn2*I)*(y-m)
		CholeskyFactorPtr pL;	VectorPtr pY_M, pAlpha;
		calculateLandAlpha(pMeanLogHyp, pCovLogHyp, pLikCovLogHyp, 
										   pX, pY, 
										   pL, pY_M, pAlpha);

		// Ks, Kss
		MatrixConstPtr pKs		= m_CovFunc.Ks(pX, pXs, pCovLogHyp);
		MatrixConstPtr pKss		= m_CovFunc.Kss(pXs, pCovLogHyp, fVarianceVector); // Vector or Matrix

		// predictive mean
		// mu = ms + Ks' * inv(Kn) * (y-m)
		//       = ms + Ks' * alpha
		pMu.reset(new Vector(m));
		pMu->noalias() = *(m_MeanFunc(pXs, pMeanLogHyp)) + (pKs->transpose()) * (*pAlpha);

		// predictive variance
		// Sigma = Kss - Ks' * inv(Kn) * Ks
		//             = Kss - Ks' * inv(LL') * Ks
		//             = Kss - Ks' * inv(L') * inv(L) * Ks
		//             = Kss - (inv(L)*Ks)' * (inv(L)*Ks)
		//             = Kss - V' * V

		// V = inv(L) * Ks 
		//        (nxn) * (nxm)
		Matrix V(n, m); // nxm
		V = pL->matrixL().solve(*pKs);

		if(fVarianceVector)
		{
			// sigma2 = kss - v' * v
			pSigma.reset(new Matrix(m, 1));					// variance vector
			(*pSigma).noalias() = (*pKss) - V.transpose().array().square().matrix().rowwise().sum();
		}
		else
		{
			// Sigma = Kss - V' *V
			pSigma.reset(new Matrix(m, m)); // covariance matrix
			(*pSigma).noalias() = (*pKss) - V.transpose() * V;
		}
	}

	void negativeLogMarginalLikelihood(MeanHypConstPtr				pMeanLogHyp, 
																	CovHypConstPtr				pCovLogHyp, 
																	LikHypConstPtr					pLikCovLogHyp, 
																	MatrixConstPtr					pX, 
																	VectorConstPtr					pY,
																	Scalar									&nlZ, 
																	VectorPtr							&pDnlZ,
																	const int								calculationMode = 0)
	{
		// calculationMode
		// [0]: calculate both nlZ and pDnlZ
		// [+]: calculate nlZ only
		// [-]: calculate pDnlZ only

		// number of training data
		const int n  = PointMatrixDirection::fRowWisePointsMatrix ? pX->rows()   : pX->cols();

		// partial derivatives w.r.t hyperparameters
		pDnlZ.reset(new Vector(pMeanLogHyp->size() + pCovLogHyp->size() + pLikCovLogHyp->size()));

		// calculate L and alpha
		// Kn = K + D = LL' 
		// alpha = inv(K + sn2*I)*(y-m)
		CholeskyFactorPtr pL;	VectorPtr pY_M, pAlpha;
		calculateLandAlpha(pMeanLogHyp, pCovLogHyp, pLikCovLogHyp, 
										   pX, pY, 
										   pL, pY_M, pAlpha); // TODO: save for speed-ups

		// marginal likelihood
		// p(y) = N(m, Kn) = (2pi)^(-n/2) * |Kn|^(-1/2) * exp[(-1/2) * (y-m)' * inv(Kn) * (y-m)]
		// nlZ = (1/2) * (y-m)' * inv(Kn) * (y-m) + (1/2) * log |Kn|		+ (n/2) * log(2pi)
		//       = (1/2) * (y-m)' * alpha                + (1/2) * log |L*L'|		+ (n/2) * log(2pi)
		//       = (1/2) * (y-m)' * alpha                + (1/2) * log |L|*|L'|	+ (n/2) * log(2pi)
		//       = (1/2) * (y-m)' * alpha                + log |L||					+ (n/2) * log(2pi)
		//       = (1/2) * (y-m)' * alpha                + tr[log (L)]				+ (n/2) * log(2pi)
		if(calculationMode >= 0)
		{
			Matrix L(pL->matrixL());
			nlZ = ((Scalar) 0.5f) * (*pY_M).dot(*pAlpha)
				  + L.diagonal().array().log().sum()
				  + ((Scalar) n) * ((Scalar) 0.918938533204673f); // log(2pi)/2 = 0.918938533204673
			//nlZ = ((Scalar) 0.5f) * ((*pY_M).transpose() * (*pAlpha)).sum()
			//	  + log(det)
			//	  + ((Scalar) n) * ((Scalar) 0.918938533204673f); // log(2pi)/2 = 0.918938533204673
			//std::cout << "1 = " << std::endl << ((Scalar) 0.5f) * (*pY_M).dot(*pAlpha) << std::endl << std::endl;
			//std::cout << "2 = " << std::endl << L.diagonal().array().log().sum() << std::endl << std::endl;
			//std::cout << "3 = " << std::endl << ((Scalar) n) * ((Scalar) 0.918938533204673f) << std::endl << std::endl;
		}

		if(calculationMode <= 0)
		{
			// derivatives (f_j = partial f / partial x_j)
			int j = 0; // partial derivative index

			// (1) w.r.t the mean parameters
			// nlZ = (1/2) * (y-m)' * inv(Kn) * (y-m)
			//       = - m' * inv(Kn) * y + (1/2) m' * inv(Kn) * m
			// nlZ_i = - m_i' * inv(Kn) * y + m_i' * inv(Kn) * m
			//          = - m_i' * inv(Kn) (y - m)
			//          = - m_i' * alpha
			for(int i = 0; i < pMeanLogHyp->size(); i++)
			{
				(*pDnlZ)(j++) = m_MeanFunc(pX, pMeanLogHyp, i)->dot(*pAlpha);
			}

			// (2) w.r.t the cov parameters
			// nlZ = (1/2) * (y-m)' * inv(Kn) * (y-m) + (1/2) * log |Kn|
			// nlZ_j = (-1/2) * (y-m)' * inv(Kn) * K_j * inv(Kn) * (y-m)	+ (1/2) * tr[inv(Kn) * K_j]
			//          = (-1/2) * alpha' * K_j * alpha								+ (1/2) * tr[inv(Kn) * K_j]
			//          = (-1/2) * tr[(alpha' * alpha) * K_j]							+ (1/2) * tr[inv(Kn) * K_j]
			//          = (1/2) tr[(inv(Kn) - alpha*alpha') * K_j]
			//          = (1/2) tr[Q * K_j]
			Matrix Q(n, n); // nxn
			Q = pL->solve(Matrix(n, n).setIdentity());
			Q -= (*pAlpha) * (pAlpha->transpose());
			for(int i = 0; i < pCovLogHyp->size(); i++)
			{
				//(*pDnlZ)(j++) = ((Scalar) 0.5f) * (Q * (m_CovFunc(pX, pCovLogHyp, i)->selfadjointView<Eigen::Upper>())).trace(); // [CAUTION] K: upper triangular matrix
				(*pDnlZ)(j++) = ((Scalar) 0.5f) * Q.cwiseProduct(*(m_CovFunc(pX, pCovLogHyp, i))).sum(); // [CAUTION] K: upper triangular matrix
			}

			// (3) w.r.t the cov parameters
			// nlZ = (1/2) * (y-m)' * inv(K + D) * (y-m) + (1/2) * log |K + D|
			// nlZ_j = (-1/2) * (y-m)' * inv(Kn) * D_j * inv(Kn) * (y-m)	+ (1/2) * tr[inv(Kn) * D_j]
			//          = (-1/2) * alpha' * D_j * alpha								+ (1/2) * tr[inv(Kn) * D_j]
			//          = (-1/2) * tr[(alpha' * alpha) * D_j]							+ (1/2) * tr[inv(Kn) * D_j]
			//          = (1/2) tr[(inv(Kn) - alpha*alpha') * D_j]
			//          = (1/2) tr[Q * D_j]
			for(int i = 0; i < pLikCovLogHyp->size(); i++)
			{
				//(*pDnlZ)(j++) = (Scalar) 0.5f * (Q * (*m_LikFunc(pX, pLikCovLogHyp, i))).trace();
				//(*pDnlZ)(j++) = ((Scalar) 0.5f) * Q.cwiseProduct(*(m_LikFunc(pX, pLikCovLogHyp, i))).sum(); // [CAUTION] K: upper triangular matrix
				(*pDnlZ)(j++) = ((Scalar) 0.5f) * Q.cwiseProduct(Matrix(m_LikFunc(pX, pLikCovLogHyp, i)->asDiagonal())).sum(); // [CAUTION] K: upper triangular matrix
			}
		}
	}

protected:
	void calculateLandAlpha(MeanHypConstPtr				pMeanLogHyp,			// [input] hyperparameters
												CovHypConstPtr				pCovLogHyp, 
												LikHypConstPtr					pLikCovLogHyp, 
												MatrixConstPtr					pX,								// [input] training data
												VectorConstPtr					pY,
												CholeskyFactorPtr			&pL,							// [output] Kn = LL'
												VectorPtr							&pY_M,						// [output] y-m
												VectorPtr							&pAlpha)					// [output] alpha = inv(Kn) * (y-m)
	{
		// number of training data
		const int n  = PointMatrixDirection::fRowWisePointsMatrix ? pX->rows()   : pX->cols();

		// memory allocation
		pL.reset(new CholeskyFactor());			// nxn
		pY_M.reset(new Vector(n));					// nx1
		pAlpha.reset(new Vector(n));				// nx1

		// K 
		MatrixPtr pKn = m_CovFunc(pX, pCovLogHyp); // [CAUTION] K: upper triangular matrix

		// D = sn2*I
		//MatrixPtr pD = m_LikFunc(pX, pLikCovLogHyp);
		VectorPtr pD = m_LikFunc(pX, pLikCovLogHyp);

		// Kn = K + D = LL' 
		//(*pKn) += (*pD);
		(*pKn) += pD->asDiagonal();
		pL->compute(*(pKn));	// compute the Cholesky decomposition of Kn
		//pL->compute(pKn->selfadjointView<Eigen::Upper>() );
		// [CAUTION]
		// Matrix L(n, n); L = pL->matrixL();
		// L*L.transpose() != pL->matrixLLT();

		// y - m
		(*pY_M).noalias() = (*pY) - (*(m_MeanFunc(pX, pMeanLogHyp)));

		// alpha = inv(K + sn2*I)*(y-m)
		(*pAlpha) = pL->solve(*pY_M);
	}

protected:
	// GP setting
	MeanFunc		m_MeanFunc;
	CovFunc			m_CovFunc;
	LikFunc			m_LikFunc;
};
}

#endif 