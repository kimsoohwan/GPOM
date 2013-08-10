#ifndef INF_EXACT_HPP
#define INF_EXACT_HPP

#include <Eigen/Cholesky> // for LLT
#include "GP/DataTypes.hpp"

namespace GP{

template<class MeanFunc, class CovFunc, class LikFunc>
class InfExact
{
public:
	typedef	typename MeanFunc::Hyp							MeanHyp;
	typedef	typename MeanFunc::HypPtr					MeanHypPtr;
	typedef	typename MeanFunc::HypConstPtr			MeanHypConstPtr;

	typedef	typename CovFunc::Hyp							CovHyp;
	typedef	typename CovFunc::HypPtr						CovHypPtr;
	typedef	typename CovFunc::HypConstPtr			CovHypConstPtr;

	typedef	typename LikFunc::Hyp								LikHyp;
	typedef	typename LikFunc::HypPtr						LikHypPtr;
	typedef	typename LikFunc::HypConstPtr				LikHypConstPtr;

public:
	// constructor
	InfExact() { }

	// destructor
	virtual ~InfExact() { }

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
		const int n = pX->rows();
		const int m = pXs->rows();

		// calculate L and alpha
		// Kn = K + D
		// LL' = D^(-1/2) * K * D^(-1/2) + I
		// alpha = inv(K + sn2*I)*(y-m)
		CholeskyFactorPtr pL;	
		VectorPtr pInvSqrtD, pY_M, pAlpha;
		calculateLandAlpha(pMeanLogHyp, pCovLogHyp, pLikCovLogHyp, 
										   pX, pY, 
										   pL, pInvSqrtD, pY_M, pAlpha);

		// Ks, Kss
		MatrixConstPtr pKs		= m_CovFunc.Ks(pX, pXs, pCovLogHyp); // nxm
		MatrixConstPtr pKss		= m_CovFunc.Kss(pXs, pCovLogHyp, fVarianceVector); // Vector (mx1) or Matrix (mxm)

		// predictive mean
		// mu = ms + Ks' * inv(Kn) * (y-m)
		//       = ms + Ks' * alpha
		pMu.reset(new Vector(m));
		pMu->noalias() = *(m_MeanFunc(pXs, pMeanLogHyp)) + (pKs->transpose()) * (*pAlpha);

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
		V = pL->matrixL().solve(pInvSqrtD->asDiagonal() * (*pKs));

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

		// partial derivatives w.r.t hyperparameters
		pDnlZ.reset(new Vector(pMeanLogHyp->size() + pCovLogHyp->size() + pLikCovLogHyp->size()));

		// number of training data
		int n = pX->rows();

		// calculate L and alpha
		// Kn = K + D
		// LL' = D^(-1/2) * K * D^(-1/2) + I
		// alpha = inv(K + sn2*I)*(y-m)
		CholeskyFactorPtr pL;	
		VectorPtr pInvSqrtD, pY_M, pAlpha;
		calculateLandAlpha(pMeanLogHyp, pCovLogHyp, pLikCovLogHyp, 
										   pX, pY, 
										   pL, pInvSqrtD, pY_M, pAlpha);

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
			Matrix L(pL->matrixL());
			nlZ = ((Scalar) 0.5f) * (*pY_M).dot(*pAlpha)
				  + L.diagonal().array().log().sum()
				  - pInvSqrtD->array().log().sum()
				  + ((Scalar) n) * ((Scalar) 0.918938533204673f); // log(2pi)/2 = 0.918938533204673
			//nlZ = ((Scalar) 0.5f) * ((*pY_M).transpose() * (*pAlpha)).sum()
			//	  + log(det)
			//	  + ((Scalar) n) * ((Scalar) 0.918938533204673f); // log(2pi)/2 = 0.918938533204673
			//std::cout << "1 = " << std::endl << ((Scalar) 0.5f) * (*pY_M).dot(*pAlpha) << std::endl << std::endl;
			//std::cout << "2 = " << std::endl << L.diagonal().array().log().sum() << std::endl << std::endl;
			//std::cout << "3 = " << std::endl <<  - pInvSqrtD->array().log().sum() << std::endl << std::endl;
			//std::cout << "4 = " << std::endl << ((Scalar) n) * ((Scalar) 0.918938533204673f) << std::endl << std::endl;
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
			//
			// Q = inv(Kn) - alpha*alpha'
			//
			// Kn * inv(Kn) = I
			// => D^(1/2) * LL' * D^(1/2) * inv(Kn) = I
			// => LL' * D^(1/2) * inv(Kn) = D^(-1/2)
			// => D^(1/2) * inv(Kn) = L.solve(D^(-1/2))
			// => inv(Kn) = D^(-1/2) * L.solve(D^(-1/2))

			Matrix Q(n, n); // nxn
			Q.noalias() = pInvSqrtD->asDiagonal() * (pL->solve(Matrix(pInvSqrtD->asDiagonal())));
			Q -= (*pAlpha) * (pAlpha->transpose());
			for(int i = 0; i < pCovLogHyp->size(); i++)
			{
				//(*pDnlZ)(j++) = ((Scalar) 0.5f) * (Q * (m_CovFunc(pX, pCovLogHyp, i)->selfadjointView<Eigen::Upper>())).trace(); // [CAUTION] K: upper triangular matrix
				(*pDnlZ)(j++) = ((Scalar) 0.5f) * Q.cwiseProduct(*(m_CovFunc(pX, pCovLogHyp, i))).sum();
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
				//(*pDnlZ)(j++) = ((Scalar) 0.5f) * (Q.array() * Matrix(m_LikFunc(pX, pLikCovLogHyp, i)->asDiagonal()).array()).sum(); // [CAUTION] K: upper triangular matrix
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
												CholeskyFactorPtr			&pL,							// [output] LL' = D^(-1/2) * K * D^(-1/2) + I
												VectorPtr							&pInvSqrtD,				// [output] D^(-1/2)
												VectorPtr							&pY_M,						// [output] y-m
												VectorPtr							&pAlpha)					// [output] alpha = inv(Kn) * (y-m)
	{
		// number of training data
		const int n = pX->rows();

		// memory allocation
		pL.reset(new CholeskyFactor());			// nxn
		//pInvSqrtD.reset(new Vector(n));		// nx1
		pY_M.reset(new Vector(n));					// nx1
		pAlpha.reset(new Vector(n));				// nx1

		// K 
		MatrixPtr pKn = m_CovFunc(pX, pCovLogHyp); // [CAUTION] K: upper triangular matrix

		// D = sn2*I = D^(1/2) * D^(1/2)
		//MatrixPtr pD = m_LikFunc(pX, pLikCovLogHyp);
		//VectorPtr pD = m_LikFunc(pX, pLikCovLogHyp);
		pInvSqrtD = m_LikFunc(pX, pLikCovLogHyp);
		(*pInvSqrtD) = pInvSqrtD->cwiseSqrt().cwiseInverse();
		//std::cout << "D^(-1/2) = " << std::endl << *pInvSqrtD << std::endl << std::endl;

		// Kn = K + D
		//(*pKn) += pD->asDiagonal();
		//(*pKn) += (*pD);

		// Kn = D^(-1/2) * K * D^(-1/2) + I
		//std::cout << "K = " << std::endl << *pKn << std::endl << std::endl;
		//std::cout << "K/sn2 = " << std::endl << pInvSqrtD->asDiagonal() * (*pKn) * pInvSqrtD->asDiagonal() << std::endl << std::endl;
		(*pKn) = pInvSqrtD->asDiagonal() * (*pKn) * pInvSqrtD->asDiagonal() + Matrix(n, n).setIdentity();
		//std::cout << "K/sn2 + I = " << std::endl << *pKn << std::endl << std::endl;

		// instead of						LL' = K + D
		// for numerical stability,	LL' = D^(-1/2) * K * D^(-1/2) + I 
		pL->compute(*pKn);	// compute the Cholesky decomposition of Kn
		//pL->compute(pKn->selfadjointView<Eigen::Upper>() );
		// [CAUTION]
		// Matrix L(n, n); L = pL->matrixL();
		// L*L.transpose() != pL->matrixLLT();
		//std::cout << "L = " << std::endl << Matrix(pL->matrixL()) << std::endl << std::endl;

		// y - m
		(*pY_M).noalias() = (*pY) - (*(m_MeanFunc(pX, pMeanLogHyp)));

		// alpha = inv(K + sn2*I)*(y-m)
		// => (K + sn2*I) * alpha = y - m
		// =>  D^(1/2) * (D^(-1/2) * K * D^(-1/2) + I ) * D^(1/2) * alpha = y - m
		// => (D^(-1/2) * K * D^(-1/2) + I ) * D^(1/2) * alpha = D^(-1/2) * (y - m)
		// => D^(1/2) * alpha = L.solve(D^(-1/2) * (y - m))
		// => alpha = D^(-1/2) * L.solve(D^(-1/2) * (y - m))
		//(*pAlpha) = pL->solve(*pY_M);
		(*pAlpha).noalias() = pInvSqrtD->asDiagonal() * (pL->solve(pInvSqrtD->asDiagonal() * (*pY_M)));
		//std::cout << "alpha = " << std::endl << *pAlpha << std::endl << std::endl;
	}

protected:
	// GP setting
	MeanFunc		m_MeanFunc;
	CovFunc			m_CovFunc;
	LikFunc			m_LikFunc;
};
}

#endif 