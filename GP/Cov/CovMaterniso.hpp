#ifndef COVARIANCE_FUNCTION_MATERN_ISO_HPP
#define COVARIANCE_FUNCTION_MATERN_ISO_HPP

#include "GP/Cov/CovMaternisoBase.hpp"

namespace GPOM{

class CovMaterniso3 : public CovMaternisoBase
{
	public:
		// hyperparameters
		typedef	Eigen::Matrix<Scalar, 2, 1>							Hyp;						// ell, sigma_f
		typedef	boost::shared_ptr<Hyp>								HypPtr;
		typedef	boost::shared_ptr<const Hyp>					HypConstPtr;

	public:
		// constructor
		CovMaterniso3() { }

		// destructor
		virtual ~CovMaterniso3() { }

		// operator
		MatrixPtr operator()(MatrixConstPtr pX, HypConstPtr pLogHyp, const int pdIndex = -1)
		{
			return K(pX, pLogHyp, pdIndex);
		}

		// self covariance
		MatrixPtr K(MatrixConstPtr pX, HypConstPtr pLogHyp, const int pdIndex = -1)
		{
			// input
			// pX (nxd): training inputs
			// pLogHyp: log hyperparameters
			// pdIndex: partial derivatives with respect to this parameter index

			// output
			// K: nxn matrix

			// pre-calculate the distances
			preCalculateDist(pX);

			// calculate the covariance matrix
			return K_FF(m_pDist, pLogHyp, pdIndex);
		}

		// cross covariance
		MatrixPtr Ks(MatrixConstPtr pX, MatrixConstPtr pXs, HypConstPtr pLogHyp) const
		{
			// input
			// pX (nxd): training inputs
			// pXx (mxd): test inputs
			// pLogHyp: log hyperparameters

			// output
			// K: nxm matrix

			// calculate the distances
			MatrixPtr pDist = crossSqDistances(pX, pXs);		// squared distances
			pDist->noalias() = pDist->cwiseSqrt();					// distances

			// calculate the covariance matrix
			return K_FF(pDist, pLogHyp);
		}

		// self-variance/covariance
		MatrixPtr Kss(MatrixConstPtr pXs, HypConstPtr pLogHyp, const bool fVarianceVector = true)
		{
			// input
			// pXs (mxd): test inputs
			// logHyp: log hyperparameters
			// fVarianceVector: [true] self-variance vector (mx1), [false] self-covariance matrix (mxm)

			// number of test data
			const int m = PointMatrixDirection::fRowWisePointsMatrix ? pXs->rows() : pXs->cols();

			// hyperparameters
			Scalar sigma_f2 = exp(((Scalar) 2.f) * (*pLogHyp)(1));

			// output
			MatrixPtr pK;

			// k(X, X') = sigma_f^2
			if(fVarianceVector)
			{
				// K: self-variance vector (mx1)
				pK.reset(new Matrix(m, 1));
				pK->fill(sigma_f2);
			}
			else					
			{
				// K: self-covariance matrix (mxm)

				// calculate the squared distances
				MatrixPtr pDist = selfSqDistances(pXs);		// squared distances
				pDist->noalias() = pDist->cwiseSqrt();			// distances

				// calculate the covariance matrix
				pK = K_FF(pDist, pLogHyp);
			}

			return pK;
		}

	protected:
		// covariance matrix given pair-wise sqaured distances
		MatrixPtr K_FF(MatrixConstPtr pDist, HypConstPtr pLogHyp, const int pdIndex = -1) const
		{
			// input
			// pSqDist (nxm): squared distances
			// pLogHyp: log hyperparameters
			// pdIndex: partial derivatives with respect to this parameter index

			// output
			// K: nxm matrix
			MatrixPtr pK(new Matrix(pDist->rows(), pDist->cols()));

			// hyperparameters
			Scalar inv_ell							= exp(((Scalar)  -1.f) * (*pLogHyp)(0));
			Scalar sigma_f2					= exp(((Scalar)  2.f) * (*pLogHyp)(1));
			Scalar neg_sqrt3_inv_ell		= - SQRT3 * inv_ell;
			Scalar twice_sigma_f2			= ((Scalar) 2.f) * sigma_f2;

			// mode
			switch(pdIndex)
			{
			// covariance matrix
			case -1:
				{
					// k(X, X') = sigma_f^2 *(1 + sqrt(3)*r / ell) * exp(- sqrt(3)*r / ell)
					(*pK).noalias() = neg_sqrt3_inv_ell * (*pDist);
					(*pK) = sigma_f2 * (((Scalar) 1.f) - pK->array()) * pK->array().exp();
					break;
				}

			// derivatives of covariance matrix w.r.t log ell
			case 0:
				{
					// k_log(l)		= sigma_f^2 * (3 * r^2 / ell^2) * exp(- sqrt(3)*r / ell)
					(*pK).noalias() = neg_sqrt3_inv_ell * (*pDist);
					(*pK) = sigma_f2 * pK->array().square() * pK->array().exp();
					break;
				}

			// derivatives of covariance matrix w.r.t log sigma_f
			case 1:
				{
					// k_log(sigma_f)	= 2 * sigma_f *(1 + sqrt(3)*r / ell) * exp(- sqrt(3)*r / ell)
					(*pK).noalias() = neg_sqrt3_inv_ell * (*pDist);
					(*pK) = twice_sigma_f2 * (((Scalar) 1.f) - pK->array()) * pK->array().exp();
					break;
				}
			}

			return pK;
		}
};
}

#endif