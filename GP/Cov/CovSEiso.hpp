#ifndef COVARIANCE_FUNCTION_SQUARED_EXPONENTIAL_ISO_HPP
#define COVARIANCE_FUNCTION_SQUARED_EXPONENTIAL_ISO_HPP

#include "GP/DataTypes.hpp"

namespace GPOM{

class CovSEiso
{
	public:
		// hyperparameters
		typedef	Eigen::Matrix<Scalar, 2, 1>							Hyp;						// ell, sigma_f
		typedef	boost::shared_ptr<Hyp>								HypPtr;
		typedef	boost::shared_ptr<const Hyp>					HypConstPtr;

	public:
		// constructor
		CovSEiso() { }

		// destructor
		virtual ~CovSEiso() { }

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

			// pre_calculate the squared distances
			preCalculateSqDist(pX);

			// calculate the covariance matrix
			return K_FF(m_pSqDist, pLogHyp, pdIndex);
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

			// calculate the squared distances
			MatrixPtr pSqDist = crossSqDistances(pX, pXs);

			// calculate the covariance matrix
			return K_FF(pSqDist, pLogHyp);
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
			MatrixPtr pKss;

			// k(X, X') = sigma_f^2
			if(fVarianceVector)
			{
				// K: self-variance vector (mx1)
				pKss.reset(new Matrix(m, 1));
				pKss->fill(sigma_f2);
			}
			else					
			{
				// K: self-covariance matrix (mxm)

				// calculate the squared distances
				MatrixPtr pSqDist = selfSqDistances(pXs);

				// calculate the covariance matrix
				pKss = K_FF(pSqDist, pLogHyp);
			}

			return pKss;
		}

	protected:
		// covariance matrix given pair-wise sqaured distances
		MatrixPtr K_FF(MatrixConstPtr pSqDist, HypConstPtr pLogHyp, const int pdIndex = -1) const
		{
			// input
			// pSqDist (nxm): squared distances
			// pLogHyp: log hyperparameters
			// pdIndex: partial derivatives with respect to this parameter index

			// output
			// K: nxm matrix
			MatrixPtr pK(new Matrix(pSqDist->rows(), pSqDist->cols()));

			// hyperparameters
			Scalar inv_ell2						= exp(((Scalar) -2.f) * (*pLogHyp)(0));
			Scalar sigma_f2					= exp(((Scalar)  2.f) * (*pLogHyp)(1));
			Scalar neg_half_inv_ell2		= ((Scalar) -0.5f) * inv_ell2;
			Scalar sigma_f2_inv_ell2		= sigma_f2 * inv_ell2;
			Scalar twice_sigma_f2			= ((Scalar) 2.f) * sigma_f2;

			// mode
			switch(pdIndex)
			{
			// covariance matrix
			case -1:
				{
					// k(X, X') = sigma_f^2 *exp(- r^2 / (2*ell^2))
					(*pK).noalias() = sigma_f2 * (neg_half_inv_ell2 * (*pSqDist)).array().exp().matrix(); // TODO: (*pK).noalias() = ???
					//(*pK).triangularView<Eigen::Upper>() = sigma_f2_inv_ell2 * ((neg_half_inv_ell2 * (*pSqDist)).array().exp()  * (*pSqDist).array()).matrix();
					//std::cout << "K = " << std::endl << *pK << std::endl << std::endl;
					break;
				}

			// derivatives of covariance matrix w.r.t log ell
			case 0:
				{
					// k(X, X')		= sigma_f^2 * exp(- r^2/(2*ell^2))
					// k_log(l)		= sigma_f^2 * exp(- r^2/(2*ell^2)) * (r^2 / ell^3) * ell
					//					= sigma_f^2 exp(- r^2/(2*ell^2)) * (r^2 / ell^2)
					(*pK).noalias() = sigma_f2_inv_ell2 * ((neg_half_inv_ell2 * (*pSqDist)).array().exp()  * (*pSqDist).array()).matrix();
					// (*pK).triangularView<Eigen::Upper>()	= sigma_f2_inv_ell2 * ((neg_half_inv_ell2 * (*pSqDist)).array().exp()  * (*pSqDist).array()).matrix();
					//std::cout << "K_log(ell) = " << std::endl << *pK << std::endl << std::endl;
					break;
				}

			// derivatives of covariance matrix w.r.t log sigma_f
			case 1:
				{
					// k(X, X')					= sigma_f^2 * exp(- r^2 / (2*ell^2))
					// k_log(sigma_f)	= 2 * sigma_f * exp(- r^2 / (2*ell^2)) * sigma_f
					//								= 2 * sigma_f^2 * exp(- r^2 / (2*ell^2))
					(*pK).noalias() = twice_sigma_f2 * ((neg_half_inv_ell2 * (*pSqDist)).array().exp()).matrix();
					// (*pK).triangularView<Eigen::Upper>()	= twice_sigma_f2 * (neg_half_inv_ell2 * (*pSqDist)).array().exp().matrix();
					//std::cout << "K_log(sigma_f) = " << std::endl << *pK << std::endl << std::endl;
					break;
				}
			}

			return pK;
		}

		// pre-calculate the squared distances
		bool preCalculateSqDist(MatrixConstPtr pX)
		{
			// check if the training inputs are the same
			if(m_pTrainingInputs == pX) return false;
			m_pTrainingInputs = pX;

			// pre-calculate the squared distances
			m_pSqDist = selfSqDistances(pX);
			return true;
		}

	protected:
		MatrixConstPtr						m_pTrainingInputs;			// training inputs
		MatrixPtr									m_pSqDist;						// squared distances of the training inputs
};

}

#endif