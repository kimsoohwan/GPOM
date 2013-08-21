#ifndef COVARIANCE_FUNCTION_SQUARED_EXPONENTIAL_ISO_HPP
#define COVARIANCE_FUNCTION_SQUARED_EXPONENTIAL_ISO_HPP

#include "GP/util/TrainingInputSetter.hpp"
#include "GP/util/sqDistances.hpp"

namespace GPOM{

	class CovSEiso : public TrainingInputSetter
	{
		public:
			// hyperparameters
			typedef	Eigen::Matrix<Scalar, 2, 1>							Hyp;						// ell, sigma_f

		public:
			// constructor
			CovSEiso() { }

			// destructor
			virtual ~CovSEiso() { }

			// setter
			bool setTrainingInputs(MatrixConstPtr pX)
			{
				// check if the training inputs are the same
				if(!TrainingInputSetter::setTrainingInputs(pX))		return false;

				// pre_calculate the squared distances
				preCalculateSqDist(pX);

				return true;
			}

			// operator
			MatrixPtr operator()(const Hyp &logHyp, const int pdIndex = -1)
			{
				return K(logHyp, pdIndex);
			}

			// self covariance
			MatrixPtr K(const Hyp &logHyp, const int pdIndex = -1)
			{
				// input
				// pX (nxd): training inputs
				// logHyp: log hyperparameters
				// pdIndex: partial derivatives with respect to this parameter index

				// output
				// K: nxn matrix

				// calculate the covariance matrix
				return K_FF(m_pSqDist, logHyp, pdIndex);
			}

			// cross covariance
			MatrixPtr Ks(MatrixConstPtr pXs, const Hyp &logHyp) const
			{
				// input
				// pX (nxd): training inputs
				// pXx (mxd): test inputs
				// logHyp: log hyperparameters

				// output
				// K: nxm matrix

				// calculate the squared distances
				MatrixPtr pSqDist = crossSqDistances(m_pX, pXs);

				// calculate the covariance matrix
				return K_FF(pSqDist, logHyp);
			}

			// self-variance/covariance
			MatrixPtr Kss(MatrixConstPtr pXs, const Hyp &logHyp, const bool fVarianceVector = true) const
			{
				// input
				// pXs (mxd): test inputs
				// logHyp: log hyperparameters
				// fVarianceVector: [true] self-variance vector (mx1), [false] self-covariance matrix (mxm)

				// number of test data
				const int m = PointMatrixDirection::fRowWisePointsMatrix ? pXs->rows() : pXs->cols();

				// hyperparameters
				Scalar sigma_f2 = exp(((Scalar) 2.f) * logHyp(1));

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
					pKss = K_FF(pSqDist, logHyp);
				}

				return pKss;
			}

		protected:
			// covariance matrix given pair-wise sqaured distances
			MatrixPtr K_FF(MatrixConstPtr pSqDist, const Hyp &logHyp, const int pdIndex = -1) const
			{
				// input
				// pSqDist (nxm): squared distances
				// logHyp: log hyperparameters
				// pdIndex: partial derivatives with respect to this parameter index

				assert(pdIndex < 2);

				// output
				// K: nxm matrix
				MatrixPtr pK(new Matrix(pSqDist->rows(), pSqDist->cols()));

				// hyperparameters
				Scalar inv_ell2						= exp(((Scalar) -2.f) * logHyp(0));
				Scalar sigma_f2					= exp(((Scalar)  2.f) * logHyp(1));
				Scalar neg_half_inv_ell2		= ((Scalar) -0.5f) * inv_ell2;
				Scalar sigma_f2_inv_ell2		= sigma_f2 * inv_ell2;
				Scalar twice_sigma_f2			= ((Scalar) 2.f) * sigma_f2;

				// mode
				switch(pdIndex)
				{
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

				// covariance matrix
				default:
					{
						// k(X, X') = sigma_f^2 *exp(- r^2 / (2*ell^2))
						(*pK).noalias() = sigma_f2 * (neg_half_inv_ell2 * (*pSqDist)).array().exp().matrix(); // TODO: (*pK).noalias() = ???
						//(*pK).triangularView<Eigen::Upper>() = sigma_f2_inv_ell2 * ((neg_half_inv_ell2 * (*pSqDist)).array().exp()  * (*pSqDist).array()).matrix();
						//std::cout << "K = " << std::endl << *pK << std::endl << std::endl;
						break;
					}
				}

				return pK;
			}

			// pre-calculate the squared distances
			void preCalculateSqDist(MatrixConstPtr pX)
			{
				// pre-calculate the squared distances
				m_pSqDist = selfSqDistances(pX);
			}

		protected:
			// pre-calculated matrices for speed up in training
			MatrixPtr									m_pSqDist;			// squared distances of the training inputs
	};

}

#endif