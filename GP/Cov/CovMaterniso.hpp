#ifndef COVARIANCE_FUNCTION_MATERN_ISO_HPP
#define COVARIANCE_FUNCTION_MATERN_ISO_HPP

#include "GP/util/TrainingInputSetter.hpp"
#include "GP/util/sqDistances.hpp"

namespace GPOM{

	class CovMaterniso3 : public TrainingInputSetter
	{
		public:
			// hyperparameters
			typedef	Eigen::Matrix<Scalar, 2, 1>							Hyp;						// ell, sigma_f

		public:
			// constructor
			CovMaterniso3() { }

			// destructor
			virtual ~CovMaterniso3() { }

			// setter
			bool setTrainingInputs(MatrixConstPtr pX)
			{
				// check if the training inputs are the same
				if(!TrainingInputSetter::setTrainingInputs(pX))		return false;

				// pre_calculate the squared distances
				preCalculateDist(pX);

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
				return K_FF(m_pDist, logHyp, pdIndex);
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

				// calculate the distances
				MatrixPtr pDist = crossSqDistances(m_pX, pXs);		// squared distances
				pDist->noalias() = pDist->cwiseSqrt();					// distances

				// calculate the covariance matrix
				return K_FF(pDist, logHyp);
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
					MatrixPtr pDist = selfSqDistances(pXs);		// squared distances
					pDist->noalias() = pDist->cwiseSqrt();			// distances

					// calculate the covariance matrix
					pKss = K_FF(pDist, logHyp);
				}

				return pKss;
			}

		protected:
			// covariance matrix given pair-wise sqaured distances
			virtual MatrixPtr K_FF(MatrixConstPtr pDist, const Hyp &logHyp, const int pdIndex = -1) const
			{
				// input
				// pDist (nxm): distances
				// logHyp: log hyperparameters
				// pdIndex: partial derivatives with respect to this parameter index

				assert(pdIndex < 2);

				// output
				// K: nxm matrix
				MatrixPtr pK(new Matrix(pDist->rows(), pDist->cols()));

				// hyperparameters
				Scalar inv_ell							= exp(((Scalar)  -1.f) * logHyp(0));
				Scalar sigma_f2					= exp(((Scalar)  2.f) * logHyp(1));
				Scalar neg_sqrt3_inv_ell		= - SQRT3 * inv_ell;
				Scalar twice_sigma_f2			= ((Scalar) 2.f) * sigma_f2;

				// mode
				switch(pdIndex)
				{
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

				// covariance matrix
				default:
					{
						// k(X, X') = sigma_f^2 *(1 + sqrt(3)*r / ell) * exp(- sqrt(3)*r / ell)
						(*pK).noalias() = neg_sqrt3_inv_ell * (*pDist);
						(*pK) = sigma_f2 * (((Scalar) 1.f) - pK->array()) * pK->array().exp();
						break;
					}
				}

				//if(pK->hasNaN())		std::cout << "CovMaterniso::K_FF has NaN." << std::endl;

				return pK;
			}

			// pre-calculate the squared distances
			void preCalculateDist(MatrixConstPtr pX)
			{
				// pre-calculate the distances
				m_pDist = selfSqDistances(pX);						// squared distances
				m_pDist->noalias() = m_pDist->cwiseSqrt();	// distances
			}

		protected:
			// pre-calculated matrices for speed up in training
			MatrixPtr									m_pDist;					// distances of the training inputs
	};

}

#endif