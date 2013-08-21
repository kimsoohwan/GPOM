#ifndef COVARIANCE_FUNCTION_SPARSE_ISO_HPP
#define COVARIANCE_FUNCTION_SPARSE_ISO_HPP

#include "GP/Cov/CovMaterniso.hpp"

namespace GPOM{

	class CovSparseiso : public CovMaterniso3
	{
		public:
			// constructor
			CovSparseiso() { }

			// destructor
			virtual ~CovSparseiso() { }

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
				const int n = pDist->rows();
				const int m = pDist->cols();
				MatrixPtr pK(new Matrix(n, m));

				// hyperparameters
				Scalar inv_ell							= exp(((Scalar)  -1.f) * logHyp(0));
				Scalar sigma_f2					= exp(((Scalar)  2.f) * logHyp(1));

				Scalar inv_three					= ((Scalar) 1.f)/((Scalar) 3.f);
				Scalar inv_two_pi					= ((Scalar) 1.f)/TWO_PI;
				Scalar neg_two_three			= - ((Scalar) 2.f) * inv_three;

				// R, 2*pi*R
				MatrixPtr pR(new Matrix(n, m));
				MatrixPtr pTwoPiR(new Matrix(n, m));
				(*pR).noalias() = inv_ell * (*pDist);
				(*pTwoPiR).noalias() = TWO_PI * (*pR);

				// mode
				switch(pdIndex)
				{
				// derivatives of covariance matrix w.r.t log ell
				case 0:
					{
						// k_log(l)		= -r * pK_pR = (-2*sf2/3)*(cos(2*pi*R) - pi*sin(2*pi*R).*(1-R) - 1).*R;
						(*pK).noalias() = neg_two_three * sigma_f2 * ((pTwoPiR->array().cos() - PI * (pTwoPiR->array().sin()) * (((Scalar) 1.f) - pR->array()) - ((Scalar) 1.f)) * pR->array()).matrix();
						break;
					}

				// derivatives of covariance matrix w.r.t log sigma_f
				case 1:
					{
						(*pK).noalias() = ((Scalar) 2.f) * sigma_f2 * (inv_three * (((Scalar) 2.f) + pTwoPiR->array().cos()) * (((Scalar) 1.f) - pR->array()) + inv_two_pi * pTwoPiR->array().sin()).matrix();
						break;
					}

				// covariance matrix
				default:
					{
						// K = sf2*((2+cos(2*pi*R)).*(1-R)/3 + sin(2*pi*R)/(2*pi));
						(*pK).noalias() = sigma_f2 * (inv_three * (((Scalar) 2.f) + pTwoPiR->array().cos()) * (((Scalar) 1.f) - pR->array()) + inv_two_pi * pTwoPiR->array().sin()).matrix();
						break;
					}
				}

				// sparse
				for(int row = 0; row < n; row++)  
					for(int col = 0; col < m; col++)   
						if((*pR)(row, col) >= (Scalar) 1.f)		(*pK)(row, col) = (Scalar) 0.f;

				//std::cout << "K = " << std::endl << *pK << std::endl << std::endl;
				return pK;
			}
	};

}

#endif
