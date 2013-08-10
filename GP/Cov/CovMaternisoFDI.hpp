#ifndef COVARIANCE_FUNCTION_MATERN_ISO_BETWEEN_FUNCTION_VALUE_DERIVATIVE_AND_INTEGRALHPP
#define COVARIANCE_FUNCTION_MATERN_ISO_BETWEEN_FUNCTION_VALUE_DERIVATIVE_AND_INTEGRALHPP

#include "GP/Cov/CovMaterniso.hpp"

namespace GP{

class CovMaterniso3FDI : public CovMaterniso3
{
protected:
	public:
		// constructor
		CovMaterniso3FDI() { }

		// destructor
		virtual ~CovMaterniso3FDI() { }

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

			// pre-calculate the squared distances and delta
			preCalculateDistAndDelta(pX);

			// calculate the covariance matrix
			return K(m_pDist, m_pDelta, pX->cols(), pLogHyp, pdIndex);
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

			// calculate the delta
			const int d = pX->cols();
			DeltaList deltaList(d);
			for(int i = 0; i < d; i++) deltaList[i] = crossDelta(pX, pXs, i);

			// calculate the covariance matrix
			return Ks(pDist, deltaList, d, pLogHyp);
		}

		// self-variance/covariance
		//MatrixPtr Kss(MatrixConstPtr pXs, HypConstPtr pLogHyp, bool fVarianceVector = true)

	protected:
		// covariance matrix given pair-wise sqaured distances
		MatrixPtr K_FD(MatrixConstPtr pDist, MatrixConstPtr pDelta, HypConstPtr pLogHyp, const int pdIndex = -1) const
		{
			// input
			// pDist (nxm): squared distances = r^2
			// pDelta (nxm): delta = x_i - x_i'
			// pLogHyp: log hyperparameters
			// pdIndex: partial derivatives with respect to this parameter index

			// output
			// K: nxm matrix
			const int n = pDist->rows();
			const int m = pDist->cols();
			MatrixPtr pK_FD(new Matrix(n, m));

			// hyperparameters
			Scalar inv_ell										= exp(((Scalar) -1.f) * (*pLogHyp)(0));
			Scalar inv_ell2									= exp(((Scalar) -2.f) * (*pLogHyp)(0));
			Scalar neg_sqrt3_inv_ell					=  -SQRT3 * inv_ell;

			Scalar three_sigma_f2_inv_ell2		= ((Scalar)  3.f) * exp(((Scalar)  2.f) * (*pLogHyp)(1)) * inv_ell2;
			Scalar six_sigma_f2_inv_ell2			= ((Scalar)  2.f) * three_sigma_f2_inv_ell2;

			// pre-compute negR = - sqrt(3) * r / ell
			Matrix negR(n, m);
			negR.noalias() = neg_sqrt3_inv_ell * (*pDist);

#if 1
			// mode
			switch(pdIndex)
			{
			// covariance matrix
			case -1:
				{
					// k(X, X') = 3 * sigma_f^2 * ((x_i - x_i') / ell^2) * exp(-sqrt(3) * r / ell)
					(*pK_FD) = three_sigma_f2_inv_ell2 * pDelta->array() * negR.array().exp();
					//std::cout << "K_FD = " << std::endl << *pK_FD << std::endl << std::endl;
					break;
				}

			// derivatives of covariance matrix w.r.t log ell
			case 0:
				{
					// k_log(ell)	 = 3 * sigma_f^2 * ((x_i - x_i') / ell^2) * (sqrt(3) * r / ell  - 2) * exp(-sqrt(3) * r / ell)
					(*pK_FD) = three_sigma_f2_inv_ell2 * pDelta->array() * (((Scalar) -2.f) - negR.array()) * negR.array().exp();
					//std::cout << "K_FD_log(ell) = " << std::endl << *pK_FD << std::endl << std::endl;
					break;
				}

			// derivatives of covariance matrix w.r.t log sigma_f
			case 1:
				{
					// k(X, X') = 6 * sigma_f^2 * ((x_i - x_i') / ell^2) * exp(-sqrt(3) * r / ell)
					(*pK_FD) = six_sigma_f2_inv_ell2 * pDelta->array() * negR.array().exp();
					//std::cout << "K_FD_log(sigma_f) = " << std::endl << *pK_FD << std::endl << std::endl;
					break;
				}
			}
#else
			// k(X, X') = 3 * sigma_f^2 * ((x_i - x_i') / ell^2) * exp(-sqrt(3) * r / ell)
			(*pK_FD) = three_sigma_f2_inv_ell2 * pDelta->array() * negR.array().exp();

			if(pdIndex == 0) (*pK_FD) = pK_FD->array() * (((Scalar) -2.f) - negR.array());
			if(pdIndex == 1) (*pK_FD) = ((Scalar) 2.f) * (*pK_FD);
#endif

			return pK_FD;
		}


		// covariance matrix given pair-wise sqaured distances
		MatrixPtr K_DD(MatrixConstPtr pDist, 
									 MatrixConstPtr pDelta1, const int i, 
									 MatrixConstPtr pDelta2, const int j,
									 HypConstPtr pLogHyp, const int pdIndex = -1) const
		{
			// input
			// pDist (nxm): squared distances = r^2
			// pDelta1 (nxm): delta = x_i - x_i'
			// i: index for delta1
			// pDelta2 (nxm): delta = x_j - x_j'
			// j: index for delta2
			// pLogHyp: log hyperparameters
			// pdIndex: partial derivatives with respect to this parameter index

			// output
			// K: nxm matrix
			const int n = pDist->rows();
			const int m = pDist->cols();

			MatrixPtr pK_DD(new Matrix(n, m));

			// hyperparameters
			Scalar inv_ell										= exp(((Scalar) -1.f) * (*pLogHyp)(0));
			Scalar inv_ell2									= exp(((Scalar) -2.f) * (*pLogHyp)(0));
			Scalar neg_sqrt3_inv_ell					= -SQRT3 * inv_ell;
			Scalar three_inv_ell2						= ((Scalar) 3.f) * inv_ell2;
			Scalar nine_inv_ell2							= ((Scalar) 9.f) * inv_ell2;

			Scalar three_sigma_f2_inv_ell2		= ((Scalar) 3.f) * exp(((Scalar) 2.f) * (*pLogHyp)(1)) * inv_ell2;
			Scalar six_sigma_f2_inv_ell2			= ((Scalar) 2.f) * three_sigma_f2_inv_ell2;

			// delta
			Scalar delta = (i == j) ? (Scalar) 1.f  : (Scalar) 0.f;

			// pre-compute negR = - sqrt(3) * r / ell
			Matrix negR(n, m);
			negR.noalias() = neg_sqrt3_inv_ell * (*pDist);

			// avoiding division by zero
			//// make the distance always greater than eps
			//for(int row = 0; row < n; row++)
			//	for(int col = 0; col < m; col++)
			//		if(negR(row, col) > - EPSILON)		negR(row, col) = - EPSILON;
			//std::cout << "Dist  = " << std::endl << *pDist << std::endl << std::endl;
			//std::cout << "-R  = " << std::endl << negR << std::endl << std::endl;

#if 1
			// mode
			switch(pdIndex)
			{
			// covariance matrix
			case -1:
				{
					// k(X, X') = 3 * sigma_f^2 * [ delta / ell^2 -3 * (ell / (sqrt(3) * r)) * ((x_i - x_i') / ell^2) * ((x_j - x_j') / ell^2) ] * exp(- sqrt(3) * r / ell)
					//               = (3 * sigma_f^2 / ell^2) * [ delta - (3 / ell^2) *  (x_i - x_i') * (x_j - x_j') / (sqrt(3) * r / ell)  ] * exp(- sqrt(3) * r / ell)
					(*pK_DD) = three_sigma_f2_inv_ell2 * (delta + three_inv_ell2 * (pDelta1->array()) * (pDelta2->array()) / negR.array()) * negR.array().exp();

					// make the distance always greater than eps
					for(int row = 0; row < n; row++)   for(int col = 0; col < m; col++)   if(negR(row, col) > - EPSILON)		(*pK_DD)(row, col) = three_sigma_f2_inv_ell2 * delta;
					//std::cout << "K_DD = " << std::endl << *pK_DD << std::endl << std::endl;
					break;
				}

			// derivatives of covariance matrix w.r.t log ell
			case 0:
				{
					// k_log(ell)	 = 3 * sigma_f^2 * [ (-2*delta / ell^2 + 9* (ell / (sqrt(3) * r)) * ((x_i - x_i') / ell^2) * ((x_j - x_j') / ell^2) 
					//                                                    +(delta / ell^2 - 3 * (ell / (sqrt(3) * r)) * ((x_i - x_i') / ell^2) * ((x_j - x_j') / ell^2)) * (sqrt(3) * r / ell) ] * exp(- sqrt(3) * r / ell)
					//                   = (3 * sigma_f^2 / ell^2) * [ (-2*delta + (9 / ell^2) (x_i - x_i') * (x_j - x_j') / (sqrt(3) * r / ell)) 
					//                                                                 +(delta/ (sqrt(3) * r / ell)  - (3 / ell^2) * (x_i - x_i') * (x_j - x_j')) ] * exp(- sqrt(3) * r / ell)
					(*pK_DD) = three_sigma_f2_inv_ell2 * (((Scalar) -2.f) * delta - nine_inv_ell2 * (pDelta1->array()) * (pDelta2->array()) / negR.array()
																						- delta * negR.array() - three_inv_ell2 * (pDelta1->array()) * (pDelta2->array())) * negR.array().exp();

					// make the distance always greater than eps
					for(int row = 0; row < n; row++)   for(int col = 0; col < m; col++)  if(negR(row, col) > - EPSILON)		(*pK_DD)(row, col) = three_sigma_f2_inv_ell2 * ((Scalar) -2.f) * delta;
					//std::cout << "K_DD_log(ell) = " << std::endl << *pK_DD << std::endl << std::endl;
					break;
				}

			// derivatives of covariance matrix w.r.t log sigma_f
			case 1:
				{
					// k_log(sigma_f) = 6 * sigma_f^2 * [ delta / ell^2 -3 * (ell / (sqrt(3) * r)) * ((x_i - x_i') / ell^2) * ((x_j - x_j') / ell^2) ] * exp(- sqrt(3) * r / ell)
					//                            = (6 * sigma_f^2 / ell^2) * [ delta - (3 / ell^2) *  (x_i - x_i') * (x_j - x_j') / (sqrt(3) * r / ell)  ] * exp(- sqrt(3) * r / ell)
					(*pK_DD) = six_sigma_f2_inv_ell2 * (delta + three_inv_ell2 * (pDelta1->array()) * (pDelta2->array()) / negR.array()) * negR.array().exp();

					// make the distance always greater than eps
					for(int row = 0; row < n; row++)   for(int col = 0; col < m; col++)   if(negR(row, col) > - EPSILON)   (*pK_DD)(row, col) = six_sigma_f2_inv_ell2 * delta;
					//std::cout << "K_DD_log(sigma_f) = " << std::endl << *pK_DD << std::endl << std::endl;
					break;
				}
			}
#else
			// simplified version

			// for all cases
			(*pK_DD) = three_sigma_f2_inv_ell2 * (delta + three_inv_ell2 * (pDelta1->array()) * (pDelta2->array()) / negR.array()) * negR.array().exp();

			// make the distance always greater than eps
			for(int row = 0; row < n; row++)   for(int col = 0; col < m; col++)   if(negR(row, col) > - EPSILON)   (*pK_DD)(row, col) = three_sigma_f2_inv_ell2 * delta;

			// particularly, derivatives of covariance matrix w.r.t log ell
			if(pdIndex == 0)
			{
				(*pK_DD) = pK_DD->cwiseProduct(- negR) 
															  + three_sigma_f2_inv_ell2 * (((Scalar) -2.f) * delta - nine_inv_ell2 * (pDelta1->array()) * (pDelta2->array()) / negR.array()) * negR.array().exp();

				// make the distance always greater than eps
				for(int row = 0; row < n; row++)   for(int col = 0; col < m; col++)   if(negR(row, col) > - EPSILON)   (*pK_DD)(row, col) = three_sigma_f2_inv_ell2 * ((Scalar) -2.f) * delta;
			}

			// particularly, derivatives of covariance matrix w.r.t log sigma_f
			if(pdIndex == 1)		(*pK_DD) *= (Scalar) 2.f;
#endif

			return pK_DD;
		}

		// covariance matrix given pair-wise sqaured distances and delta
		MatrixPtr K(MatrixConstPtr pDist, ConstDeltaList &deltaList, const int d, HypConstPtr pLogHyp, const int pdIndex = -1) const
		{
			// input
			// pDist (nxn): squared distances
			// deltaList: list of delta (nxn)
			// d: dimension of training inputs
			// pLogHyp: log hyperparameters
			// pdIndex: partial derivatives with respect to this parameter index

			// output
			// K: n(d+1) by n(d+1)
			// 
			// for example, when d = 3
			//                    |   F (n)   |  D1 (n)  |  D2 (n)  |  D3 (n)  |
			// K = ---------------------
			//        F    (n) |    FF,          FD1,        FD2,       FD3 
			//        D1 (n) |        -,       D1D1,     D1D2,     D1D3
			//        D2 (n) |        -,                -,     D2D2,     D2D3
			//        D3 (n) |        -,                -,              -,     D3D3

			assert(pDist->rows() == pDist->cols());
			//std::cout << "Dist  = " << std::endl << *pDist << std::endl << std::endl;

			const int n = pDist->rows();

			// covariance matrix
			MatrixPtr pK(new Matrix(n*(d+1), n*(d+1))); // n(d+1) by n(d+1)

			// fill block matrices of FF, FD and DD in order
			for(int row = 0; row <= d; row++)
			{
				const int startingRow = n*row;
				for(int col = row; col <= d; col++)
				{
					const int startingCol = n*col;

					// calculate the upper triangle
					if(row == 0)
					{
						// F-F
						if(col == 0)	pK->block(startingRow, startingCol, n, n) = *(K_FF(pDist, pLogHyp, pdIndex));

						// F-D
						else				pK->block(startingRow, startingCol, n, n) = *(K_FD(pDist, deltaList[col-1], pLogHyp, pdIndex));
					}
					else
					{
						// D-D
											pK->block(startingRow, startingCol, n, n) = *(K_DD(pDist, 
																																	 deltaList[row-1], row-1, deltaList[col-1], col-1,
																																	 pLogHyp, pdIndex));
					}

					// copy its transpose
					if(row != col)	pK->block(startingCol, startingRow, n, n).noalias() = pK->block(startingRow, startingCol, n, n).transpose();
				}
			}

			return pK;
		}

		// covariance matrix given pair-wise sqaured distances and delta
		MatrixPtr Ks(MatrixConstPtr pDist, ConstDeltaList &deltaList, const int d, HypConstPtr pLogHyp) const
		{
			// input
			// pDist (nxm): squared distances
			// deltaList: list of delta (nxm)
			// d: dimension of training inputs
			// pLogHyp: log hyperparameters
			// pdIndex: partial derivatives with respect to this parameter index

			// output
			// K: n(d+1) x m
			// 
			// for example, when d = 3
			//                    |  F (m)  |
			// K = ---------------------
			//        F    (n) |       FF
			//        D1 (n) |    D1F
			//        D2 (n) |    D2F
			//        D3 (n) |    D3F

			const int n = pDist->rows();
			const int m = pDist->cols();

			// covariance matrix
			MatrixPtr pK(new Matrix(n*(d+1), m)); // n(d+1) x m

			// fill block matrices of FF, FD and DD in order
			for(int row = 0; row <= d; row++)
			{
				// F-F
				if(row == 0)		pK->block(n*row, 0, n, m) = *(K_FF(pDist, pLogHyp));

				// D-F
				else					pK->block(n*row, 0, n, m) = ((Scalar) -1.f) * (*(K_FD(pDist, deltaList[row-1], pLogHyp)));
			}

			return pK;
		}
};

}

#endif