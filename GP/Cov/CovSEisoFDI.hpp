#ifndef COVARIANCE_FUNCTION_SQUARED_EXPONENTIAL_ISO_BETWEEN_FUNCTION_VALUE_DERIVATIVE_AND_INTEGRAL_HPP
#define COVARIANCE_FUNCTION_SQUARED_EXPONENTIAL_ISO_BETWEEN_FUNCTION_VALUE_DERIVATIVE_AND_INTEGRAL_HPP

#include "GP/Cov/CovFDIBase.hpp"
#include "GP/Cov/CovSEiso.hpp"

namespace GPOM{

class CovSEisoFDI : public CovSEiso, public CovFDIBase
{
protected:
	public:
		// constructor
		CovSEisoFDI() { }

		// destructor
		virtual ~CovSEisoFDI() { }

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

			// dimension
			const int d = PointMatrixDirection::fRowWisePointsMatrix ? pX->cols() : pX->rows();

			// pre-calculate the squared distances and delta
			preCalculateSqDistAndDelta(pX);

			// calculate the covariance matrix
			return K(m_pSqDist, m_pDelta, d, pLogHyp, pdIndex);
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

			// dimension
			const int d = PointMatrixDirection::fRowWisePointsMatrix ? pX->cols() : pX->rows();

			// calculate the squared distances
			MatrixPtr pSqDist = crossSqDistances(pX, pXs);

			// calculate the delta
			DeltaList deltaList(d);
			for(int i = 0; i < d; i++) deltaList[i] = crossDelta(pX, pXs, i);

			// calculate the covariance matrix
			return Ks(pSqDist, deltaList, d, pLogHyp);
		}

		// self-variance/covariance
		//MatrixPtr Kss(MatrixConstPtr pXs, HypConstPtr pLogHyp, bool fVarianceVector = true)

	protected:
		// covariance matrix given pair-wise sqaured distances
		MatrixPtr K_FD(MatrixConstPtr pSqDist, MatrixConstPtr pDelta, HypConstPtr pLogHyp, const int pdIndex = -1) const
		{
			// input
			// pSqDist (nxm): squared distances = r^2
			// pDelta (nxm): delta = x_i - x_i'
			// pLogHyp: log hyperparameters
			// pdIndex: partial derivatives with respect to this parameter index

			// output
			// K: nxm matrix
			// if pdIndex == -1:		K_FF
			// else							partial K_FF / partial theta_i
			MatrixPtr pK_FD = K_FF(pSqDist, pLogHyp, pdIndex);

			// hyperparameters
			Scalar inv_ell2 = exp(((Scalar) -2.f) * (*pLogHyp)(0));

			// mode
			switch(pdIndex)
			{
			// covariance matrix
			case -1:
				{
					// k(X, X') = ((x_i - x_i') / ell^2) * K_FF(X, X')
					(*pK_FD) = inv_ell2 * pDelta->cwiseProduct(*pK_FD);
					//std::cout << "K_FD = " << std::endl << *pK_FD << std::endl << std::endl;
					break;
				}

			// derivatives of covariance matrix w.r.t log ell
			case 0:
				{
					// k_log(ell)	 = ((x_i - x_i') / ell^2) * (K_FF_log(ell) - 2K_FF)
					MatrixPtr pK_FF = K_FF(pSqDist, pLogHyp); // K_FF
					(*pK_FD) = inv_ell2 * pDelta->cwiseProduct(*pK_FD - (((Scalar) 2.f) * (*pK_FF)));
					//std::cout << "K_FD_log(ell) = " << std::endl << *pK_FD << std::endl << std::endl;
					break;
				}

			// derivatives of covariance matrix w.r.t log sigma_f
			case 1:
				{
					// k_log(sigma_f) = ((x_i - x_i') / ell^2) * K_FF_log(sigma_f)
					(*pK_FD) = inv_ell2 * pDelta->cwiseProduct(*pK_FD);
					//std::cout << "K_FD_log(sigma_f) = " << std::endl << *pK_FD << std::endl << std::endl;
					break;
				}
			}

			return pK_FD;
		}

		// covariance matrix given pair-wise sqaured distances
		MatrixPtr K_DD(MatrixConstPtr pSqDist, 
									 MatrixConstPtr pDelta1, const int i, 
									 MatrixConstPtr pDelta2, const int j,
									 HypConstPtr pLogHyp, const int pdIndex = -1) const
		{
			// input
			// pSqDist (nxm): squared distances = r^2
			// pDelta1 (nxm): delta = x_i - x_i'
			// i: index for delta1
			// pDelta2 (nxm): delta = x_j - x_j'
			// j: index for delta2
			// pLogHyp: log hyperparameters
			// pdIndex: partial derivatives with respect to this parameter index

			// output
			// K: nxm matrix
			// if pdIndex == -1:		K_FF
			// else							partial K_FF / partial theta_i
			MatrixPtr pK_DD = K_FF(pSqDist, pLogHyp, pdIndex);

			// hyperparameters
			Scalar inv_ell2				= exp(((Scalar) -2.f) * (*pLogHyp)(0));
			Scalar inv_ell4				= exp(((Scalar) -4.f) * (*pLogHyp)(0));
			Scalar neg2_inv_ell2	= ((Scalar) -2.f) * inv_ell2;
			Scalar four_inv_ell4		= ((Scalar) 4.f) * inv_ell4;

			// delta
			Scalar delta = (i == j) ? (Scalar) 1.f  : (Scalar) 0.f;

#if 0
			// mode
			switch(pdIndex)
			{
			// covariance matrix
			case -1:
				{
					// k(X, X') = [ delta / ell^2 - ((x_i - x_i')*(x_j - x_j') / ell^4) ] * K_FF(X, X')
					(*pK_DD) = (inv_ell2*delta - inv_ell4*(pDelta1->array())*(pDelta2->array())) * pK_DD->array();
					//std::cout << "K_DD = " << std::endl << *pK_DD << std::endl << std::endl;
					break;
				}

			// derivatives of covariance matrix w.r.t log ell
			case 0:
				{
					// k_log(ell)	 = [ -2*delta / ell^2 + 4*((x_i - x_i')*(x_j - x_j') / ell^4) ] * K_FF(X, X')
					//                   + [ delta / ell^2 - ((x_i - x_i')*(x_j - x_j') / ell^4) ] * K_FF_log(ell)(X, X')
					MatrixPtr pK_FF = K_FF(pSqDist, pLogHyp); // K_FF
					(*pK_DD) = (neg2_inv_ell2*delta + four_inv_ell4*(pDelta1->array())*(pDelta2->array())) * pK_FF->array()
									  + (inv_ell2*delta - inv_ell4*(pDelta1->array())*(pDelta2->array())) * pK_DD->array();
					//std::cout << "K_DD_log(ell) = " << std::endl << *pK_DD << std::endl << std::endl;
					break;
				}

			// derivatives of covariance matrix w.r.t log sigma_f
			case 1:
				{
					// k_log(sigma_f) = [ delta / ell^2 - ((x_i - x_i')*(x_j - x_j') / ell^4) ] * K_FF_log(sigma_f)(X, X')
					(*pK_DD) = (inv_ell2*delta - inv_ell4*(pDelta1->array())*(pDelta2->array())) * pK_DD->array();
					//std::cout << "K_DD_log(sigma_f) = " << std::endl << *pK_DD << std::endl << std::endl;
					break;
				}
			}
#else
			// simplified version

			// for all cases
			(*pK_DD) = (inv_ell2*delta - inv_ell4*(pDelta1->array())*(pDelta2->array())) * pK_DD->array();

			// particularly, derivatives of covariance matrix w.r.t log ell
			if(pdIndex == 0)
				(*pK_DD) += (neg2_inv_ell2*delta + four_inv_ell4*(pDelta1->array())*(pDelta2->array())).matrix().cwiseProduct(*(K_FF(pSqDist, pLogHyp)));
				//(*pK_DD) += (neg2_inv_ell2*delta + four_inv_ell4*(pDelta1->array())*(pDelta2->array())) * (K_FF(pSqDist, pLogHyp)->array());
#endif

			return pK_DD;
		}

		//// covariance matrix given pair-wise sqaured distances and delta
		//MatrixPtr K(MatrixPtr pSqDist, ConstDeltaList &deltaList, const int d, HypConstPtr pLogHyp, const int pdIndex = -1) const
		//{
		//	// input
		//	// pSqDist (nxn): squared distances
		//	// deltaList: list of delta (nxn)
		//	// d: dimension of training inputs
		//	// pLogHyp: log hyperparameters
		//	// pdIndex: partial derivatives with respect to this parameter index

		//	// output
		//	// K: n(d+1) by n(d+1)
		//	// 
		//	// for example, when d = 3
		//	//                    |   F (n)   |  D1 (n)  |  D2 (n)  |  D3 (n)  |
		//	// K = -------------------------------------------------------------
		//	//        F    (n) |    FF,          FD1,        FD2,       FD3 
		//	//        D1 (n) |        -,       D1D1,     D1D2,     D1D3
		//	//        D2 (n) |        -,                -,     D2D2,     D2D3
		//	//        D3 (n) |        -,                -,              -,     D3D3

		//	assert(pSqDist->rows() == pSqDist->cols());

		//	const int n = pSqDist->rows();

		//	// covariance matrix
		//	MatrixPtr pK(new Matrix(n*(d+1), n*(d+1))); // n(d+1) by n(d+1)

		//	// fill block matrices of FF, FD and DD in order
		//	for(int row = 0; row <= d; row++)
		//	{
		//		const int startingRow = n*row;
		//		for(int col = row; col <= d; col++)
		//		{
		//			const int startingCol = n*col;

		//			// calculate the upper triangle
		//			if(row == 0)
		//			{
		//				// F-F
		//				if(col == 0)	pK->block(startingRow, startingCol, n, n) = *(K_FF(pSqDist, pLogHyp, pdIndex));

		//				// F-D
		//				else				pK->block(startingRow, startingCol, n, n) = *(K_FD(pSqDist, deltaList[col-1], pLogHyp, pdIndex));
		//			}
		//			else
		//			{
		//				// D-D
		//									pK->block(startingRow, startingCol, n, n) = *(K_DD(pSqDist, 
		//																															 deltaList[row-1], row-1, deltaList[col-1], col-1,
		//																															 pLogHyp, pdIndex));
		//			}

		//			// copy its transpose
		//			if(row != col)	pK->block(startingCol, startingRow, n, n).noalias() = pK->block(startingRow, startingCol, n, n).transpose();
		//		}
		//	}

		//	return pK;
		//}

		// covariance matrix given pair-wise sqaured distances and delta
		MatrixPtr K(MatrixPtr pSqDist, ConstDeltaList &deltaList, const int d, HypConstPtr pLogHyp, const int pdIndex = -1) const
		{
			// input
			// pSqDist (nxn): squared distances
			// deltaList: list of delta (nxn)
			// d: dimension of training inputs
			// pLogHyp: log hyperparameters
			// pdIndex: partial derivatives with respect to this parameter index

			// output
			// K: n(d+1) by n(d+1)
			// 
			// for example, when d = 3
			//                    |   F (n1)   |  D1 (n1)  |  D2 (n1)  |  D3 (n1)  | F (n2) |
			// K = -------------------------------------------------------------
			//        F    (n1) |    FF,           FD1,         FD2,        FD3,         FF
			//        D1 (n1) |        -,        D1D1,      D1D2,      D1D3,      D1F
			//        D2 (n1) |        -,                 -,      D2D2,      D2D3,      D2F
			//        D3 (n1) |        -,                 -,               -,      D3D3,      D3F
			//        F    (n2) |        -,                 -,               -,               -,         FF
			//
 			//                          |    (n1)    |    (n2)   |
			// sqDist = -------------------------------------------------------------
			//                 (n1) |  n1 x n1,   n1 x n2
			//                 (n2) |  n2 x n1,   n2 x n2

			assert(pSqDist->rows() == pSqDist->cols());

			const int n1 = pSqDist->rows() - NumRobotPoses;
			const int n2 = NumRobotPoses;

			// covariance matrix
			MatrixPtr pK(new Matrix(n1*(d+1)+n2, n1*(d+1)+n2)); // n1(d+1)+n2 by n1(d+1)+n2

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
						if(col == 0)	pK->block(startingRow, startingCol, n, n) = *(K_FF(pSqDist.topLeftCorner(n1, n1), pLogHyp, pdIndex));

						// F-D
						else				pK->block(startingRow, startingCol, n, n) = *(K_FD(pSqDist.topLeftCorner(n1, n1), deltaList[col-1], pLogHyp, pdIndex));
					}
					else
					{
						// D-D
											pK->block(startingRow, startingCol, n, n) = *(K_DD(pSqDist, 
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
		MatrixPtr Ks(MatrixConstPtr pSqDist, ConstDeltaList &deltaList, const int d, HypConstPtr pLogHyp) const
		{
			// input
			// pSqDist (nxm): squared distances
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

			const int n = pSqDist->rows();
			const int m = pSqDist->cols();

			// covariance matrix
			MatrixPtr pK(new Matrix(n*(d+1), m)); // n(d+1) x m

			// fill block matrices of FF, FD and DD in order
			for(int row = 0; row <= d; row++)
			{
				// F-F
				if(row == 0)		pK->block(n*row, 0, n, m) = *(K_FF(pSqDist, pLogHyp));

				// D-F
				else					pK->block(n*row, 0, n, m) = ((Scalar) -1.f) * (*(K_FD(pSqDist, deltaList[row-1], pLogHyp)));
			}

			return pK;
		}
};

}

#endif