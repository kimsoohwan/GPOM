#ifndef COVARIANCE_FUNCTION_SQUARED_EXPONENTIAL_ISO_BETWEEN_FUNCTION_VALUE_DERIVATIVE_AND_INTEGRAL_HPP
#define COVARIANCE_FUNCTION_SQUARED_EXPONENTIAL_ISO_BETWEEN_FUNCTION_VALUE_DERIVATIVE_AND_INTEGRAL_HPP

#include <vector>

#include "GP/Cov/CovSEiso.hpp"

namespace GPOM{

class CovSEisoFDI : public CovSEiso
{
	protected:
		// type
		typedef		std::vector<MatrixPtr>								MatrixPtrList;
		typedef		std::vector< std::vector<MatrixPtr> >		MatrixPtrListList;

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
			// K: (n1*(d+1) + n2) x (n1*(d+1) + n2)
			// 
			// for example, when d = 3
			//                    |   F (n1)   |  D1 (n1)  |  D2 (n1)  |  D3 (n1)  | F2 (n2) |
			// K = -------------------------------------------------------------------------------
			//        F1  (n1) |  F1F1,   F1D1,       F1D2,      F1D3,     |  F1F2
			//        D1 (n1) |           -,   D1D1,      D1D2,      D1D3,    |  D1F2
			//        D2 (n1) |           -,            -,      D2D2,      D2D3,    |  D2F2
			//        D3 (n1) |           -,            -,               -,      D3D3,    |  D3F2
			//       --------------------------------------------------------------------------------
			//        F2  (n2) |           -,            -,               -,               -,    |  F2F2
			//
 			//                          |    (n1)    |    (n2)   |
			// sqDist = ----------------------------------
			//                 (n1) |  n1 x n1,   n1 x n2
			//                 (n2) |  n2 x n1,   n2 x n2

			assert(numRobotPoses > 0);

			// numbers of training data and dimension
			const int n	= PointMatrixDirection::fRowWisePointsMatrix ?		pX->rows()		: pX->cols();
			const int d	= PointMatrixDirection::fRowWisePointsMatrix ?		pX->cols()		: pX->rows();
			const int n1	= n - numRobotPoses;
			const int n2	= numRobotPoses;

			// pre-calculate the squared distances and delta
			preCalculateSqDistAndDelta(pX);

			// covariance matrix
			MatrixPtr pK(new Matrix(n1*(d+1)+n2, n1*(d+1)+n2)); // n1(d+1)+n2 by n1(d+1)+n2

			// fill block matrices of FF, FD and DD in order
			for(int row = 0; row <= d; row++)
			{
				const int startingRow = n1*row;
				for(int col = row; col <= d; col++)
				{
					const int startingCol = n1*col;

					// calculate the upper triangle
					if(row == 0)
					{
						// F1F1
						if(col == 0)	pK->block(startingRow, startingCol, n1, n1) = *(K_FF(m_pSqDistList[0], pLogHyp, pdIndex));

						// F1D*
						else				pK->block(startingRow, startingCol, n1, n1) = *(K_FD(m_pSqDistList[0], m_pDeltaListList[0][col-1], pLogHyp, pdIndex));
					}
					else
					{
						// D*D*
											pK->block(startingRow, startingCol, n1, n1) = *(K_DD(m_pSqDistList[0], 
																																		 m_pDeltaListList[0][row-1],		row-1, 
																																		 m_pDeltaListList[0][col-1],		col-1,
																																		 pLogHyp, pdIndex));
					}

					// copy its transpose
					if(row != col)	pK->block(startingCol, startingRow, n1, n1).noalias() = pK->block(startingRow, startingCol, n1, n1).transpose();
				}

				// F1F2
				if(row == 0)		pK->block(startingRow, n1*(d+1), n1, n2) = *(K_FF(m_pSqDistList[1], pLogHyp, pdIndex));

				// D*F2
				else					pK->block(startingRow, n1*(d+1), n1, n2) = *(K_FD(m_pSqDistList[1], m_pDeltaListList[1][row-1], pLogHyp, pdIndex));

				// copy its transpose
				pK->block(n1*(d+1), startingRow, n2, n1).noalias() = pK->block(startingRow, n1*(d+1), n1, n2).transpose();
			}

			// F2F2
			pK->block(n1*(d+1), n1*(d+1), n2, n2) = *(K_FF(m_pSqDistList[2], pLogHyp, pdIndex));

			return pK;
		}

		// cross covariance
		MatrixPtr Ks(MatrixConstPtr pX, MatrixConstPtr pXs, HypConstPtr pLogHyp) const
		{
			// input
			// pX ((n1+n2) x d): training inputs
			// pXs (m x d): test inputs
			// pLogHyp: log hyperparameters

			// output
			// K: (n1*(d+1) + n2) x m

			//                    |  F (m)  |
			// K = ---------------------
			//        F    (n1) |    F1F
			//        D1 (n1) |    D1F
			//        D2 (n1) |    D2F
			//        D3 (n)1 |    D3F
			//        F    (n2) |    F2F

			//                            |    Xs(m)   | 
			// sqDist = ----------------------
			//                 Xf(n1)  |  n1 x m
			//                 Xd(n2) |  n2 x m

			assert(numRobotPoses > 0);

			// numbers of training data and dimension
			const int n	= PointMatrixDirection::fRowWisePointsMatrix ?		pX->rows()		: pX->cols();
			const int m	= PointMatrixDirection::fRowWisePointsMatrix ?		pXs->rows()	: pXs->cols();
			const int d	= PointMatrixDirection::fRowWisePointsMatrix ?		pX->cols()		: pX->rows();
			const int n1	= n - numRobotPoses;
			const int n2	= numRobotPoses;

			// X
			MatrixPtr pX1, pX2;
			if(PointMatrixDirection::fRowWisePointsMatrix)
			{
				pX1.reset(new Matrix(pX->topRows(n1)));
				pX2.reset(new Matrix(pX->bottomRows(n2)));
			}
			else
			{
				pX1.reset(new Matrix(pX->leftCols(n1)));
				pX2.reset(new Matrix(pX->rightCols(n2)));
			}

			// calculate the squared distances
			MatrixPtr pSqDist1	= crossSqDistances(pX1, pXs);		// X1Xs
			MatrixPtr pSqDist2	= crossSqDistances(pX2, pXs);		// X2Xs

			// calculate the delta
			MatrixPtrList deltaList;
			deltaList.resize(d);
			for(int i = 0; i < d; i++) deltaList[i] = crossDelta(pX1, pXs, i);		// X1-Xs

			// covariance matrix
			MatrixPtr pKs(new Matrix(n1*(d+1)+n2, m)); //(n1*(d+1) + n2) x m

			// F1F
			pKs->block(0, 0, n1, m) = *(K_FF(pSqDist1, pLogHyp));

			// D1F, D2F, D3F
			for(int row = 1; row <= d; row++)
				pKs->block(n1*row, 0, n1, m) = ((Scalar) -1.f) * (*(K_FD(pSqDist1, deltaList[row-1], pLogHyp)));

			// F2F
			pKs->block(n1*(d+1), 0, n2, m) = *(K_FF(pSqDist2, pLogHyp));

			return pKs;
		}

		// self-variance/covariance: inherited

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

		// pre-calculate the squared distances and deltas
		bool preCalculateSqDistAndDelta(MatrixConstPtr pX)
		{
			//                              | X1(n1) | X2(n2) |
			// sqDist = -----------------------------------
			//                 X1 (n1) |  X1X1,   X1X2
			//                 X2 (n2) |           -,   X2X2

			//                              | X1(n1) | X2(n2) |
			// delta = -------------------------------------
			//                 X1 (n1) |  X1X1,   X1X2

			// check if the training inputs are the same
			if(m_pTrainingInputs == pX) return false;
			m_pTrainingInputs = pX;

			// numbers of training data and dimension
			const int n = PointMatrixDirection::fRowWisePointsMatrix ? pX->rows() : pX->cols();
			const int d = PointMatrixDirection::fRowWisePointsMatrix ? pX->cols() : pX->rows();
			const int n1 = n - numRobotPoses;
			const int n2 = numRobotPoses;

			assert(n2 > 0);

			// X
			MatrixPtr pX1, pX2;
			if(PointMatrixDirection::fRowWisePointsMatrix)
			{
				pX1.reset(new Matrix(pX->topRows(n1)));
				pX2.reset(new Matrix(pX->bottomRows(n2)));
			}
			else
			{
				pX1.reset(new Matrix(pX->leftCols(n1)));
				pX2.reset(new Matrix(pX->rightCols(n2)));
			}

			// pre-calculate squared distances and delta(upper triangle)
			m_pSqDistList.resize(3);
			m_pDeltaListList.resize(2);

			// squared distances
			m_pSqDistList[0] = selfSqDistances(pX1);						// X1X1
			m_pSqDistList[1] = crossSqDistances(pX1, pX2);			// X1X2
			m_pSqDistList[2] = selfSqDistances(pX2);						// X2X2

			// delta
			m_pDeltaListList[0].resize(d);		for(int i = 0; i < d; i++) m_pDeltaListList[0][i] = selfDelta(pX1, i);					// X1X1
			m_pDeltaListList[1].resize(d);		for(int i = 0; i < d; i++) m_pDeltaListList[1][i] = crossDelta(pX1, pX2, i);	// X1X2
			
			return true;
		}

	protected:
		MatrixPtrList							m_pSqDistList;			// FF, FD, DD: squared distances
		MatrixPtrListList					m_pDeltaListList;		// FF, FD, DD: x_i - x_i'

	public:
		static unsigned int					numRobotPoses;		// n2: X = {X1, X2}
};

// initialize the static member
//unsigned int CovSEisoFDI::numRobotPoses = 0;

}

#endif