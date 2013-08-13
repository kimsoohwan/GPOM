#ifndef COVARIANCE_FUNCTION_MATERN_ISO_BETWEEN_FUNCTION_VALUE_DERIVATIVE_AND_INTEGRAL_HPP
#define COVARIANCE_FUNCTION_MATERN_ISO_BETWEEN_FUNCTION_VALUE_DERIVATIVE_AND_INTEGRAL_HPP

#include <vector>

#include "GP/Cov/CovMaterniso.hpp"

namespace GPOM{

class CovMaterniso3FDI : public CovMaterniso3
{
	protected:
		// type
		typedef		std::vector<MatrixPtr>								MatrixPtrList;
		typedef		std::vector< std::vector<MatrixPtr> >		MatrixPtrListList;

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
			preCalculateDistAndDelta(pX);

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
						if(col == 0)	pK->block(startingRow, startingCol, n1, n1) = *(K_FF(m_pDistList[0], pLogHyp, pdIndex));

						// F1D*
						else				pK->block(startingRow, startingCol, n1, n1) = *(K_FD(m_pDistList[0], m_pDeltaListList[0][col-1], pLogHyp, pdIndex));
					}
					else
					{
						// D*D*
											pK->block(startingRow, startingCol, n1, n1) = *(K_DD(m_pDistList[0], 
																																		 m_pDeltaListList[0][row-1],		row-1, 
																																		 m_pDeltaListList[0][col-1],		col-1,
																																		 pLogHyp, pdIndex));
					}

					// copy its transpose
					if(row != col)	pK->block(startingCol, startingRow, n1, n1).noalias() = pK->block(startingRow, startingCol, n1, n1).transpose();
				}

				// F1F2
				if(row == 0)		pK->block(startingRow, n1*(d+1), n1, n2) = *(K_FF(m_pDistList[1], pLogHyp, pdIndex));

				// D*F2
				else					pK->block(startingRow, n1*(d+1), n1, n2) = *(K_FD(m_pDistList[1], m_pDeltaListList[1][row-1], pLogHyp, pdIndex));

				// copy its transpose
				pK->block(n1*(d+1), startingRow, n2, n1).noalias() = pK->block(startingRow, n1*(d+1), n1, n2).transpose();
			}

			// F2F2
			pK->block(n1*(d+1), n1*(d+1), n2, n2) = *(K_FF(m_pDistList[2], pLogHyp, pdIndex));

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

			// calculate the distances
			MatrixPtr pDist1	= crossSqDistances(pX1, pXs);		pDist1->noalias() = pDist1->cwiseSqrt();		// X1Xs
			MatrixPtr pDist2	= crossSqDistances(pX2, pXs);		pDist2->noalias() = pDist2->cwiseSqrt();		// X2Xs

			// calculate the delta
			MatrixPtrList deltaList;
			deltaList.resize(d);
			for(int i = 0; i < d; i++) deltaList[i] = crossDelta(pX1, pXs, i);		// X1-Xs

			// covariance matrix
			MatrixPtr pKs(new Matrix(n1*(d+1)+n2, m)); //(n1*(d+1) + n2) x m

			// F1F
			pKs->block(0, 0, n1, m) = *(K_FF(pDist1, pLogHyp));

			// D1F, D2F, D3F
			for(int row = 1; row <= d; row++)
				pKs->block(n1*row, 0, n1, m) = ((Scalar) -1.f) * (*(K_FD(pDist1, deltaList[row-1], pLogHyp)));

			// F2F
			pKs->block(n1*(d+1), 0, n2, m) = *(K_FF(pDist2, pLogHyp));

			return pKs;
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

		// pre-calculate the squared distances and deltas
		bool preCalculateDistAndDelta(MatrixConstPtr pX)
		{
			//                              | X1(n1) | X2(n2) |
			// dist = ---------------------------------------
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
			m_pDistList.resize(3);
			m_pDeltaListList.resize(2);

			// squared distances
			m_pDistList[0] = selfSqDistances(pX1);						// X1X1
			m_pDistList[1] = crossSqDistances(pX1, pX2);			// X1X2
			m_pDistList[2] = selfSqDistances(pX2);						// X2X2
			for(int i = 0; i < 3; i++)	m_pDistList[i]->noalias() = m_pDistList[i]->cwiseSqrt();

			// delta
			m_pDeltaListList[0].resize(d);		for(int i = 0; i < d; i++) m_pDeltaListList[0][i] = selfDelta(pX1, i);					// X1X1
			m_pDeltaListList[1].resize(d);		for(int i = 0; i < d; i++) m_pDeltaListList[1][i] = crossDelta(pX1, pX2, i);	// X1X2
			
			return true;
		}

	protected:
		MatrixPtrList							m_pDistList;				// FF, FD, DD: distances
		MatrixPtrListList					m_pDeltaListList;		// FF, FD, DD: x_i - x_i'

	public:
		static unsigned int					numRobotPoses;		// n2: X = {X1, X2}
};

// initialize the static member
//unsigned int CovMaterniso3FDI::numRobotPoses = 0;

}

#endif