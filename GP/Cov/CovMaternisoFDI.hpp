#ifndef COVARIANCE_FUNCTION_MATERN_ISO_BETWEEN_FUNCTION_VALUE_DERIVATIVE_AND_INTEGRAL_HPP
#define COVARIANCE_FUNCTION_MATERN_ISO_BETWEEN_FUNCTION_VALUE_DERIVATIVE_AND_INTEGRAL_HPP

#include <vector>

#include "GP/Cov/CovMaterniso.hpp"
#include "GP/util/TrainingInputSetterDerivatives.hpp"

namespace GPOM{

	class CovMaterniso3FDI : public CovMaterniso3, public TrainingInputSetterDerivatives
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

			// setter
		private:
			bool setTrainingInputs(MatrixConstPtr pX) { }
		public:
			bool setTrainingInputs(MatrixConstPtr pXd, MatrixConstPtr pX)
			{
				// pX: function observations
				// pXd: function and derivative observations

				assert(PointMatrixDirection::fRowWisePointsMatrix ? pX->cols() == pXd->cols() : pX->rows() == pXd->rows());

				// check if the training inputs are the same
				bool fDifferentX = TrainingInputSetter::setTrainingInputs(pX);
				bool fDifferentXd = TrainingInputSetterDerivatives::setTrainingInputs(pXd);
				if(!fDifferentX && !fDifferentXd)		return false;

				// pre_calculate the squared distances
				preCalculateDistAndDelta(pXd, pX);

				return true;
			}

			// getter
			virtual int	getN() const { return m_nd*(m_d+1)+m_n; }

			// operator
			MatrixPtr operator()(HypConstPtr pLogHyp, const int pdIndex = -1)
			{
				return K(pLogHyp, pdIndex);
			}

			// self covariance
			MatrixPtr K(HypConstPtr pLogHyp, const int pdIndex = -1)
			{
				// input
				// pX (nxd): training inputs
				// pLogHyp: log hyperparameters
				// pdIndex: partial derivatives with respect to this parameter index

				// output
				// K: (nd*(d+1) + n) x (nd*(d+1) + n)
				// 
				// for example, when d = 3
				//                    |   F (nd)   |  D1 (nd)  |  D2 (nd)  |  D3 (nd)  | F2 (n) |
				// K = -------------------------------------------------------------------------------
				//        F1  (nd) |  F1F1,   F1D1,       F1D2,      F1D3,     |  F1F2
				//        D1 (nd) |           -,   D1D1,      D1D2,      D1D3,    |  D1F2
				//        D2 (nd) |           -,            -,      D2D2,      D2D3,    |  D2F2
				//        D3 (nd) |           -,            -,               -,      D3D3,    |  D3F2
				//       --------------------------------------------------------------------------------
				//        F2  (n) |           -,            -,               -,               -,      |  F2F2
				//
 				//                          |    (nd)    |    (n)   |
				// sqDist = ----------------------------------
				//                 (nd) |  nd x nd,   nd x nd
				//                 (n)   |             -,     n x n

				// number of training data
				const int n = getN();

				// covariance matrix
				MatrixPtr pK(new Matrix(n, n)); // nd(d+1)+n by nd(d+1)+n

				// fill block matrices of FF, FD and DD in order
				for(int row = 0; row <= m_d; row++)
				{
					const int startingRow = m_nd*row;
					for(int col = row; col <= m_d; col++)
					{
						const int startingCol = m_nd*col;

						// calculate the upper triangle
						if(row == 0)
						{
							// F1F1
							if(col == 0)	pK->block(startingRow, startingCol, m_nd, m_nd) = *(K_FF(m_pDistList[0], pLogHyp, pdIndex));

							// F1D*
							else				pK->block(startingRow, startingCol, m_nd, m_nd) = *(K_FD(m_pDistList[0], m_pDeltaListList[0][col-1], pLogHyp, pdIndex));
						}
						else
						{
							// D*D*
												pK->block(startingRow, startingCol, m_nd, m_nd) = *(K_DD(m_pDistList[0], 
																																			 m_pDeltaListList[0][row-1],		row-1, 
																																			 m_pDeltaListList[0][col-1],		col-1,
																																			 pLogHyp, pdIndex));
						}

						// copy its transpose
						if(row != col)	pK->block(startingCol, startingRow, m_nd, m_nd).noalias() = pK->block(startingRow, startingCol, m_nd, m_nd).transpose();
					}

					// F1F2
					if(row == 0)		pK->block(startingRow, m_nd*(m_d+1), m_nd, m_n) = *(K_FF(m_pDistList[1], pLogHyp, pdIndex));

					// D*F2
					else					pK->block(startingRow, m_nd*(m_d+1), m_nd, m_n) = ((Scalar) -1.f) * (*(K_FD(m_pDistList[1], m_pDeltaListList[1][row-1], pLogHyp, pdIndex)));

					// copy its transpose
					pK->block(m_nd*(m_d+1), startingRow, m_n, m_nd).noalias() = pK->block(startingRow, m_nd*(m_d+1), m_nd, m_n).transpose();
				}

				// F2F2
				pK->block(m_nd*(m_d+1), m_nd*(m_d+1), m_n, m_n) = *(K_FF(m_pDistList[2], pLogHyp, pdIndex));

				return pK;
			}

			// cross covariance
			MatrixPtr Ks(MatrixConstPtr pXs, HypConstPtr pLogHyp) const
			{
				// input
				// pXs (m x d): test inputs
				// pLogHyp: log hyperparameters

				// output
				// K: (nd*(d+1) + n) x m

				//                    |  F (m)  |
				// K = ---------------------
				//        F    (nd) |    F1F
				//        D1 (nd) |    D1F
				//        D2 (nd) |    D2F
				//        D3 (nd) |    D3F
				//        F    (n)   |    F2F

				//                               |    Xs(m)   | 
				// sqDist = ----------------------
				//                 Xd(nd)  |  nd x m
				//                 X(n)      |   n x m

				// number of training data
				const int n = getN();
				const int m = PointMatrixDirection::fRowWisePointsMatrix ? pXs->rows() : pXs->cols();

				// calculate the distances
				MatrixPtr pDist1	= crossSqDistances(m_pXd, pXs);		pDist1->noalias() = pDist1->cwiseSqrt();		// XdXs
				MatrixPtr pDist2	= crossSqDistances(m_pX, pXs);			pDist2->noalias() = pDist2->cwiseSqrt();		// XXs

				// calculate the delta
				MatrixPtrList deltaList;
				deltaList.resize(m_d);
				for(int i = 0; i < m_d; i++) deltaList[i] = crossDelta(m_pXd, pXs, i);		// XdXs

				// covariance matrix
				MatrixPtr pKs(new Matrix(n, m)); // (nd*(d+1) + n) x m

				// F1F
				pKs->block(0, 0, m_nd, m) = *(K_FF(pDist1, pLogHyp));

				// D1F, D2F, D3F
				for(int row = 1; row <= m_d; row++)
					pKs->block(m_nd*row, 0, m_nd, m) = ((Scalar) -1.f) * (*(K_FD(pDist1, deltaList[row-1], pLogHyp)));

				// F2F
				pKs->block(m_nd*(m_d+1), 0, m_n, m) = *(K_FF(pDist2, pLogHyp));

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

				assert(pdIndex < 2);

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

				// covariance matrix
				default:
					{
						// k(X, X') = 3 * sigma_f^2 * ((x_i - x_i') / ell^2) * exp(-sqrt(3) * r / ell)
						(*pK_FD) = three_sigma_f2_inv_ell2 * pDelta->array() * negR.array().exp();
						//std::cout << "K_FD = " << std::endl << *pK_FD << std::endl << std::endl;
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

				assert(pdIndex < 2);

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

				// covariance matrix
				default:
					{
						// k(X, X') = 3 * sigma_f^2 * [ delta / ell^2 -3 * (ell / (sqrt(3) * r)) * ((x_i - x_i') / ell^2) * ((x_j - x_j') / ell^2) ] * exp(- sqrt(3) * r / ell)
						//               = (3 * sigma_f^2 / ell^2) * [ delta - (3 / ell^2) *  (x_i - x_i') * (x_j - x_j') / (sqrt(3) * r / ell)  ] * exp(- sqrt(3) * r / ell)
						(*pK_DD) = three_sigma_f2_inv_ell2 * (delta + three_inv_ell2 * (pDelta1->array()) * (pDelta2->array()) / negR.array()) * negR.array().exp();

						// make the distance always greater than eps
						for(int row = 0; row < n; row++)   for(int col = 0; col < m; col++)   if(negR(row, col) > - EPSILON)		(*pK_DD)(row, col) = three_sigma_f2_inv_ell2 * delta;
						//std::cout << "K_DD = " << std::endl << *pK_DD << std::endl << std::endl;
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
			void preCalculateDistAndDelta(MatrixConstPtr pXd, MatrixConstPtr pX)
			{
				//                              | Xd(nd) | X(n) |
				// dist = -----------------------------------
				//                 Xd (nd) |  XdXd,   XdX
				//                 X  (n)    |           -,      XX

				//                              | Xd(nd) | X(n) |
				// delta = -------------------------------------
				//                 Xd (nd) |  XdXd,   XdX

				// pre-calculate squared distances and delta(upper triangle)
				m_pDistList.resize(3);
				m_pDeltaListList.resize(2);

				// squared distances
				m_pDistList[0] = selfSqDistances(pXd);					// XdXd
				m_pDistList[1] = crossSqDistances(pXd, pX);			// XdX
				m_pDistList[2] = selfSqDistances(pX);						// XX
				for(int i = 0; i < 3; i++)	m_pDistList[i]->noalias() = m_pDistList[i]->cwiseSqrt();

				// delta
				m_pDeltaListList[0].resize(m_d);		for(int i = 0; i < m_d; i++) m_pDeltaListList[0][i] = selfDelta(pXd, i);					// XdXd
				m_pDeltaListList[1].resize(m_d);		for(int i = 0; i < m_d; i++) m_pDeltaListList[1][i] = crossDelta(pXd, pX, i);		// XdX
			}

		protected:
			MatrixPtrList							m_pDistList;				// FF, FD, DD: distances
			MatrixPtrListList					m_pDeltaListList;		// FF, FD, DD: x_i - x_i'
	};

}

#endif