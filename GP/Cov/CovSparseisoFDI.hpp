#ifndef COVARIANCE_FUNCTION_SPARSE_ISO_BETWEEN_FUNCTION_VALUE_DERIVATIVE_AND_INTEGRAL_HPP
#define COVARIANCE_FUNCTION_SPARSE_ISO_BETWEEN_FUNCTION_VALUE_DERIVATIVE_AND_INTEGRAL_HPP

#include <vector>

#include "GP/Cov/CovSparseiso.hpp"
#include "GP/util/TrainingInputSetterDerivatives.hpp"

namespace GPOM{

	class CovSparseisoFDI : public CovSparseiso, public TrainingInputSetterDerivatives
	{
		protected:
			// type
			typedef		std::vector<MatrixPtr>								MatrixPtrList;
			typedef		std::vector< std::vector<MatrixPtr> >		MatrixPtrListList;

		public:
			// constructor
			CovSparseisoFDI() { }

			// destructor
			virtual ~CovSparseisoFDI() { }

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
							if(col == 0)	pK->block(startingRow, startingCol, m_nd, m_nd) = *(K_FF(m_pDistList[0], logHyp, pdIndex));

							// F1D*
							else				pK->block(startingRow, startingCol, m_nd, m_nd) = *(K_FD(m_pDistList[0], m_pDeltaListList[0][col-1], logHyp, pdIndex));
						}
						else
						{
							// D*D*
												pK->block(startingRow, startingCol, m_nd, m_nd) = *(K_DD(m_pDistList[0], 
																																			 m_pDeltaListList[0][row-1],		row-1, 
																																			 m_pDeltaListList[0][col-1],		col-1,
																																			 logHyp, pdIndex));
						}

						// copy its transpose
						if(row != col)	pK->block(startingCol, startingRow, m_nd, m_nd).noalias() = pK->block(startingRow, startingCol, m_nd, m_nd).transpose();
					}

					// F1F2
					if(row == 0)		pK->block(startingRow, m_nd*(m_d+1), m_nd, m_n) = *(K_FF(m_pDistList[1], logHyp, pdIndex));

					// D*F2
					else					pK->block(startingRow, m_nd*(m_d+1), m_nd, m_n) = ((Scalar) -1.f) * (*(K_FD(m_pDistList[1], m_pDeltaListList[1][row-1], logHyp, pdIndex)));

					// copy its transpose
					pK->block(m_nd*(m_d+1), startingRow, m_n, m_nd).noalias() = pK->block(startingRow, m_nd*(m_d+1), m_nd, m_n).transpose();
				}

				// F2F2
				pK->block(m_nd*(m_d+1), m_nd*(m_d+1), m_n, m_n) = *(K_FF(m_pDistList[2], logHyp, pdIndex));

				return pK;
			}

			// cross covariance
			MatrixPtr Ks(MatrixConstPtr pXs, const Hyp &logHyp) const
			{
				// input
				// pXs (m x d): test inputs
				// logHyp: log hyperparameters

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
				pKs->block(0, 0, m_nd, m) = *(K_FF(pDist1, logHyp));

				// D1F, D2F, D3F
				for(int row = 1; row <= m_d; row++)
					pKs->block(m_nd*row, 0, m_nd, m) = ((Scalar) -1.f) * (*(K_FD(pDist1, deltaList[row-1], logHyp)));

				// F2F
				pKs->block(m_nd*(m_d+1), 0, m_n, m) = *(K_FF(pDist2, logHyp));

				return pKs;
			}

			// self-variance/covariance
			//MatrixPtr Kss(MatrixConstPtr pXs, const Hyp &logHyp, bool fVarianceVector = true)

		protected:
			// covariance matrix given pair-wise sqaured distances
			MatrixPtr K_FD(MatrixConstPtr pDist, MatrixConstPtr pDelta, const Hyp &logHyp, const int pdIndex = -1) const
			{
				// input
				// pDist (nxm): squared distances = r^2
				// pDelta (nxm): delta = x_i - x_i'
				// logHyp: log hyperparameters
				// pdIndex: partial derivatives with respect to this parameter index

				assert(pdIndex < 2);

				// output
				// K: nxm matrix
				const int n = pDist->rows();
				const int m = pDist->cols();
				MatrixPtr pK_FD(new Matrix(n, m));

				// hyperparameters
				Scalar inv_ell							= exp(((Scalar)  -1.f) * logHyp(0));
				Scalar inv_ell2						= exp(((Scalar)  -2.f) * logHyp(0));
				Scalar sigma_f2					= exp(((Scalar)  2.f) * logHyp(1));

				Scalar inv_three					= ((Scalar) 1.f)/((Scalar) 3.f);
				Scalar inv_two_pi					= ((Scalar) 1.f)/TWO_PI;
				Scalar two_three					= ((Scalar) 2.f) * inv_three;
				Scalar neg_two_three			= - two_three;

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
						// k_log(ell)	 = [(x_j - x'_j)/ (l^2 r)] * (pK_pR + r * p2K_p2R);
						(*pK_FD).noalias() = (inv_ell2 * (pDelta->array() / pR->array())
															* (two_three * sigma_f2 * (pTwoPiR->array().cos() - PI * (pTwoPiR->array().sin()) * (((Scalar) 1.f) - pR->array()) - ((Scalar) 1.f))
																+ (pR->array()) * (neg_two_three * PI * sigma_f2) * (pTwoPiR->array().sin() + TWO_PI * (pTwoPiR->array().cos()) * (((Scalar) 1.f) - pR->array())))).matrix();
						//std::cout << "K_FD_log(ell) = " << std::endl << *pK_FD << std::endl << std::endl;
						break;
					}

				// derivatives of covariance matrix w.r.t log sigma_f
				case 1:
					{
						(*pK_FD).noalias() = ((Scalar) 2.f) * ((-inv_ell2 * (pDelta->array() / pR->array())) 
															* two_three * sigma_f2 * (pTwoPiR->array().cos() - PI * pTwoPiR->array().sin() * (((Scalar) 1.f) - pR->array()) - ((Scalar) 1.f))).matrix();
						//std::cout << "K_FD_log(sigma_f) = " << std::endl << *pK_FD << std::endl << std::endl;
						break;
					}

				// covariance matrix
				default:
					{
						// k(X, X') = -[(x_j - x'_j)/ (l^2 r)] * pK_pR;
						(*pK_FD).noalias() = ((-inv_ell2 * (pDelta->array() / pR->array())) 
															* two_three * sigma_f2 * (pTwoPiR->array().cos() - PI * pTwoPiR->array().sin() * (((Scalar) 1.f) - pR->array()) - ((Scalar) 1.f))).matrix();
						//std::cout << "K_FD = " << std::endl << *pK_FD << std::endl << std::endl;
						break;
					}
				}

				// avoiding division by zero
				for(int row = 0; row < n; row++)  
					for(int col = 0; col < m; col++)   
						if((*pR)(row, col) >= (Scalar) 1.f || (*pDist)(row, col) < EPSILON)		(*pK_FD)(row, col) = (Scalar) 0.f;

				//std::cout << "K_FD = " << std::endl << *pK_FD << std::endl << std::endl;

				return pK_FD;
			}


			// covariance matrix given pair-wise sqaured distances
			MatrixPtr K_DD(MatrixConstPtr pDist, 
										 MatrixConstPtr pDelta1, const int i, 
										 MatrixConstPtr pDelta2, const int j,
										 const Hyp &logHyp, const int pdIndex = -1) const
			{
				// input
				// pDist (nxm): squared distances = r^2
				// pDelta1 (nxm): delta = x_i - x_i'
				// i: index for delta1
				// pDelta2 (nxm): delta = x_j - x_j'
				// j: index for delta2
				// logHyp: log hyperparameters
				// pdIndex: partial derivatives with respect to this parameter index

				// output
				// K: nxm matrix

				assert(pdIndex < 2);

				const int n = pDist->rows();
				const int m = pDist->cols();

				MatrixPtr pK_DD(new Matrix(n, m));

				// hyperparameters
				Scalar ell								= exp(logHyp(0));
				Scalar inv_ell							= exp(((Scalar)  -1.f) * logHyp(0));
				Scalar inv_ell2						= exp(((Scalar)  -2.f) * logHyp(0));
				Scalar sigma_f2					= exp(((Scalar)  2.f) * logHyp(1));

				Scalar inv_three					= ((Scalar) 1.f)/((Scalar) 3.f);
				Scalar inv_two_pi					= ((Scalar) 1.f)/TWO_PI;
				Scalar two_three					= ((Scalar) 2.f) * inv_three;
				Scalar neg_two_three			= - two_three;
				Scalar eight_three					= ((Scalar) 8.f) * inv_three;

				// R, 2*pi*R
				MatrixPtr pR(new Matrix(n, m));
				MatrixPtr pTwoPiR(new Matrix(n, m));
				(*pR).noalias() = inv_ell * (*pDist);
				(*pTwoPiR).noalias() = TWO_PI * (*pR);

				// delta
				Scalar delta = (i == j) ? (Scalar) 1.f  : (Scalar) 0.f;

				// mode
				switch(pdIndex)
				{
				// derivatives of covariance matrix w.r.t log ell
				case 0:
					{
						// k_log(ell)	 = (delta(i, j) / (l^2r) - [(x_i - x'_i)/ (l^2 r)] * [(x_j - x'_j)/ (l^2 r)] / r) * (pK_pR + r * p2K_p2R)
						//                     +[(x_i - x'_i)/ (l^2 r)] * [(x_j - x'_j)/ (l^2 r)] * (2* p2K_p2R + r * p3K_p3R)
						(*pK_DD).noalias() = ((delta * inv_ell2 * (pR->array().inverse()) - inv_ell2 * inv_ell2 * (pDelta1->array()) * (pDelta2->array()) / pR->array().cube())
															  * (two_three * sigma_f2 * (pTwoPiR->array().cos() - PI * (pTwoPiR->array().sin()) * (((Scalar) 1.f) - pR->array()) - ((Scalar) 1.f))
																  + pR->array() * (neg_two_three * PI * sigma_f2) * (pTwoPiR->array().sin() + TWO_PI * (pTwoPiR->array().cos()) * (((Scalar) 1.f) - pR->array())))
														  + (inv_ell2 * inv_ell2 * (pDelta1->array()) * (pDelta2->array()) / pR->array().square())
															 * (((Scalar) 2.f) * (neg_two_three * PI * sigma_f2) * (pTwoPiR->array().sin() + TWO_PI * (pTwoPiR->array().cos()) * (((Scalar) 1.f) - pR->array()))
																  + pR->array() * (eight_three * pow(PI, (Scalar) 3.f) * sigma_f2) * (pTwoPiR->array().sin() * (((Scalar) 1.f) - pR->array())))).matrix();
						//std::cout << "K_DD_log(ell) = " << std::endl << *pK_DD << std::endl << std::endl;
						break;
					}

				// derivatives of covariance matrix w.r.t log sigma_f
				case 1:
					{
						(*pK_DD).noalias() = ((Scalar) 2.f) * ((((inv_ell2 * inv_ell2 * (pDelta1->array()) * (pDelta2->array()) / pR->array().cube()) - delta * inv_ell2 * (pR->array().inverse()))
                                                                         * (two_three * sigma_f2 * (pTwoPiR->array().cos() - PI * pTwoPiR->array().sin() * (((Scalar) 1.f) - pR->array()) - ((Scalar) 1.f)))
							                                           - (inv_ell2 * inv_ell2 * (pDelta1->array()) * (pDelta2->array()) / pR->array().square())
										                                 * (neg_two_three * PI * sigma_f2) * (pTwoPiR->array().sin() + TWO_PI * (pTwoPiR->array().cos()) * (((Scalar) 1.f) - pR->array())))).matrix();
						//std::cout << "K_DD_log(sigma_f) = " << std::endl << *pK_DD << std::endl << std::endl;
						break;
					}

				// covariance matrix
				default:
					{
						// k(X, X') = ([(x_i - x'_i)/ (l^2 r)] * [(x_j - x'_j)/ (l^2 r)] / r  - delta(i, j) / (l^2r)) * pK_pR
						//                  -[(x_i - x'_i)/ (l^2 r)] * [(x_j - x'_j)/ (l^2 r)] * p2K_p2R
						(*pK_DD).noalias() = (((inv_ell2 * inv_ell2 * (pDelta1->array()) * (pDelta2->array()) / pR->array().cube()) - delta * inv_ell2 * (pR->array().inverse()))
                                              * (two_three * sigma_f2 * (pTwoPiR->array().cos() - PI * pTwoPiR->array().sin() * (((Scalar) 1.f) - pR->array()) - ((Scalar) 1.f)))
							               - (inv_ell2 * inv_ell2 * (pDelta1->array()) * (pDelta2->array()) / pR->array().square())
										     * (neg_two_three * PI * sigma_f2) * (pTwoPiR->array().sin() + TWO_PI * (pTwoPiR->array().cos()) * (((Scalar) 1.f) - pR->array()))).matrix();
						//std::cout << "K_DD = " << std::endl << *pK_DD << std::endl << std::endl;
						break;
					}
				}

				// avoiding division by zero
				for(int row = 0; row < n; row++)  
					for(int col = 0; col < m; col++)   
						if((*pR)(row, col) >= (Scalar) 1.f || (*pDist)(row, col) < EPSILON)		(*pK_DD)(row, col) = (Scalar) 0.f;

				//std::cout << "K_DD = " << std::endl << *pK_DD << std::endl << std::endl;

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