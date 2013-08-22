#ifndef COVARIANCE_FUNCTION_SQUARED_EXPONENTIAL_ISO_BETWEEN_FUNCTION_VALUE_DERIVATIVE_AND_INTEGRAL_HPP
#define COVARIANCE_FUNCTION_SQUARED_EXPONENTIAL_ISO_BETWEEN_FUNCTION_VALUE_DERIVATIVE_AND_INTEGRAL_HPP

#include <vector>

#include "GP/Cov/CovSEiso.hpp"
#include "GP/util/TrainingInputSetterDerivatives.hpp"

namespace GPOM{

	class CovSEisoFDI : public CovSEiso, public TrainingInputSetterDerivatives
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
				preCalculateSqDistAndDelta(pXd, pX);

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
							if(col == 0)	pK->block(startingRow, startingCol, m_nd, m_nd) = *(K_FF(m_pSqDistList[0], logHyp, pdIndex));

							// F1D*
							else				pK->block(startingRow, startingCol, m_nd, m_nd) = *(K_FD(m_pSqDistList[0], m_pDeltaListList[0][col-1], logHyp, pdIndex));
						}
						else
						{
							// D*D*
												pK->block(startingRow, startingCol, m_nd, m_nd) = *(K_DD(m_pSqDistList[0], 
																																			 m_pDeltaListList[0][row-1],		row-1, 
																																			 m_pDeltaListList[0][col-1],		col-1,
																																			 logHyp, pdIndex));
						}

						// copy its transpose
						if(row != col)	pK->block(startingCol, startingRow, m_nd, m_nd).noalias() = pK->block(startingRow, startingCol, m_nd, m_nd).transpose();
					}

					if(m_n > 0)
					{
						const int startingCol = m_nd*(m_d+1);

						// F1F2
						if(row == 0)		pK->block(startingRow, startingCol, m_nd, m_n) = *(K_FF(m_pSqDistList[1], logHyp, pdIndex));

						// D*F2
						else					pK->block(startingRow, startingCol, m_nd, m_n) = ((Scalar) -1.f) * (*(K_FD(m_pSqDistList[1], m_pDeltaListList[1][row-1], logHyp, pdIndex)));

						// copy its transpose
						pK->block(startingCol, startingRow, m_n, m_nd).noalias() = pK->block(startingRow, startingCol, m_nd, m_n).transpose();
					}
				}

				if(m_n > 0)
				{
					const int startingRow = m_nd*(m_d+1);

					// F2F2
					pK->block(startingRow, startingRow, m_n, m_n) = *(K_FF(m_pSqDistList[2], logHyp, pdIndex));
				}

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

				// calculate the squared distances
				MatrixPtr pSqDist1	= crossSqDistances(m_pXd, pXs);		// XdXs
				MatrixPtr pSqDist2	= crossSqDistances(m_pX, pXs);			// XXs

				// calculate the delta
				MatrixPtrList deltaList;
				deltaList.resize(m_d);
				for(int i = 0; i < m_d; i++) deltaList[i] = crossDelta(m_pXd, pXs, i);		// XdXs

				// covariance matrix
				MatrixPtr pKs(new Matrix(n, m)); // (nd*(d+1) + n) x m

				// F1F
				pKs->block(0, 0, m_nd, m) = *(K_FF(pSqDist1, logHyp));

				// D1F, D2F, D3F
				for(int row = 1; row <= m_d; row++)
					pKs->block(m_nd*row, 0, m_nd, m) = ((Scalar) -1.f) * (*(K_FD(pSqDist1, deltaList[row-1], logHyp)));

				// F2F
				pKs->block(m_nd*(m_d+1), 0, m_n, m) = *(K_FF(pSqDist2, logHyp));

				return pKs;
			}

			// self-variance/covariance: inherited

		protected:
			// covariance matrix given pair-wise sqaured distances
			MatrixPtr K_FD(MatrixConstPtr pSqDist, MatrixConstPtr pDelta, const Hyp &logHyp, const int pdIndex = -1) const
			{
				// input
				// pSqDist (nxm): squared distances = r^2
				// pDelta (nxm): delta = x_i - x_i'
				// logHyp: log hyperparameters
				// pdIndex: partial derivatives with respect to this parameter index

				// output
				// K: nxm matrix
				// if pdIndex == -1:		K_FF
				// else							partial K_FF / partial theta_i

				assert(pdIndex < 2);

				MatrixPtr pK_FD = K_FF(pSqDist, logHyp, pdIndex);

				// hyperparameters
				Scalar inv_ell2 = exp(((Scalar) -2.f) * logHyp(0));

				// mode
				switch(pdIndex)
				{
				// derivatives of covariance matrix w.r.t log ell
				case 0:
					{
						// k_log(ell)	 = ((x_i - x_i') / ell^2) * (K_FF_log(ell) - 2K_FF)
						MatrixPtr pK_FF = K_FF(pSqDist, logHyp); // K_FF
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

				// covariance matrix
				default:
					{
						// k(X, X') = ((x_i - x_i') / ell^2) * K_FF(X, X')
						(*pK_FD) = inv_ell2 * pDelta->cwiseProduct(*pK_FD);
						//std::cout << "K_FD = " << std::endl << *pK_FD << std::endl << std::endl;
						break;
					}
				}

				return pK_FD;
			}

			// covariance matrix given pair-wise sqaured distances
			MatrixPtr K_DD(MatrixConstPtr pSqDist, 
										 MatrixConstPtr pDelta1, const int i, 
										 MatrixConstPtr pDelta2, const int j,
										 const Hyp &logHyp, const int pdIndex = -1) const
			{
				// input
				// pSqDist (nxm): squared distances = r^2
				// pDelta1 (nxm): delta = x_i - x_i'
				// i: index for delta1
				// pDelta2 (nxm): delta = x_j - x_j'
				// j: index for delta2
				// logHyp: log hyperparameters
				// pdIndex: partial derivatives with respect to this parameter index

				// output
				// K: nxm matrix
				// if pdIndex == -1:		K_FF
				// else							partial K_FF / partial theta_i

				assert(pdIndex < 2);

				MatrixPtr pK_DD = K_FF(pSqDist, logHyp, pdIndex);

				// hyperparameters
				Scalar inv_ell2				= exp(((Scalar) -2.f) * logHyp(0));
				Scalar inv_ell4				= exp(((Scalar) -4.f) * logHyp(0));
				Scalar neg2_inv_ell2	= ((Scalar) -2.f) * inv_ell2;
				Scalar four_inv_ell4		= ((Scalar) 4.f) * inv_ell4;

				// delta
				Scalar delta = (i == j) ? (Scalar) 1.f  : (Scalar) 0.f;

	#if 0
				// mode
				switch(pdIndex)
				{
				// derivatives of covariance matrix w.r.t log ell
				case 0:
					{
						// k_log(ell)	 = [ -2*delta / ell^2 + 4*((x_i - x_i')*(x_j - x_j') / ell^4) ] * K_FF(X, X')
						//                   + [ delta / ell^2 - ((x_i - x_i')*(x_j - x_j') / ell^4) ] * K_FF_log(ell)(X, X')
						MatrixPtr pK_FF = K_FF(pSqDist, logHyp); // K_FF
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

				// covariance matrix
				default:
					{
						// k(X, X') = [ delta / ell^2 - ((x_i - x_i')*(x_j - x_j') / ell^4) ] * K_FF(X, X')
						(*pK_DD) = (inv_ell2*delta - inv_ell4*(pDelta1->array())*(pDelta2->array())) * pK_DD->array();
						//std::cout << "K_DD = " << std::endl << *pK_DD << std::endl << std::endl;
						break;
					}
				}
	#else
				// simplified version

				// for all cases
				(*pK_DD) = (inv_ell2*delta - inv_ell4*(pDelta1->array())*(pDelta2->array())) * pK_DD->array();

				// particularly, derivatives of covariance matrix w.r.t log ell
				if(pdIndex == 0)
					(*pK_DD) += (neg2_inv_ell2*delta + four_inv_ell4*(pDelta1->array())*(pDelta2->array())).matrix().cwiseProduct(*(K_FF(pSqDist, logHyp)));
					//(*pK_DD) += (neg2_inv_ell2*delta + four_inv_ell4*(pDelta1->array())*(pDelta2->array())) * (K_FF(pSqDist, logHyp)->array());
	#endif

				return pK_DD;
			}

			// pre-calculate the squared distances and deltas
			void preCalculateSqDistAndDelta(MatrixConstPtr pXd, MatrixConstPtr pX)
			{
				//                              | Xd(nd) | X(n) |
				// sqDist = -----------------------------------
				//                 Xd (nd) |  XdXd,   XdX
				//                 X  (n)    |           -,      XX

				//                              | Xd(nd) | X(n) |
				// delta = -------------------------------------
				//                 Xd (nd) |  XdXd,   XdX

				// pre-calculate squared distances and delta(upper triangle)
				m_pSqDistList.resize(3);
				m_pDeltaListList.resize(2);

				// squared distances
				m_pSqDistList[0] = selfSqDistances(pXd);						// XdXd
				m_pSqDistList[1] = crossSqDistances(pXd, pX);			// XdX
				m_pSqDistList[2] = selfSqDistances(pX);							// XX

				// delta
				m_pDeltaListList[0].resize(m_d);		for(int i = 0; i < m_d; i++) m_pDeltaListList[0][i] = selfDelta(pXd, i);					// XdXd
				m_pDeltaListList[1].resize(m_d);		for(int i = 0; i < m_d; i++) m_pDeltaListList[1][i] = crossDelta(pXd, pX, i);	// XdX
			}

		protected:
			MatrixPtrList							m_pSqDistList;			// FF, FD, DD: squared distances
			MatrixPtrListList					m_pDeltaListList;		// FF, FD, DD: x_i - x_i'
	};

}

#endif