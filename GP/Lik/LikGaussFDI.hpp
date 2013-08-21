#ifndef LIKELIHOOD_FUNCTION_GAUSSIAN_BETWEEN_FUNCTION_VALUE_DERIVATIVE_AND_INTEGRAL_HPP
#define LIKELIHOOD_FUNCTION_GAUSSIAN_BETWEEN_FUNCTION_VALUE_DERIVATIVE_AND_INTEGRAL_HPP

#include <cmath>

#include "GP/util/TrainingInputSetterDerivatives.hpp"
#include "GP/Lik/LikGauss.hpp"

namespace GPOM{

	class LikGaussFDI : public LikGauss, public TrainingInputSetterDerivatives
	{
	public:
		// hyperparameters
		typedef	Eigen::Matrix<Scalar, 2, 1>							Hyp;						// s_nf, s_nd

		public:
			// constructor
			LikGaussFDI()
			{
			}

			// destructor
			virtual ~LikGaussFDI()
			{
			}

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

				return true;
			}

			// getter
			virtual int	getN() const { return m_nd*(m_d+1)+m_n; }

			// diagonal vector
			VectorPtr operator()(const Hyp &logHyp, const int pdIndex = -1) const
			{			
				assert(pdIndex < 2);

				// number of training data
				const int n = getN();
				VectorPtr pD(new Vector(n));

				// F1(nd), D1(nd), D2(nd), D3(nd), F2(n)
				switch(pdIndex)
				{
				// derivative w.r.t s_nf
				case 0:
					{
						pD->head(m_nd).fill(((Scalar) 2.f) * exp((Scalar) 2.f * logHyp(0)));				// F1
						pD->segment(m_nd, m_nd*m_d).setZero();													// D*
						pD->tail(m_n).fill(((Scalar) 2.f) * exp((Scalar) 2.f * logHyp(0)));					// F2
						break;
					}

				// derivative w.r.t s_nd
				case 1:
					{
						pD->head(m_nd).setZero();																										// F1
						pD->segment(m_nd, m_nd*m_d).fill(((Scalar) 2.f) * exp((Scalar) 2.f * logHyp(1)));			// D*
						pD->tail(m_n).setZero();																											// F2
						break;
					}

				// likelihood
				default:
					{
						pD->head(m_nd).fill(exp((Scalar) 2.f * logHyp(0)));								// F1
						pD->segment(m_nd, m_nd*m_d).fill(exp((Scalar) 2.f * logHyp(1)));	// D*
						pD->tail(m_n).fill(exp((Scalar) 2.f * logHyp(0)));										// F2
						break;
					}
				}

				return pD;
			}

			//// diagonal matrix
			//MatrixPtr operator()(MatrixConstPtr pX, const Hyp &logHyp, const int pdIndex = -1) const
			//{
			//	// number of training data
			//	const int n = getN();
			//	MatrixPtr pD(new Matrix(n, n));
			//	pD->setZero();

			//	// fill
			//	switch(pdIndex)
			//	{
			//	// derivative w.r.t s_nd
			//	case 0:
			//		{
			//			pD->block(0, 0, m_nd, m_nd).diagonal().fill(((Scalar) 2.f) * exp((Scalar) 2.f * logHyp(0)));
			//			break;
			//		}

			//	// derivative w.r.t s_nf
			//	case 0:
			//		{
			//			pD->block(m_nd, m_nd, m_n, m_n).diagonal().fill(((Scalar) 2.f) * exp((Scalar) 2.f * logHyp(1)));
			//			break;
			//		}			

			//	// likelihood
			//	default:
			//		{
			//			pD->block(0, 0, m_nd, m_nd).diagonal().fill(exp((Scalar) 2.f * logHyp(0)));
			//			pD->block(m_nd, m_nd, m_n, m_n).diagonal().fill(exp((Scalar) 2.f * logHyp(1)));
			//			break;
			//		}
			//	}

			//	return pD;
			//}
	};

}

#endif