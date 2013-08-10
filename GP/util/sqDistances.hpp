#ifndef SQUARED_DISTANCES_HPP
#define SQUARED_DISTANCES_HPP

#include <assert.h>

#include "DataTypes.hpp"

namespace GP{
	// self
	MatrixPtr selfSqDistances(MatrixConstPtr pX)
	{
		// pX: nxd

		// number of training data
		const int n = pX->rows();

		// squared distances: nxn
		MatrixPtr pSqDist(new Matrix(n, n));

		// initialization: diagonal is always zero
#if 0
		for(int row = 0; row < n; row++)
		{
			(*pSqDist)(row, row) = (Scalar) 0.f;
		}
#else
		(*pSqDist).setZero();
#endif

		// upper triangle
		for(int row = 0; row < n; row++)
		{
			for(int col = row + 1; col < n; col++)
			{
				(*pSqDist)(row, col) = (pX->row(row) - pX->row(col)).array().square().sum();
				//(*pSqDist)(row, col) = (pX->row(row) - pX->row(col)).squaredNorm();
			}
		}

		// lower triangle
#if 0
		for(int row = 0; row < n; row++)
		{
			for(int col = 0; col < row; col++)
			{
				(*pSqDist)(row, col) = (*pSqDist)(col, row);
			}
		}
#else
		pSqDist->triangularView<Eigen::StrictlyLower>() = pSqDist->transpose().eval().triangularView<Eigen::StrictlyLower>();
#endif

		return pSqDist;
	}

	// cross
	MatrixPtr crossSqDistances(MatrixConstPtr pX, MatrixConstPtr pXs)
	{
		// X: nxd
		// Xs: mxd
		const int n	= pX->rows();
		const int m	= pXs->rows();

		// check if the dimensions are same
		assert(pX->cols() == pXs->cols());

		// squared distances: nxn
		MatrixPtr pSqDist(new Matrix(n, m));

		// dense
		for(int row = 0; row < n; row++)
		{
			for(int col = 0; col < m; col++)
			{
				(*pSqDist)(row, col) = (pX->row(row) - pXs->row(col)).array().square().sum();
				//(*pSqDist)(row, col) = (pX->row(row) - pX->row(col)).squaredNorm();
			}
		}

		return pSqDist;
	}

	// self
	MatrixPtr selfDelta(MatrixConstPtr pX, const int i)
	{
		// [input]
		// X: mxd
		// i: coordinate index

		// [output]
		// x_i - x_i'

		assert(i < pX->cols());

		// number of training data
		const int n = pX->rows();

		// squared distances: nxn
		MatrixPtr pDelta(new Matrix(n, n));

		// initialization: diagonal is always zero
		(*pDelta).setZero();

		// upper triangle
		for(int row = 0; row < n; row++)
			for(int col = row + 1; col < n; col++)
				(*pDelta)(row, col) = (*pX)(row, i) - (*pX)(col, i);

		// lower triangle (CAUTION: not symmetric but skew symmetric)
		pDelta->triangularView<Eigen::StrictlyLower>() = (((Scalar) -1.f) * (*pDelta)).transpose().eval().triangularView<Eigen::StrictlyLower>();

		return pDelta;
	}

	// cross
	MatrixPtr crossDelta(MatrixConstPtr pX, MatrixConstPtr pXs, const int i)
	{
		// [input]
		// X: nxd
		// Xs: mxd
		// i: coordinate index

		// [output]
		// x_i - x_i'

		// check if the dimensions are same
		assert(pX->cols() == pXs->cols());
		assert(i < pX->cols());

		// number of data
		const int n	= pX->rows();
		const int m	= pXs->rows();

		// squared distances: nxn
		MatrixPtr pDelta(new Matrix(n, m));

		// dense
		for(int row = 0; row < n; row++)
			for(int col = 0; col < m; col++)
				(*pDelta)(row, col) = (*pX)(row, i) - (*pXs)(col, i);

		return pDelta;
	}
}

#endif