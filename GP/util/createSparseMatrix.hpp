#ifndef CREATE_SPARSE_MATRIX_HPP
#define CREATE_SPARSE_MATRIX_HPP

#include <vector>
#include "GP/DataTypes.hpp"

namespace GPOM{

	SparseMatrixPtr createSparseMatrix(Matrix &dense, const Scalar cutLine, const bool fUpperTriangleOnly = true)
	{
		// matrix size
		const int n = dense.rows();
		const int m = dense.cols();

		// triple list
		std::vector< Eigen::Triplet<Scalar> > tripletList;
		tripletList.reserve(n*m);

		if(fUpperTriangleOnly)
		{
			for(int row = 0; row < n; row++)
				for(int col = row; col < m; col++)
					if(dense(row, col) > cutLine)		tripletList.push_back(Eigen::Triplet<Scalar>(row, col, dense(row, col)));
					else												dense(row, col) = (Scalar) 0.f;
		}
		else
		{
			for(int row = 0; row < n; row++)
				for(int col = 0; col < m; col++)
					if(dense(row, col) > cutLine)		tripletList.push_back(Eigen::Triplet<Scalar>(row, col, dense(row, col)));
					else												dense(row, col) = (Scalar) 0.f;
		}

		// initialize with the triple list
		SparseMatrixPtr pSparse(new SparseMatrix(n, m));
		pSparse->setFromTriplets(tripletList.begin(), tripletList.end());
		return pSparse;
	}

	SparseMatrixPtr createSparseMatrix(MatrixPtr pDense, const Scalar cutLine, const bool fUpperTriangleOnly = true)
	{
		return createSparseMatrix(*pDense, cutLine, fUpperTriangleOnly);
	}

	void makeSymmetricPositiveDefiniteMatrix(Matrix &dense)
	{
		// size
		assert(dense.rows() == dense.cols());
		const Scalar n = (Scalar) dense.rows();

		// symmetric matrix
		dense = ((Scalar) 0.5f) * (dense + dense.transpose());

		// diagonal
		dense.diagonal().array() += n;
	}
}

#endif