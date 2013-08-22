#include <iostream>
#include <vector>

//#include <boost/timer/timer.hpp> // boost timer, boost::timer::auto_cpu_timer t;
#include <boost/chrono.hpp>			// boost chrono

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

#include "GP/util/createSparseMatrix.hpp"
using namespace GPOM;

int main()
{
	// timer
	boost::chrono::system_clock::time_point		startTime, endTime;
	boost::chrono::duration<double>						durationTime;

	// matrix
	const int n = 3000;
	const float cutLine = -0.5f;
	Eigen::MatrixXf dense(n , n);

	//const int numTests = 10;
	//for(int i = 0; i < numTests; i++)
	//{
		// random dense matrix
		dense.setRandom();
		makeSymmetricPositiveDefiniteMatrix(dense);

		// random sparse matrix
#if 0
		Eigen::SparseMatrix<Scalar> sparse(n, n);
		startTime = boost::chrono::system_clock::now();

		for(int row = 0; row < n; row++)
			for(int col = 0; col < n; col++)
				if(dense(row, col) > cutLine) sparse.insert(row, col) = dense(row, col);

		endTime = boost::chrono::system_clock::now();
		durationTime = endTime - startTime;
		std::cout << "non-zeros: " << sparse.nonZeros() << " / " << n*n << " ( " << pSparse->nonZeros() * 100.f / ((float) n*n) << " % ) " << std::endl;
		std::cout << "time to initialize: " << durationTime.count() << " seconds" << std::endl;
#else
		startTime = boost::chrono::system_clock::now();

		SparseMatrixPtr pSparse = createSparseMatrix(dense, cutLine, false);

		endTime = boost::chrono::system_clock::now();
		durationTime = endTime - startTime;
		std::cout << "non-zeros: " << pSparse->nonZeros() << " / " << n*n << " ( " << pSparse->nonZeros() * 100.f / ((float) n*n) << " % ) " << std::endl;
		std::cout << "time to initialize: " << durationTime.count() << " seconds" << std::endl;
#endif

		// choleskey factorization

		// Dense - LLT
		Eigen::LLT<Matrix> denseCholLLT;
		startTime = boost::chrono::system_clock::now();
#if 1
		denseCholLLT.compute(dense);
#else
		denseCholLLT.compute(dense.selfadjointView<Eigen::Upper>());
#endif
		endTime = boost::chrono::system_clock::now();
		durationTime = endTime - startTime;
		std::cout << "Dense - LLT: " << durationTime.count() << " seconds" << std::endl;

		// Dense - LDLT
		Eigen::LDLT<Matrix> denseCholLDLT;
		startTime = boost::chrono::system_clock::now();
#if 1
		denseCholLDLT.compute(dense);
#else
		denseCholLDLT.compute(dense.selfadjointView<Eigen::Upper>());
#endif
		endTime = boost::chrono::system_clock::now();
		durationTime = endTime - startTime;
		std::cout << "Dense - LDLT: " << durationTime.count() << " seconds" << std::endl;

		// Sparse - LLT
		Eigen::SimplicialLLT<SparseMatrix, Eigen::Upper> sparseCholLLT;
		startTime = boost::chrono::system_clock::now();
		sparseCholLLT.compute(*pSparse);
		endTime = boost::chrono::system_clock::now();
		durationTime = endTime - startTime;
		std::cout << "Sparse - SimplicialLLT: " << durationTime.count() << " seconds" << std::endl;

		// Sparse - LDLT
		Eigen::SimplicialLDLT<SparseMatrix, Eigen::Upper> sparseCholLDLT;
		startTime = boost::chrono::system_clock::now();
		sparseCholLDLT.compute(*pSparse);
		endTime = boost::chrono::system_clock::now();
		durationTime = endTime - startTime;
		std::cout << "Sparse - SimplicialLDLT: " << durationTime.count() << " seconds" << std::endl;

		// Sparse - CG
		Eigen::ConjugateGradient<SparseMatrix, Eigen::Upper> sparseCholCG;
		startTime = boost::chrono::system_clock::now();
		sparseCholCG.compute(*pSparse);
		endTime = boost::chrono::system_clock::now();
		durationTime = endTime - startTime;
		std::cout << "Sparse - CG: " << durationTime.count() << " seconds" << std::endl;

		// Dense - CG
		Eigen::ConjugateGradient<Matrix, Eigen::Upper> denseCholCG;
		startTime = boost::chrono::system_clock::now();
		denseCholCG.compute(dense);
		endTime = boost::chrono::system_clock::now();
		durationTime = endTime - startTime;
		std::cout << "Dense - CG: " << durationTime.count() << " seconds" << std::endl;

		//std::cout << "dense = " << std::endl << dense << std::endl;
		//std::cout << "sparse = " << std::endl << *pSparse << std::endl;

		// 
		//sparse.prune(cutLine);
		//std::cout << "non-zeros: " << sparse.nonZeros() << std::endl;
	//}

	return 0;
}