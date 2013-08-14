#if 1
#define BOOST_TEST_MODULE Simple testcases
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <cmath>

#include "GP/DataTypes.hpp"
#include "GP/util/sqDistances.hpp"
#include "GP/Mean/MeanZero.hpp"
#include "GP/Mean/MeanZeroFDI.hpp"
#include "GP/Cov/CovSEiso.hpp"
#include "GP/Cov/CovMaterniso.hpp"
#include "GP/Cov/CovSEisoFDI.hpp"
#include "GP/Cov/CovMaternisoFDI.hpp"
#include "GP/Lik/LikGauss.hpp"
#include "GP/Lik/LikGaussFDI.hpp"
#include "GP/Inf/InfExact.hpp"
#include "GP/Inf/InfExactFDI.hpp"
//#include "InfExactUnstable.hpp"
#include "GP/GP.hpp"

#include "GPOM.hpp"

using namespace GPOM;


// epsilon
//Scalar epsilon = 1E-6f;
Scalar epsilon = 1E5f;

// n, m, d
const int n = 4;
const int m = 7;
const int d = 3;
const int n1 = 2;
const int n2 = 2;

// X, Xs
Matrix X(n, d);
Matrix Xs(m, d);
MatrixPtr pX, pXs, pXd;
VectorPtr pY;

// hyp
CovSEiso::HypPtr pLogHyp;

// covariance
MatrixPtr pK, pKs, pKss;

// covariance matrix objects
CovSEiso						covSEiso;
CovMaterniso3				covMaterniso3;
CovSEisoFDI				covSEisoFDI;
CovMaterniso3FDI		covMaterniso3FDI;

// GP
typedef GaussianProcess<MeanZero,			CovSEiso,					LikGauss,			InfExact>				GPCovSEiso;
typedef GaussianProcess<MeanZero,			CovMaterniso3,			LikGauss,			InfExact>				GPCovMaterniso3;
typedef GaussianProcess<MeanZeroFDI,	CovSEisoFDI,			LikGaussFDI,		InfExactFDI>		GPCovSEisoFDI;
typedef GaussianProcess<MeanZeroFDI,	CovMaterniso3FDI,	LikGaussFDI,		InfExactFDI>		GPCovMaterniso3FDI;
GPCovSEiso						gpCovSEiso;
GPCovMaterniso3			gpCovMaterniso3;
GPCovSEisoFDI				gpCovSEisoFDI;
GPCovMaterniso3FDI		gpCovMaterniso3FDI;

// hyperparameters
GPCovSEiso::MeanHypPtr				pMeanLogHyp;
GPCovSEiso::CovHypPtr				pCovLogHyp;
GPCovSEiso::LikHypPtr					pLikLogHyp;
GPCovSEisoFDI::MeanHypPtr		pMeanLogHypFDI;
GPCovSEisoFDI::CovHypPtr			pCovLogHypFDI;
GPCovSEisoFDI::LikHypPtr			pLikLogHypFDI;

// nlZ, dnlZ
Scalar nlZ;
VectorPtr pDnlZ;

// GPOM
typedef GaussianProcess<MeanZeroFDI, CovMaterniso3FDI, LikGaussFDI, InfExactFDI>		GPOMType;
GPOMType gpom;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
BOOST_AUTO_TEST_SUITE(suite_covariance)

// TEST1: self_squared_distance
BOOST_AUTO_TEST_CASE(self_squared_distance) {

	// X
	X << 	2,     2,     2,
		        8,     9,     6,
				2,     3,     4,
				8,     1 ,    3;
	if(PointMatrixDirection::fRowWisePointsMatrix)
	{
		pX.reset(new Matrix (n, d));
		(*pX) = X;
	}
	else
	{
		X.transposeInPlace();
		pX.reset(new Matrix (d, n));
		(*pX) = X;
	}

	// sqDist = (X - X)^2
	Matrix sqDist_(n, n);
	sqDist_ << 0,   101,      5,    38,
					    101,       0,    76,    73,
					     5,     76,      0,    41,
					     38,     73,    41,      0;

	// square distances
	MatrixPtr pSqDist = selfSqDistances(pX);

	// check
    BOOST_CHECK_EQUAL(sqDist_, (*pSqDist));
}

// TEST2: cross_squared_distance
BOOST_AUTO_TEST_CASE(cross_squared_distance) {
	// Xs
	Xs << 8,     3,     1,
			   5,     5,     5,
			   5,     0,     4,
			   9,     0,     0,
			   2,     5,     3,
			   7,     7,     1,
			   7,     9,     7;

	if(PointMatrixDirection::fRowWisePointsMatrix)
	{
		pXs.reset(new Matrix (m, d));
		(*pXs) = Xs;
	}
	else
	{
		Xs.transposeInPlace();
		pXs.reset(new Matrix (d, m));
		(*pXs) = Xs;
	}


	// sqDists = (X - Xs)^2
	Matrix sqDist_(n, m);
	sqDist_ <<   38,    27,    17,    57,    10,    51,    99,
					      61,    26,    94,   118,    61,    30,     2,
						  45,    14,    18,    74,     5,    50,    70,
						  8,    29,    11,    11,    52,    41,    81;

	// square distances
	MatrixPtr pSqDist = crossSqDistances(pX, pXs);

	// check
    BOOST_CHECK_EQUAL(sqDist_, (*pSqDist));
}


// TEST3: covSEiso: K
BOOST_AUTO_TEST_CASE(covSEiso_K) {
	// K
	Matrix K_(n, n);
	K_ << 6.250000000000000,   0.000000001117845,   2.057456173799411,   0.001344325362889,
			   0.000000001117845,   6.250000000000000,   0.000000289153709,   0.000000563194522,
			   2.057456173799411,   0.000000289153709,   6.250000000000000,   0.000690199654857,
			   0.001344325362889,   0.000000563194522,   0.000690199654857,   6.250000000000000;

	// hyp
	pLogHyp.reset(new CovSEiso::Hyp());
	(*pLogHyp)(0) = log(1.5f);
	(*pLogHyp)(1) = log(2.5f);

	// covSEiso
	covSEiso.setTrainingInputs(pX);
	pK = covSEiso.K(pLogHyp);
	//pK->triangularView<Eigen::StrictlyLower>() = pK->transpose().eval().triangularView<Eigen::StrictlyLower>();

	// check
    BOOST_CHECK_EQUAL(((K_ - (*pK)).array().abs() < epsilon).all(), true);
}


// TEST4: covSEiso: partial derivatives of K with respect to log ell
BOOST_AUTO_TEST_CASE(covSEiso_K1) {
	// K
	Matrix K_(n, n);
	K_ <<                   0,   0.000000050178827,   4.572124830665357,   0.022704161684350,
				   0.000000050178827,                   0,   0.000009766969727,   0.000018272533385,
				   4.572124830665357,   0.000009766969727,                   0,   0.012576971488505,
				   0.022704161684350,   0.000018272533385,   0.012576971488505,                   0;

	// covSEiso
	pK = covSEiso.K(pLogHyp, 0);
	//pK->triangularView<Eigen::StrictlyLower>() = pK->transpose().eval().triangularView<Eigen::StrictlyLower>();

	// check
    BOOST_CHECK_EQUAL(((K_ - (*pK)).array().abs() < epsilon).all(), true);
}


// TEST5: covSEiso: partial derivatives of K with respect to log sigma_f
BOOST_AUTO_TEST_CASE(covSEiso_K2) {
	// K
	Matrix K_(n, n);
	K_ <<  12.500000000000000,   0.000000002235690,   4.114912347598822,   0.002688650725778,
				 0.000000002235690,  12.500000000000000,   0.000000578307418,   0.000001126389044,
				 4.114912347598822,   0.000000578307418,  12.500000000000000,   0.001380399309714,
				 0.002688650725778,   0.000001126389044,   0.001380399309714,  12.500000000000000;

	// covSEiso
	pK = covSEiso.K(pLogHyp, 1);
	//pK->triangularView<Eigen::StrictlyLower>() = pK->transpose().eval().triangularView<Eigen::StrictlyLower>();

	// check
    BOOST_CHECK_EQUAL(((K_ - (*pK)).array().abs() < epsilon).all(), true);
}


// TEST6: covSEiso: Ks
BOOST_AUTO_TEST_CASE(covSEiso_Ks) {
	// Ks
	Matrix Ks_(n, m);
	Ks_ << 0.001344325362889,   0.015492201104165,   0.142959155695243,   0.000019715898782,   0.677300145136849,   0.000074795572225,   0.000000001743418,
				  0.000008105448308,   0.019347417827283,   0.000000005296035,   0.000000000025569,   0.000008105448308,   0.007953961258374,   4.007377427687215,
				  0.000283749561016,   0.278446414028060,   0.114472743054588,   0.000000450970919,   2.057456173799409,   0.000093408365780,   0.000001096953142,
				  1.056333221287912,   0.009933295521603,   0.542339559212058,   0.542339559212058,   0.000059891612253,   0.000690199654857,   0.000000095187373;

	// covSEiso
	pKs = covSEiso.Ks(pXs, pLogHyp);

	// check
    BOOST_CHECK_EQUAL(((Ks_ - (*pKs)).array().abs() < epsilon).all(), true);
}


// TEST7: covSEiso: Kss
BOOST_AUTO_TEST_CASE(covSEiso_Kss) {
	// Ks
	Matrix Kss_(m, 1);
	Kss_ << 6.250000000000000,
				   6.250000000000000,
				   6.250000000000000,
				   6.250000000000000,
				   6.250000000000000,
				   6.250000000000000,
				   6.250000000000000;

	// covSEiso
	pKss = covSEiso.Kss(pXs, pLogHyp);

	// check
    BOOST_CHECK_EQUAL(((Kss_ - (*pKss)).array().abs() < epsilon).all(), true);
}

// TEST8: CovMaterniso3: K
BOOST_AUTO_TEST_CASE(covMaterniso3_K) {
	// K
	Matrix K_(n, n);
	K_ <<
   6.250000000000000,   0.000718781058796,   1.693014673677355,   0.041115074044017,
   0.000718781058796,   6.250000000000000,   0.002938233259377,   0.003526084701227,
   1.693014673677355,   0.002938233259377,   6.250000000000000,   0.032269615509369,
   0.041115074044017,   0.003526084701227,   0.032269615509369,   6.250000000000000;

	// covMaterniso3
	covMaterniso3.setTrainingInputs(pX);
	pK = covMaterniso3.K(pLogHyp);

	// check
    BOOST_CHECK_EQUAL(((K_ - (*pK)).array().abs() < epsilon).all(), true);
}


// TEST9: covMaterniso3: partial derivatives of K with respect to log ell
BOOST_AUTO_TEST_CASE(covMaterniso3_K1) {
	// K
	Matrix K_(n, n);
	K_ <<
                   0,   0.007679408620615,   3.150976961193057,   0.256608815569039,
   0.007679408620615,                   0,   0.026904841229566,   0.031585954113480,
   3.150976961193057,   0.026904841229566,                   0,   0.210166458786028,
   0.256608815569039,   0.031585954113480,   0.210166458786028,                   0;

	// covMaterniso3
	pK = covMaterniso3.K(pLogHyp, 0);

	// check
    BOOST_CHECK_EQUAL(((K_ - (*pK)).array().abs() < epsilon).all(), true);
}


// TEST10: covMaterniso3: partial derivatives of K with respect to log sigma_f
BOOST_AUTO_TEST_CASE(covMaterniso3_K2) {
	// K
	Matrix K_(n, n);
	K_ <<
  12.500000000000000,   0.001437562117592,   3.386029347354711,   0.082230148088033,
   0.001437562117592,  12.500000000000000,   0.005876466518754,   0.007052169402453,
   3.386029347354711,   0.005876466518754,  12.500000000000000,   0.064539231018739,
   0.082230148088033,   0.007052169402453,   0.064539231018739,  12.500000000000000,

	// covMaterniso3
	pK = covMaterniso3.K(pLogHyp, 1);

	// check
    BOOST_CHECK_EQUAL(((K_ - (*pK)).array().abs() < epsilon).all(), true);
}


// TEST11: covMaterniso3: Ks
BOOST_AUTO_TEST_CASE(covMaterniso3_Ks) {
	// Ks
	Matrix Ks_(n, m);
	Ks_ <<
   0.041115074044017,   0.108445407729153,   0.308119361567033,   0.009939312900957,   0.754487924615610,   0.015155134185115,   0.000799370879559,
   0.007585743053829,   0.119373088616289,   0.001047226386691,   0.000302089229212,   0.007585743053829,   0.082024162486835,   3.214621384099057,
   0.023640531963645,   0.442043839782440,   0.274825575237353,   0.003316874388471,   1.693014673677356,   0.016293383157891,   0.004246110669769,
   1.017418517656050,   0.089899911429187,   0.655510558942211,   0.655510558942210,   0.014105293314179,   0.032269615509369,   0.002183589344730;

	// covMaterniso3
	pKs = covMaterniso3.Ks(pXs, pLogHyp);

	// check
    BOOST_CHECK_EQUAL(((Ks_ - (*pKs)).array().abs() < epsilon).all(), true);
}


// TEST12: covMaterniso3: Kss
BOOST_AUTO_TEST_CASE(covMaterniso3_Kss) {
	// Ks
	Matrix Kss_(m, 1);
	Kss_ << 6.250000000000000,
				   6.250000000000000,
				   6.250000000000000,
				   6.250000000000000,
				   6.250000000000000,
				   6.250000000000000,
				   6.250000000000000;

	// covMaterniso3
	pKss = covMaterniso3.Kss(pXs, pLogHyp);

	// check
    BOOST_CHECK_EQUAL(((Kss_ - (*pKss)).array().abs() < epsilon).all(), true);
}
BOOST_AUTO_TEST_SUITE_END()


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
BOOST_AUTO_TEST_SUITE(suite_gp)

// TEST1: CovSEiso predict
BOOST_AUTO_TEST_CASE(covSEiso_predict) {
	// mu
	Vector Mu(7);
	Mu <<    1.166388129663797,
				   0.154564253186609,
				   0.653392529498473,
				   0.598769534186285,
				   0.973426313158394,
				   0.005824671062165,
				   2.528313977728111;

	// variance
	Matrix Variance(7, 1);
	Variance <<      6.073999815134176,
							   6.236708155251939,
							   6.199571963620151,
							   6.203606907346916,
							   5.582298224304473,
							   6.249989944355704,
							   3.717023052375709;

	// hyperparameters
	pMeanLogHyp.reset(new GPCovSEiso::MeanHyp());
	pCovLogHyp.reset(new GPCovSEiso::CovHyp());
	pLikLogHyp.reset(new GPCovSEiso::LikHyp());	
	(*pCovLogHyp)(0) = log(1.5f);
	(*pCovLogHyp)(1) = log(2.5f);
	(*pLikLogHyp)(0)	= log(0.3f);

	// Y
	if(PointMatrixDirection::fRowWisePointsMatrix)	pY.reset(new Vector(pX->rows()));
	else																				pY.reset(new Vector(pX->cols()));
	(*pY) << 1, 4, 3, 7;

	// predict 
	VectorPtr pMu;
	MatrixPtr pVariance;
	gpCovSEiso.setTrainingData(pX, pY);
	gpCovSEiso.predict(pMeanLogHyp, pCovLogHyp, pLikLogHyp, pXs, 
										pMu, pVariance);


	// check
    BOOST_CHECK_EQUAL(((Mu - (*pMu)).array().abs() < epsilon).all(), true);
    BOOST_CHECK_EQUAL(((Variance - (*pVariance)).array().abs() < epsilon).all(), true);
}

// TEST2: CovMaterniso3 predict
BOOST_AUTO_TEST_CASE(covMaterniso3_predict) {
	// mu
	Vector Mu(7);
	Mu <<    
   1.137202789120280,
   0.380512134332986,
   0.857454698800947,
   0.723859070225765,
   0.819818684285977,
   0.095131637329176,
   2.029867734890605;


	// variance
	Matrix Variance(7, 1);
	Variance <<
   6.086518434128104,
   6.215712362847178,
   6.161506203378075,
   6.182219400070902,
   5.782362681655123,
   6.248714952570278,
   4.620063330443635;

	// hyperparameters
	pMeanLogHyp.reset(new GPCovMaterniso3::MeanHyp());
	pCovLogHyp.reset(new GPCovMaterniso3::CovHyp());
	pLikLogHyp.reset(new GPCovMaterniso3::LikHyp());	
	(*pCovLogHyp)(0) = log(1.5f);
	(*pCovLogHyp)(1) = log(2.5f);
	(*pLikLogHyp)(0)	= log(0.3f);

	// predict 
	VectorPtr pMu;
	MatrixPtr pVariance;
	gpCovMaterniso3.setTrainingData(pX, pY);
	gpCovMaterniso3.predict(pMeanLogHyp, pCovLogHyp, pLikLogHyp, pXs, 
												 pMu, pVariance);


	// check
    BOOST_CHECK_EQUAL(((Mu - (*pMu)).array().abs() < epsilon).all(), true);
    BOOST_CHECK_EQUAL(((Variance - (*pVariance)).array().abs() < epsilon).all(), true);
}

// TEST3: nlZ, dnlZ
BOOST_AUTO_TEST_CASE(covSEiso_nlZ_dnlZ) {
	// mu
	Scalar nlZ_ =  13.149528003766861;

	// variance
	Vector dnlZ(3);
	dnlZ <<  -0.277834368243279,
				   -7.565890184811604,
				   -0.105424058454826;

	// nlZ, dnlZ
	gpCovSEiso.negativeLogMarginalLikelihood(pMeanLogHyp, pCovLogHyp, pLikLogHyp, 
																				   nlZ, 
																				   pDnlZ);

	//std::cout << "nlZ_ = " << std::endl << nlZ_ << std::endl << std::endl;
	//std::cout << "nlZ = " << std::endl << nlZ << std::endl << std::endl;
	//std::cout << "dnlZ_ = " << std::endl << dnlZ << std::endl << std::endl;
	//std::cout << "dnlZ = " << std::endl << *pDnlZ << std::endl << std::endl;

	// check
	BOOST_CHECK_EQUAL(std::abs(nlZ_ - nlZ) < epsilon, true);
    BOOST_CHECK_EQUAL(((dnlZ - (*pDnlZ)).array().abs() < epsilon).all(), true);
}

// TEST4: nlZ, dnlZ
BOOST_AUTO_TEST_CASE(covMaterniso3_nlZ_dnlZ) {
	// mu
	Scalar nlZ_ =   13.150661157847448;

	// variance
	Vector dnlZ(3);
	dnlZ <<  -0.327512260108195,
				  -7.531317952672277,
				  -0.105014705653317;

	// nlZ, dnlZ
	gpCovMaterniso3.negativeLogMarginalLikelihood(pMeanLogHyp, pCovLogHyp, pLikLogHyp, 
																							nlZ, 
																							pDnlZ);

	// check
	BOOST_CHECK_EQUAL(std::abs(nlZ_ - nlZ) < epsilon, true);
    BOOST_CHECK_EQUAL(((dnlZ - (*pDnlZ)).array().abs() < epsilon).all(), true);
}

// TEST5: covSEiso train hyperparameters
BOOST_AUTO_TEST_CASE(covSEiso_train) {
	// initial hyperparameters
	(*pCovLogHyp)(0) = log(1.5f);
	(*pCovLogHyp)(1) = log(2.5f);
	(*pLikLogHyp)(0)	= log(0.3f);

	// trained hyperparameters
	GPCovSEiso::CovHyp			covLogHyp;	covLogHyp << 2.052422630033926, 1.440277497666226;
	GPCovSEiso::LikHyp			likLogHyp;		likLogHyp << 0.341039567937783;

	// nlZ
	Scalar nlZ_ = 10.478356782033270;

	// train
	gpCovSEiso.train<BFGS, DeltaFunc>(pMeanLogHyp, pCovLogHyp, pLikLogHyp);
	gpCovSEiso.negativeLogMarginalLikelihood(pMeanLogHyp, pCovLogHyp, pLikLogHyp, 
																				   nlZ, 
																				   pDnlZ);

	// check
	std::cout << "covHyp = " << std::endl << covLogHyp.array().exp() << std::endl << std::endl;
	std::cout << "pCovHyp = " << std::endl << pCovLogHyp->array().exp() << std::endl << std::endl;
	std::cout << "likHyp = " << std::endl << likLogHyp.array().exp() << std::endl << std::endl;
	std::cout << "pLikHyp = " << std::endl << pLikLogHyp->array().exp() << std::endl << std::endl;
	std::cout << "nlZ_ = " << std::endl << nlZ_ << std::endl << std::endl;
	std::cout << "nlZ = " << std::endl << nlZ << std::endl << std::endl;
    BOOST_CHECK_EQUAL(((covLogHyp - (*pCovLogHyp)).array().abs() < epsilon).all(), true);
    BOOST_CHECK_EQUAL(((likLogHyp - (*pLikLogHyp)).array().abs() < epsilon).all(), true);
	BOOST_CHECK_EQUAL(std::abs(nlZ_ - nlZ) < epsilon, true);
}

// TEST6: covMaterniso3 train hyperparameters
BOOST_AUTO_TEST_CASE(covMaterniso3_train) {
	// initial hyperparameters
	(*pCovLogHyp)(0) = log(1.5f);
	(*pCovLogHyp)(1) = log(2.5f);
	(*pLikLogHyp)(0)	= log(0.3f);

	// trained hyperparameters
	GPCovSEiso::CovHyp			covLogHyp;	covLogHyp << 2.312806781816551, 1.454245754217935;
	GPCovSEiso::LikHyp			likLogHyp;		likLogHyp << 0.225717246667746;

	// nlZ
	Scalar nlZ_ = 10.449405846170231;

	// train
	gpCovMaterniso3.train<BFGS, DeltaFunc>(pMeanLogHyp, pCovLogHyp, pLikLogHyp);
	gpCovMaterniso3.negativeLogMarginalLikelihood(pMeanLogHyp, pCovLogHyp, pLikLogHyp, 
																							nlZ, 
																							pDnlZ);

	// check
	std::cout << "covHyp = " << std::endl << covLogHyp.array().exp() << std::endl << std::endl;
	std::cout << "pCovHyp = " << std::endl << pCovLogHyp->array().exp() << std::endl << std::endl;
	std::cout << "likHyp = " << std::endl << likLogHyp.array().exp() << std::endl << std::endl;
	std::cout << "pLikHyp = " << std::endl << pLikLogHyp->array().exp() << std::endl << std::endl;
	std::cout << "nlZ_ = " << std::endl << nlZ_ << std::endl << std::endl;
	std::cout << "nlZ = " << std::endl << nlZ << std::endl << std::endl;
    BOOST_CHECK_EQUAL(((covLogHyp - (*pCovLogHyp)).array().abs() < epsilon).all(), true);
    BOOST_CHECK_EQUAL(((likLogHyp - (*pLikLogHyp)).array().abs() < epsilon).all(), true);
	BOOST_CHECK_EQUAL(std::abs(nlZ_ - nlZ) < epsilon, true);
}

BOOST_AUTO_TEST_SUITE_END()


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
BOOST_AUTO_TEST_SUITE(suite_CovFDI)

// TEST1: CovSEisoFDI
BOOST_AUTO_TEST_CASE(CovSEisoFDI) {

	// X, Xd
	if(PointMatrixDirection::fRowWisePointsMatrix)
	{
		pXd.reset(new Matrix(X.topRows(n1)));
		pX.reset(new Matrix(X.bottomRows(n2)));
	}
	else	
	{
		pXd.reset(new Matrix(X.leftCols(n1)));
		pX.reset(new Matrix(X.rightCols(n2)));
	}

	// K
	Matrix K(n1*(d+1)+n2, n1*(d+1)+n2);

	// K_FDI
	K << 
6.250000000000000, 0.000000001117845, 0.000000000000000, -0.000000002980920, 0.000000000000000, -0.000000003477740, 0.000000000000000, -0.000000001987280, 2.057456173799408, 0.001344325362889, 
0.000000001117845, 6.250000000000000, 0.000000002980920, 0.000000000000000, 0.000000003477740, 0.000000000000000, 0.000000001987280, 0.000000000000000, 0.000000289153709, 0.000000563194522, 
0.000000000000000, 0.000000002980920, 2.777777777777778, -0.000000007452301, 0.000000000000000, -0.000000009273975, 0.000000000000000, -0.000000005299414, 0.000000000000000, 0.003584867634371, 
-0.000000002980920, 0.000000000000000, -0.000000007452301, 2.777777777777778, -0.000000009273975, 0.000000000000000, -0.000000005299414, 0.000000000000000, -0.000000771076557, 0.000000000000000, 
0.000000000000000, 0.000000003477740, 0.000000000000000, -0.000000009273975, 2.777777777777778, -0.000000010322817, 0.000000000000000, -0.000000006182650, 0.914424966133070, -0.000597477939062, 
-0.000000003477740, 0.000000000000000, -0.000000009273975, 0.000000000000000, -0.000000010322817, 2.777777777777778, -0.000000006182650, 0.000000000000000, -0.000000771076557, -0.000002002469412, 
0.000000000000000, 0.000000001987280, 0.000000000000000, -0.000000005299414, 0.000000000000000, -0.000000006182650, 2.777777777777778, -0.000000003036123, 1.828849932266140, 0.000597477939062, 
-0.000000001987280, 0.000000000000000, -0.000000005299414, 0.000000000000000, -0.000000006182650, 0.000000000000000, -0.000000003036123, 2.777777777777778, -0.000000257025519, -0.000000750926030, 
2.057456173799408, 0.000000289153709, 0.000000000000000, -0.000000771076557, 0.914424966133070, -0.000000771076557, 1.828849932266140, -0.000000257025519, 6.250000000000000, 0.000690199654857, 
0.001344325362889, 0.000000563194522, 0.003584867634371, 0.000000000000000, -0.000597477939062, -0.000002002469412, 0.000597477939062, -0.000000750926030, 0.000690199654857, 6.250000000000000;

	// CovSEisoFDI
	covSEisoFDI.setTrainingInputs(pXd, pX);
	pK = covSEisoFDI.K(pLogHyp);

	// check
    BOOST_CHECK_EQUAL(((K - (*pK)).array().abs() < epsilon).all(), true);

	// K_FDI_log(ell)
	K << 
0.000000000000000, 0.000000050178827, -0.000000000000000, -0.000000127848363, -0.000000000000000, -0.000000149156424, -0.000000000000000, -0.000000085232242, 4.572124830665355, 0.022704161684350, 
0.000000050178827, 0.000000000000000, 0.000000127848363, -0.000000000000000, 0.000000149156424, -0.000000000000000, 0.000000085232242, -0.000000000000000, 0.000009766969727, 0.000018272533385, 
-0.000000000000000, 0.000000127848363, -5.555555555555555, -0.000000303722667, 0.000000000000000, -0.000000379202515, 0.000000000000000, -0.000000216687151, 0.000000000000000, 0.053374695889525, 
-0.000000127848363, -0.000000000000000, -0.000000303722667, -5.555555555555555, -0.000000379202515, 0.000000000000000, -0.000000216687151, 0.000000000000000, -0.000024503099489, 0.000000000000000, 
-0.000000000000000, 0.000000149156424, 0.000000000000000, -0.000000379202515, -5.555555555555555, -0.000000421094874, 0.000000000000000, -0.000000252801677, 0.203205548029573, -0.008895782648254, 
-0.000000149156424, -0.000000000000000, -0.000000379202515, 0.000000000000000, -0.000000421094874, -5.555555555555555, -0.000000252801677, 0.000000000000000, -0.000024503099489, -0.000060964068766, 
-0.000000000000000, 0.000000085232242, 0.000000000000000, -0.000000216687151, 0.000000000000000, -0.000000252801677, -5.555555555555555, -0.000000123150040, 0.406411096059146, 0.008895782648254, 
-0.000000085232242, -0.000000000000000, -0.000000216687151, 0.000000000000000, -0.000000252801677, 0.000000000000000, -0.000000123150040, -5.555555555555555, -0.000008167699830, -0.000022861525787, 
4.572124830665355, 0.000009766969727, 0.000000000000000, -0.000024503099489, 0.203205548029573, -0.000024503099489, 0.406411096059146, -0.000008167699830, 0.000000000000000, 0.012576971488505, 
0.022704161684350, 0.000018272533385, 0.053374695889525, 0.000000000000000, -0.008895782648254, -0.000060964068766, 0.008895782648254, -0.000022861525787, 0.012576971488505, 0.000000000000000;

	// partial CovSEisoFDI w.r.t log(ell)
	pK = covSEisoFDI.K(pLogHyp, 0);

	// check
    BOOST_CHECK_EQUAL(((K - (*pK)).array().abs() < epsilon).all(), true);

	// K_FDI_log(sigma_f)
	K << 
12.500000000000000, 0.000000002235690, 0.000000000000000, -0.000000005961841, 0.000000000000000, -0.000000006955481, 0.000000000000000, -0.000000003974561, 4.114912347598816, 0.002688650725778, 
0.000000002235690, 12.500000000000000, 0.000000005961841, 0.000000000000000, 0.000000006955481, 0.000000000000000, 0.000000003974561, 0.000000000000000, 0.000000578307418, 0.000001126389044, 
0.000000000000000, 0.000000005961841, 5.555555555555555, -0.000000014904602, 0.000000000000000, -0.000000018547949, 0.000000000000000, -0.000000010598828, 0.000000000000000, 0.007169735268742, 
-0.000000005961841, 0.000000000000000, -0.000000014904602, 5.555555555555555, -0.000000018547949, 0.000000000000000, -0.000000010598828, 0.000000000000000, -0.000001542153115, 0.000000000000000, 
0.000000000000000, 0.000000006955481, 0.000000000000000, -0.000000018547949, 5.555555555555555, -0.000000020645634, 0.000000000000000, -0.000000012365299, 1.828849932266140, -0.001194955878124, 
-0.000000006955481, 0.000000000000000, -0.000000018547949, 0.000000000000000, -0.000000020645634, 5.555555555555555, -0.000000012365299, 0.000000000000000, -0.000001542153115, -0.000004004938824, 
0.000000000000000, 0.000000003974561, 0.000000000000000, -0.000000010598828, 0.000000000000000, -0.000000012365299, 5.555555555555555, -0.000000006072245, 3.657699864532281, 0.001194955878124, 
-0.000000003974561, 0.000000000000000, -0.000000010598828, 0.000000000000000, -0.000000012365299, 0.000000000000000, -0.000000006072245, 5.555555555555555, -0.000000514051038, -0.000001501852059, 
4.114912347598816, 0.000000578307418, 0.000000000000000, -0.000001542153115, 1.828849932266140, -0.000001542153115, 3.657699864532281, -0.000000514051038, 12.500000000000000, 0.001380399309714, 
0.002688650725778, 0.000001126389044, 0.007169735268742, 0.000000000000000, -0.001194955878124, -0.000004004938824, 0.001194955878124, -0.000001501852059, 0.001380399309714, 12.500000000000000;

	// partial CovSEisoFDI w.r.t log(sigma_f)
	pK = covSEisoFDI.K(pLogHyp, 1);

	// check
    BOOST_CHECK_EQUAL(((K - (*pK)).array().abs() < epsilon).all(), true);

	// Ks_FDI
	Matrix K_(n1*(d+1)+n2, m);
	K_ << 
0.001344325362889, 0.015492201104165, 0.142959155695243, 0.000019715898782, 0.677300145136849, 0.000074795572225, 0.000000001743418, 
0.000008105448308, 0.019347417827283, 0.000000005296035, 0.000000000025569, 0.000008105448308, 0.007953961258374, 4.007377427687215, 
0.003584867634371, 0.020656268138886, 0.190612207593658, 0.000061338351767, 0.000000000000000, 0.000166212382723, 0.000000003874261, 
0.000000000000000, -0.025796557103044, -0.000000007061380, 0.000000000011364, -0.000021614528821, -0.003535093892611, -1.781056634527650, 
0.000597477939062, 0.020656268138886, -0.127074805062438, -0.000017525243362, 0.903066860182465, 0.000166212382723, 0.000000005423966, 
-0.000021614528821, -0.034395409470725, -0.000000021184140, -0.000000000102276, -0.000014409685881, -0.007070187785221, 0.000000000000000, 
-0.000597477939062, 0.020656268138886, 0.127074805062438, -0.000017525243362, 0.301022286727488, -0.000033242476545, 0.000000003874261, 
-0.000018012107351, -0.008598852367681, -0.000000004707587, -0.000000000068184, -0.000010807264410, -0.017675469463053, 1.781056634527652, 
0.000283749561016, 0.278446414028061, 0.114472743054589, 0.000000450970919, 2.057456173799412, 0.000093408365780, 0.000001096953142, 
1.056333221287913, 0.009933295521603, 0.542339559212059, 0.542339559212058, 0.000059891612253, 0.000690199654857, 0.000000095187373;

	// partial CovSEisoFDI w.r.t log(ell)
	pK = covSEisoFDI.Ks(pXs, pLogHyp);

	// check
    BOOST_CHECK_EQUAL(((K_ - (*pK)).array().abs() < epsilon).all(), true);
}

// TEST2: CovMaterniso3FDI
BOOST_AUTO_TEST_CASE(CovMaterniso3FDI) {

	// K
	Matrix K(n1*(d+1)+n2, n1*(d+1)+n2);

	// K_FDI
	K << 
6.250000000000000, 0.000718781058796, 0.000000000000000, -0.000456202492314, 0.000000000000000, -0.000532236241033, 0.000000000000000, -0.000304134994876, 1.693014673677355, 0.041115074044017, 
0.000718781058796, 6.250000000000000, 0.000456202492314, 0.000000000000000, 0.000532236241033, 0.000000000000000, 0.000304134994876, 0.000000000000000, 0.002938233259377, 0.003526084701227, 
0.000000000000000, 0.000456202492314, 8.333333333333334, -0.000238464032156, 0.000000000000000, -0.000366914077687, 0.000000000000000, -0.000209665187250, 0.000000000000000, 0.040517181405638, 
-0.000456202492314, 0.000000000000000, -0.000238464032156, 8.333333333333334, -0.000366914077687, 0.000000000000000, -0.000209665187250, 0.000000000000000, -0.002124066412860, 0.000000000000000, 
0.000000000000000, 0.000532236241033, 0.000000000000000, -0.000366914077687, 8.333333333333334, -0.000352032675249, 0.000000000000000, -0.000244609385125, 0.630195392238611, -0.006752863567606, 
-0.000532236241033, 0.000000000000000, -0.000366914077687, 0.000000000000000, -0.000352032675249, 8.333333333333334, -0.000244609385125, 0.000000000000000, -0.002124066412860, -0.003461474423395, 
0.000000000000000, 0.000304134994876, 0.000000000000000, -0.000209665187250, 0.000000000000000, -0.000244609385125, 8.333333333333334, -0.000063743042781, 1.260390784477221, 0.006752863567606, 
-0.000304134994876, 0.000000000000000, -0.000209665187250, 0.000000000000000, -0.000244609385125, 0.000000000000000, -0.000063743042781, 8.333333333333334, -0.000708022137620, -0.001298052908773, 
1.693014673677355, 0.002938233259377, 0.000000000000000, -0.002124066412860, 0.630195392238611, -0.002124066412860, 1.260390784477221, -0.000708022137620, 6.250000000000000, 0.032269615509369, 
0.041115074044017, 0.003526084701227, 0.040517181405638, 0.000000000000000, -0.006752863567606, -0.003461474423395, 0.006752863567606, -0.001298052908773, 0.032269615509369, 6.250000000000000;

	// CovMaterniso3FDI
	covMaterniso3FDI.setTrainingInputs(pXd, pX);
	pK = covMaterniso3FDI.K(pLogHyp);

	// check
    BOOST_CHECK_EQUAL(((K - (*pK)).array().abs() < epsilon).all(), true);

	// K_FDI_log(ell)
	K << 
0.000000000000000, 0.007679408620615, -0.000000000000000, -0.004381640993428, -0.000000000000000, -0.005111914492333, -0.000000000000000, -0.002921093995619, 3.150976961193057, 0.256608815569039, 
0.007679408620615, 0.000000000000000, 0.004381640993428, -0.000000000000000, 0.005111914492333, -0.000000000000000, 0.002921093995619, -0.000000000000000, 0.026904841229566, 0.031585954113480, 
-0.000000000000000, 0.004381640993428, -16.666666666666668, -0.001975853096982, -0.000000000000000, -0.003157147695201, -0.000000000000000, -0.001804084397258, 0.000000000000000, 0.207369048135220, 
-0.004381640993428, -0.000000000000000, -0.001975853096982, -16.666666666666668, -0.003157147695201, -0.000000000000000, -0.001804084397258, -0.000000000000000, -0.017133666836434, 0.000000000000000, 
-0.000000000000000, 0.005111914492333, -0.000000000000000, -0.003157147695201, -16.666666666666668, -0.002953065478830, -0.000000000000000, -0.002104765130134, 0.366766721520639, -0.034561508022537, 
-0.005111914492333, -0.000000000000000, -0.003157147695201, -0.000000000000000, -0.002953065478830, -16.666666666666668, -0.002104765130134, -0.000000000000000, -0.017133666836434, -0.027227146876233, 
-0.000000000000000, 0.002921093995619, -0.000000000000000, -0.001804084397258, -0.000000000000000, -0.002104765130134, -16.666666666666668, -0.000472449432600, 0.733533443041279, 0.034561508022537, 
-0.002921093995619, -0.000000000000000, -0.001804084397258, -0.000000000000000, -0.002104765130134, -0.000000000000000, -0.000472449432600, -16.666666666666668, -0.005711222278811, -0.010210180078587, 
3.150976961193057, 0.026904841229566, 0.000000000000000, -0.017133666836434, 0.366766721520639, -0.017133666836434, 0.733533443041279, -0.005711222278811, 0.000000000000000, 0.210166458786028, 
0.256608815569039, 0.031585954113480, 0.207369048135220, 0.000000000000000, -0.034561508022537, -0.027227146876233, 0.034561508022537, -0.010210180078587, 0.210166458786028, 0.000000000000000;

	// partial CovMaterniso3FDI w.r.t log(ell)
	pK = covMaterniso3FDI.K(pLogHyp, 0);
	//std::cout << "error < eps " << std::endl << ((K - (*pK)).array().abs() < epsilon).matrix() << std::endl << std::endl;

	// check
    BOOST_CHECK_EQUAL(((K - (*pK)).array().abs() < epsilon).all(), true);

	// K_FDI_log(sigma_f)
	K << 
12.500000000000000, 0.001437562117592, 0.000000000000000, -0.000912404984628, 0.000000000000000, -0.001064472482065, 0.000000000000000, -0.000608269989752, 3.386029347354711, 0.082230148088033, 
0.001437562117592, 12.500000000000000, 0.000912404984628, 0.000000000000000, 0.001064472482065, 0.000000000000000, 0.000608269989752, 0.000000000000000, 0.005876466518754, 0.007052169402453, 
0.000000000000000, 0.000912404984628, 16.666666666666668, -0.000476928064311, 0.000000000000000, -0.000733828155374, 0.000000000000000, -0.000419330374499, 0.000000000000000, 0.081034362811276, 
-0.000912404984628, 0.000000000000000, -0.000476928064311, 16.666666666666668, -0.000733828155374, 0.000000000000000, -0.000419330374499, 0.000000000000000, -0.004248132825721, 0.000000000000000, 
0.000000000000000, 0.001064472482065, 0.000000000000000, -0.000733828155374, 16.666666666666668, -0.000704065350498, 0.000000000000000, -0.000489218770249, 1.260390784477221, -0.013505727135213, 
-0.001064472482065, 0.000000000000000, -0.000733828155374, 0.000000000000000, -0.000704065350498, 16.666666666666668, -0.000489218770249, 0.000000000000000, -0.004248132825721, -0.006922948846790, 
0.000000000000000, 0.000608269989752, 0.000000000000000, -0.000419330374499, 0.000000000000000, -0.000489218770249, 16.666666666666668, -0.000127486085562, 2.520781568954442, 0.013505727135213, 
-0.000608269989752, 0.000000000000000, -0.000419330374499, 0.000000000000000, -0.000489218770249, 0.000000000000000, -0.000127486085562, 16.666666666666668, -0.001416044275240, -0.002596105817546, 
3.386029347354711, 0.005876466518754, 0.000000000000000, -0.004248132825721, 1.260390784477221, -0.004248132825721, 2.520781568954442, -0.001416044275240, 12.500000000000000, 0.064539231018739, 
0.082230148088033, 0.007052169402453, 0.081034362811276, 0.000000000000000, -0.013505727135213, -0.006922948846790, 0.013505727135213, -0.002596105817546, 0.064539231018739, 12.500000000000000;

	// partial CovMaterniso3FDI w.r.t log(sigma_f)
	pK = covMaterniso3FDI.K(pLogHyp, 1);

	// check
    BOOST_CHECK_EQUAL(((K - (*pK)).array().abs() < epsilon).all(), true);

	// Ks_FDI
	Matrix K_(n1*(d+1)+n2, m);
	K_ << 
0.041115074044017, 0.108445407729153, 0.308119361567033, 0.009939312900957, 0.754487924615610, 0.015155134185115, 0.000799370879559, 
0.007585743053829, 0.119373088616289, 0.001047226386691, 0.000302089229212, 0.007585743053829, 0.082024162486836, 3.214621384099061, 
0.040517181405638, 0.061968804416659, 0.213936409320460, 0.009546084564308, 0.000000000000000, 0.010927094910066, 0.000426702356811, 
0.000000000000000, -0.069323955612641, -0.000343487012054, 0.000029740674262, -0.006057388573651, -0.014931356949246, -1.627866683271207, 
0.006752863567606, 0.061968804416659, -0.142624272880306, -0.002727452732659, 0.648814847534887, 0.010927094910066, 0.000597383299535, 
-0.006057388573651, -0.092431940816855, -0.001030461036163, -0.000267666068362, -0.004038259049101, -0.029862713898492, 0.000000000000000, 
-0.006752863567606, 0.061968804416659, 0.142624272880306, -0.002727452732659, 0.216271615844962, -0.002185418982013, 0.000426702356811, 
-0.005047823811376, -0.023107985204214, -0.000228991341370, -0.000178444045575, -0.003028694286826, -0.074656784746230, 1.627866683271209, 
0.023640531963645, 0.442043839782440, 0.274825575237353, 0.003316874388471, 1.693014673677356, 0.016293383157891, 0.004246110669769, 
1.017418517656051, 0.089899911429187, 0.655510558942211, 0.655510558942210, 0.014105293314179, 0.032269615509369, 0.002183589344730;

	// partial CovMaterniso3FDI w.r.t log(ell)
	pK = covMaterniso3FDI.Ks(pXs, pLogHyp);

	// check
    BOOST_CHECK_EQUAL(((K_ - (*pK)).array().abs() < epsilon).all(), true);
}

BOOST_AUTO_TEST_SUITE_END()

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
BOOST_AUTO_TEST_SUITE(suite_gpFDI)

// TEST1: CovSEiso predict
BOOST_AUTO_TEST_CASE(covSEisoFDI_predict) {
	// mu
	Vector Mu(7);
	Mu << 
1.167265651407814, 
0.121667561812148, 
0.727590824727964, 
0.598656126110267, 
1.085253102760424, 
0.029276929914667, 
1.351836296683708;

	// variance
	Matrix Variance(7, 1);
	Variance <<
6.073996552673053, 
6.233713024506768, 
6.177563738357781, 
6.203606885673762, 
5.432609949234843, 
6.249866103880930, 
1.621649502390111;

	// hyperparameters
	pMeanLogHypFDI.reset(new GPCovSEisoFDI::MeanHyp());
	pCovLogHypFDI.reset(new GPCovSEisoFDI::CovHyp());
	pLikLogHypFDI.reset(new GPCovSEisoFDI::LikHyp());	
	(*pCovLogHypFDI)(0)	= log(1.5f);
	(*pCovLogHypFDI)(1)	= log(2.5f);
	(*pLikLogHypFDI)(0)	= log(0.3f);
	(*pLikLogHypFDI)(1)	= log(0.5f);

	// Y
	const int n = n1*(d+1) + n2;
	pY.reset(new Vector(n));

	(*pY) << 1, 4, 1, -1, 2, -2, 3, -3, 3, 7;

	// predict 
	VectorPtr pMu;
	MatrixPtr pVariance;
	gpCovSEisoFDI.setTrainingData(pXd, pX, pY);
	gpCovSEisoFDI.predict(pMeanLogHypFDI, pCovLogHypFDI, pLikLogHypFDI, pXs, 
											   pMu, pVariance);

	//std::cout << "Mu = " << std::endl << Mu << std::endl;
	//std::cout << "pMu = " << std::endl << *pMu << std::endl;
	//std::cout << "Variance = " << std::endl << Variance << std::endl;
	//std::cout << "pVariance = " << std::endl << *pVariance << std::endl;

	// check
    BOOST_CHECK_EQUAL(((Mu - (*pMu)).array().abs() < epsilon).all(), true);
    BOOST_CHECK_EQUAL(((Variance - (*pVariance)).array().abs() < epsilon).all(), true);
}

// TEST2: CovMaterniso3 predict
BOOST_AUTO_TEST_CASE(covMaterniso3FDI_predict) {
	// mu
	Vector Mu(7);
	Mu <<    
1.142484608740144, 
0.420795894405955, 
0.877201196518749, 
0.723152425724845, 
0.892923033731164, 
0.131651508394764, 
1.650658671872036;

	// variance
	Matrix Variance(7, 1);
	Variance <<
6.086360539738958, 
6.213542618602832, 
6.152116346571196, 
6.182214471291521, 
5.753165083525668, 
6.247909118334887, 
4.002599441046522;

	// hyperparameters
	pMeanLogHypFDI.reset(new GPCovMaterniso3FDI::MeanHyp());
	pCovLogHypFDI.reset(new GPCovMaterniso3FDI::CovHyp());
	pLikLogHypFDI.reset(new GPCovMaterniso3FDI::LikHyp());	
	(*pCovLogHypFDI)(0) = log(1.5f);
	(*pCovLogHypFDI)(1) = log(2.5f);
	(*pLikLogHypFDI)(0)	= log(0.3f);
	(*pLikLogHypFDI)(1)	= log(0.5f);

	// predict 
	VectorPtr pMu;
	MatrixPtr pVariance;
	gpCovMaterniso3FDI.setTrainingData(pXd, pX, pY);
	gpCovMaterniso3FDI.predict(pMeanLogHypFDI, pCovLogHypFDI, pLikLogHypFDI, pXs, 
												 pMu, pVariance);


	// check
    BOOST_CHECK_EQUAL(((Mu - (*pMu)).array().abs() < epsilon).all(), true);
    BOOST_CHECK_EQUAL(((Variance - (*pVariance)).array().abs() < epsilon).all(), true);
}

// TEST3: nlZ, dnlZ
BOOST_AUTO_TEST_CASE(covSEisoFDI_nlZ_dnlZ) {
	// mu
	Scalar nlZ_ =  22.463837737581208;

	// variance
	Vector dnlZ(4);
	dnlZ <<  1.904846643457619,
				   -9.370540033620424,
				   -0.081760870083573,
				   -0.217650229763713;

	// nlZ, dnlZ
	gpCovSEisoFDI.negativeLogMarginalLikelihood(pMeanLogHypFDI, pCovLogHypFDI, pLikLogHypFDI, 
																							nlZ, 
																							pDnlZ);

	//std::cout << "nlZ_ = " << std::endl << nlZ_ << std::endl << std::endl;
	//std::cout << "nlZ = " << std::endl << nlZ << std::endl << std::endl;
	//std::cout << "dnlZ_ = " << std::endl << dnlZ << std::endl << std::endl;
	//std::cout << "dnlZ = " << std::endl << *pDnlZ << std::endl << std::endl;

	// check
	BOOST_CHECK_EQUAL(std::abs(nlZ_ - nlZ) < epsilon, true);
    BOOST_CHECK_EQUAL(((dnlZ - (*pDnlZ)).array().abs() < epsilon).all(), true);
}

// TEST4: nlZ, dnlZ
BOOST_AUTO_TEST_CASE(covMaterniso3FDI_nlZ_dnlZ) {
	// mu
	Scalar nlZ_ =   22.766376457009354;

	// variance
	Vector dnlZ(4);
	dnlZ <<  -3.625130096684514,
				  -4.427828362546538,
				  -0.097970450407589,
				  0.093270935853114;

	// nlZ, dnlZ
	gpCovMaterniso3FDI.negativeLogMarginalLikelihood(pMeanLogHypFDI, pCovLogHypFDI, pLikLogHypFDI, 
																								  nlZ, 
																								  pDnlZ);

	// check
	BOOST_CHECK_EQUAL(std::abs(nlZ_ - nlZ) < epsilon, true);
    BOOST_CHECK_EQUAL(((dnlZ - (*pDnlZ)).array().abs() < epsilon).all(), true);
}

// TEST5: covSEiso train hyperparameters
BOOST_AUTO_TEST_CASE(covSEisoFDI_train) {
	// initial hyperparameters
	(*pCovLogHypFDI)(0) = log(1.5f);
	(*pCovLogHypFDI)(1) = log(2.5f);
	(*pLikLogHypFDI)(0)	= log(0.3f);
	(*pLikLogHypFDI)(1)	= log(0.5f);

	// trained hyperparameters
	GPCovSEisoFDI::CovHyp			covLogHypFDI;		covLogHypFDI << 1.731029732129951, 1.523397558917514;
	GPCovSEisoFDI::LikHyp				likLogHypFDI;		likLogHypFDI << -2.562328044825808, 0.504947234287582;

	// nlZ
	Scalar nlZ_ = 22.463837737581208;

	// train
	gpCovSEisoFDI.train<BFGS, DeltaFunc>(pMeanLogHypFDI, pCovLogHypFDI, pLikLogHypFDI, 10);
	gpCovSEisoFDI.negativeLogMarginalLikelihood(pMeanLogHypFDI, pCovLogHypFDI, pLikLogHypFDI, 
																						 nlZ, 
																						 pDnlZ);

	// check
	std::cout << "covHypFDI = " << std::endl << covLogHypFDI.array().exp() << std::endl << std::endl;
	std::cout << "pCovHypFDI = " << std::endl << pCovLogHypFDI->array().exp() << std::endl << std::endl;
	std::cout << "likHypFDI = " << std::endl << likLogHypFDI.array().exp() << std::endl << std::endl;
	std::cout << "pLikHypFDI = " << std::endl << pLikLogHypFDI->array().exp() << std::endl << std::endl;
	std::cout << "nlZ_ = " << std::endl << nlZ_ << std::endl << std::endl;
	std::cout << "nlZ = " << std::endl << nlZ << std::endl << std::endl;
    BOOST_CHECK_EQUAL(((covLogHypFDI - (*pCovLogHypFDI)).array().abs() < epsilon).all(), true);
    BOOST_CHECK_EQUAL(((likLogHypFDI - (*pLikLogHypFDI)).array().abs() < epsilon).all(), true);
	BOOST_CHECK_EQUAL(std::abs(nlZ_ - nlZ) < epsilon, true);
}

// TEST6: covMaterniso3 train hyperparameters
BOOST_AUTO_TEST_CASE(covMaterniso3FDI_train) {
	// initial hyperparameters
	(*pCovLogHypFDI)(0) = log(1.5f);
	(*pCovLogHypFDI)(1) = log(2.5f);
	(*pLikLogHypFDI)(0)	= log(0.3f);
	(*pLikLogHypFDI)(0)	= log(0.5f);

	// trained hyperparameters
	GPCovSEisoFDI::CovHyp			covLogHypFDI;		covLogHypFDI << 2.032838483561243, 1.502523725227370;
	GPCovSEisoFDI::LikHyp				likLogHypFDI;		likLogHypFDI << -2.153018386984240, 0.517336654669191;

	// nlZ
	Scalar nlZ_ = 22.766376457009354;

	// train
	gpCovMaterniso3FDI.train<BFGS, DeltaFunc>(pMeanLogHypFDI, pCovLogHypFDI, pLikLogHypFDI);
	gpCovMaterniso3FDI.negativeLogMarginalLikelihood(pMeanLogHypFDI, pCovLogHypFDI, pLikLogHypFDI, 
																								  nlZ, 
																								  pDnlZ);

	// check
	std::cout << "covHypFDI = " << std::endl << covLogHypFDI.array().exp() << std::endl << std::endl;
	std::cout << "pCovHypFDI = " << std::endl << pCovLogHypFDI->array().exp() << std::endl << std::endl;
	std::cout << "likHypFDI = " << std::endl << likLogHypFDI.array().exp() << std::endl << std::endl;
	std::cout << "pLikHypFDI = " << std::endl << pLikLogHypFDI->array().exp() << std::endl << std::endl;
	std::cout << "nlZ_ = " << std::endl << nlZ_ << std::endl << std::endl;
	std::cout << "nlZ = " << std::endl << nlZ << std::endl << std::endl;
    BOOST_CHECK_EQUAL(((covLogHypFDI - (*pCovLogHypFDI)).array().abs() < epsilon).all(), true);
    BOOST_CHECK_EQUAL(((likLogHypFDI - (*pLikLogHypFDI)).array().abs() < epsilon).all(), true);
	BOOST_CHECK_EQUAL(std::abs(nlZ_ - nlZ) < epsilon, true);
}

BOOST_AUTO_TEST_SUITE_END()

#else

#include <iostream>
#include <string>

#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/mls.h>
#include <pcl/visualization/cloud_viewer.h>	// pcl::visualization::CloudViewer

#include "GP/Mean/MeanZero.hpp"
#include "GP/Cov/CovMaternisoFDI.hpp"
#include "GP/Lik/LikGauss.hpp"
#include "GP/Inf/InfExact.hpp"
//#include "InfExactUnstable.hpp"

#include "util/surfaceNormals.hpp"
#include "GPOM.hpp"
using namespace GPOM;

typedef GaussianProcessOccupancyMap<MeanZero, CovMaterniso3FDI, LikGauss, InfExact> GPOMType;

int main()
{
	// Point Clouds - Hits
	pcl::PointCloud<pcl::PointXYZ>::Ptr pHitPoints(new pcl::PointCloud<pcl::PointXYZ>);

	// Load data from a PCD file
	//std::string filenName("input.pcd");
	std::string filenName("../../../PCL/PCL-1.5.1-Source/test/bunny.pcd");
	if (pcl::io::loadPCDFile<pcl::PointXYZ>(filenName, *pHitPoints) == -1)
	{
		PCL_ERROR("Couldn't read file!\n");
		return -1;
	}
	else
	{
		std::cout << pHitPoints->size() << " points are successfully loaded." << std::endl;
	}

	// Point Clouds - Robot positions
	pcl::PointCloud<pcl::PointXYZ>::Ptr pRobotPositions(new pcl::PointCloud<pcl::PointXYZ>);
	pRobotPositions->push_back(pcl::PointXYZ(0.f, 0.075f, 1.0f));

	//// min, max
	//pcl::PointXYZ min, max;
	//pcl::getMinMax3D (*pHitPoints, min, max);
	//std::cout << "min = " << min << std::endl;
	//std::cout << "max = " << max << std::endl;

	//// viewer
	//pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");
	//viewer.showCloud(pHitPoints);
	//while(!viewer.wasStopped ())
	//{
	//}

	// surface normals
	//pcl::PointCloud<pcl::PointNormal>::Ptr pPointNormals;
	//smoothAndNormalEstimation(pHitPoints, pPointNormals);
	const float searchRadius = 0.03f;
	pcl::PointCloud<pcl::Normal>::Ptr pNormals = estimateSurfaceNormals(pHitPoints, searchRadius);


	// GPOM
	const float mapResolution = 0.1f; // 10cm
	const float octreeResolution = 0.02f; //10.f; 
	GPOMType gpom;
	gpom.build(pHitPoints, pNormals, pRobotPositions, mapResolution);
	//gpom.build(pHitPoints, pNormals, mapResolution, octreeResolution);
}
#endif
