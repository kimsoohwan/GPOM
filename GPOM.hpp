#ifndef GAUSSIAN_PROCESS_CONTINUOUS_OCCUPANCY_MAP_HPP
#define GAUSSIAN_PROCESS_CONTINUOUS_OCCUPANCY_MAP_HPP

//#include <Eigen/Core>

#include <pcl/point_types.h>							
#include <pcl/octree/octree.h>
#include <pcl/octree/octree_impl.h> 

#include "GP/GP.hpp"
#include "util/meshGrid.hpp"

namespace GPOM{

template <class MeanFunc, class CovFunc, class LikFunc, 
				  template <class, class, class> class InfMethod>
class GaussianProcessOccupancyMap
{
public:
	// constructor
	GaussianProcessOccupancyMap() { }

	// destructor
	virtual ~GaussianProcessOccupancyMap() { }

	// set point cloud
	//void setPointCloud(pcl::PointCloud<pcl::PointXYZ> &cloud)
	//{
	//}

	void build(pcl::PointCloud<pcl::PointXYZ>::ConstPtr			pHitPoints, 
					  pcl::PointCloud<pcl::Normal>::ConstPtr				pNormals,
					  pcl::PointCloud<pcl::PointXYZ>::ConstPtr			pRobotPositions, 
					  const float																	mapResolution)
	{
		pcl::PointXYZ min, max;
		pcl::getMinMax3D (*pHitPoints, min, max);
		build(pHitPoints, pNormals, pRobotPositions, min, max, mapResolution);
	}

	void build(pcl::PointCloud<pcl::PointXYZ>::ConstPtr			pHitPoints, 
					  pcl::PointCloud<pcl::Normal>::ConstPtr				pNormals,
					  pcl::PointCloud<pcl::PointXYZ>::ConstPtr			pRobotPositions, 
					  const pcl::PointXYZ													&min,
					  const pcl::PointXYZ													&max,
					  const float																	mapResolution)
	{
		// training inputs
		Matrix Xd		= pHitPoints->getMatrixXfMap(3, 4, 0);
		Matrix X			= pRobotPositions->getMatrixXfMap(3, 4, 0);
		MatrixPtr pXd(&Xd);
		MatrixPtr pX(&X);
		//MatrixPtr pXd = boost::make_shared(pHitPoints->getMatrixXfMap(3, 4, 0));
		//MatrixPtr pX	= boost::make_shared(pRobotPositions->getMatrixXfMap(3, 4, 0));

		// training outputs
		const int nd		= pXd->cols();
		const int n		= pX->cols();
		const int d		= 3;
		std::cout << "nd = " << nd << std::endl;
		std::cout << "n = " << n << std::endl;
		std::cout << "d = " << d << std::endl;
		VectorPtr pY(new Vector(nd*(d+1) + n));
		pY->setZero();
		for(int i = 0; i < nd; i++)
		{
			//(*pY)[i] = (Scalar) 0.f;											// F1(nd),
			(*pY)[nd*1 + i] = (*pNormals)[i].normal_x;			// D1(nd)
			(*pY)[nd*2 + i] = (*pNormals)[i].normal_y;			// D2(nd)
			(*pY)[nd*3 + i] = (*pNormals)[i].normal_z;			// D3(nd)
		}
		//for(int i = 0; i < n; i++)		(*pY)[nd*(d+1) + i] = (Scalar) 0.f;			// F2(n)

		// set training data
		m_gp.setTrainingData(pXd, pX, pY);

		// hyperparameters
		MeanFunc::Hyp		meanLogHyp;
		CovFunc::Hyp		covLogHyp;
		LikFunc::Hyp			likLogHyp;

		// default values
		covLogHyp << log(1.f), log(1.f);
		likLogHyp << log(1.f), log(1.f);

		// train
		std::cout << "training ... " << std::endl;
		m_gp.train<BFGS, DeltaFunc>(meanLogHyp, covLogHyp, likLogHyp, 10);
		std::cout << "done in seconds" << std::endl;

		std::cout << "Mean: " << std::endl << meanLogHyp << std::endl;
		std::cout << "Cov: " << std::endl << covLogHyp << std::endl;
		std::cout << "Lik: " << std::endl << likLogHyp << std::endl;

		// test points
		pcl::PointCloud<pcl::PointXYZ>::Ptr pTestPoints = meshGrid(min, max, mapResolution);
		Matrix Xs		= pTestPoints->getMatrixXfMap(3, 4, 0);
		MatrixPtr pXs(&Xs);
		//MatrixPtr pXs= boost::make_shared(pTestPoints->getMatrixXfMap(3, 4, 0));

		// predict
		VectorPtr	pMu;
		MatrixPtr		pSigma;
		m_gp.predict(meanLogHyp, covLogHyp, likLogHyp, pXs, 
								pMu, pSigma);
	}

	// create a continuous occupancy map
	void build(pcl::PointCloud<pcl::PointXYZ>::ConstPtr			pPoints, 
					  pcl::PointCloud<pcl::Normal>::ConstPtr				pNormals,
					  const float																	mapResolution, 
					  const float																	octreeResolution)
	{
		// store in an octree
		pcl::octree::OctreePointCloud<pcl::PointXYZ> octree(octreeResolution);
		octree.setInputCloud(pPoints);
		octree.addPointsFromInputCloud();

		// instantiate iterator for octreeA
		pcl::octree::OctreePointCloud<pcl::PointXYZ>::LeafNodeIterator leafNodeIter(octree);

		// for each leaf node
		std::vector<int> indexVector;
		unsigned int totalNumPoints = 0;
		while (*++leafNodeIter)
		{
			leafNodeIter.getData (indexVector);
			std::cout << indexVector.size() << std::endl;
			totalNumPoints += indexVector.size();
			indexVector.clear();
		}
		std::cout << "In total, " << totalNumPoints << " points" << std::endl;

		// occupied voxels
		pcl::octree::OctreePointCloud<pcl::PointXYZ>::AlignedPointTVector voxelCenterList;
		octree.getOccupiedVoxelCenters(voxelCenterList);
		pcl::octree::OctreePointCloud<pcl::PointXYZ>::AlignedPointTVector::const_iterator voxelCenterIter;
		for(voxelCenterIter = voxelCenterList.begin(); voxelCenterIter != voxelCenterList.end(); voxelCenterIter++)
		{
			if(octree.isVoxelOccupiedAtPoint (*voxelCenterIter))		std::cout << "occupied: ";
			else																							std::cout << "empty: ";
			std::cout << *voxelCenterIter << std::endl;
		}
	}


protected:
	GaussianProcess<MeanFunc, CovFunc, LikFunc, InfMethod> m_gp;
};

}

#endif