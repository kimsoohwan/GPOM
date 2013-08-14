#ifndef GAUSSIAN_PROCESS_CONTINUOUS_OCCUPANCY_MAP_HPP
#define GAUSSIAN_PROCESS_CONTINUOUS_OCCUPANCY_MAP_HPP

//#include <Eigen/Core>

#include <pcl/point_types.h>							
#include <pcl/octree/octree.h>
#include <pcl/octree/octree_impl.h> 

#include "GP/GP.hpp"

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
		// concatenate point clouds
		//const pcl::PointCloud<pcl::PointXYZ> pX = pHitPoints + pRobotPositions;
		//CovFunc::numRobotPositions = pRobotPositions->size();

		// training inputs
		//const Matrix HitPointMatrix = pHitPoints->getMatrixXfMap(3, 4, 0);
		//const Matrix RobotPositionMatrix = pRobotPositions->getMatrixXfMap(3, 4, 0);
		const int n1 = pHitPoints->size();
		const int n2 = pRobotPositions->size();
		const int d = 3;
		MatrixPtr  pX(new Matrix(d, n1+n2));
		pX->leftCols(n1) = pHitPoints->getMatrixXfMap(3, 4, 0);
		pX->rightCols(n2) = pRobotPositions->getMatrixXfMap(3, 4, 0);

		//pointsMatrix.transposeInPlace();
		//std::cout << "rows: " << X.rows() << std::endl;
		//std::cout << "cols: " << X.cols() << std::endl;
		//std::cout << pointsMatrix << std::endl;

		// training outputs
		VectorPtr pY(new Vector(n1*(d+1) + n2));
		pY->setZero();
		for(int i = 0; i < n1; i++)
		{
			//(*pY)[i] = (Scalar) 0.f;								// F1(n1),
			(*pY)[n1*1 + i] = (*pNormals)[i].normal_x;			// D1(n1)
			(*pY)[n1*2 + i] = (*pNormals)[i].normal_y;			// D2(n1)
			(*pY)[n1*3 + i] = (*pNormals)[i].normal_z;			// D3(n1)
		}
		//for(int i = 0; i < n2; i++)		(*pY)[n1*(d+1) + i] = (Scalar) 0.f;			// F2(n2)

		// train with default parameters
		MeanFunc::HypPtr		pMeanLogHyp(new MeanFunc::Hyp(*(MeanFunc::pDefaultHyp)));
		CovFunc::HypPtr			pCovLogHyp(new CovFunc::Hyp(*(CovFunc::pDefaultHyp)));
		LikFunc::HypPtr				pLikLogHyp(new LikFunc::Hyp(*(LikFunc::pDefaultHyp)));
		m_gp.train<BFGS, DeltaFunc>(pX, pY, pMeanLogHyp, pCovLogHyp, pLikLogHyp);
		std::cout << "Mean: " << std::endl << pMeanLogHyp->array().exp().matrix() << std::endl;
		std::cout << "Cov: " << std::endl << pCovLogHyp->array().exp().matrix() << std::endl;
		std::cout << "Lik: " << std::endl << pLikLogHyp->array().exp().matrix() << std::endl;

		// test points

		//pcl::PointXYZ min, max;
		//pcl::getMinMax3D (*pPoints, min, max);
		//std::cout << "min = " << min << std::endl;
		//std::cout << "max = " << max << std::endl;
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