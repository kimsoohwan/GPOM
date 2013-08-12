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

	void build(pcl::PointCloud<pcl::PointXYZ>::ConstPtr			pPoints, 
					  pcl::PointCloud<pcl::Normal>::ConstPtr				pNormals,
					  const float																	mapResolution)
	{
		pcl::PointXYZ min, max;
		pcl::getMinMax3D (*pPoints, min, max);
		//std::cout << "min = " << min << std::endl;
		//std::cout << "max = " << max << std::endl;
		build(pPoints, pNormals, min, max, mapResolution);
	}

	void build(pcl::PointCloud<pcl::PointXYZ>::ConstPtr			pPoints, 
					  pcl::PointCloud<pcl::Normal>::ConstPtr				pNormals,
					  const pcl::PointXYZ													&min,
					  const pcl::PointXYZ													&max,
					  const float																	mapResolution)
	{
		// training data
		const Eigen::MatrixXf pointsMatrix = pPoints->getMatrixXfMap(3, 4, 0);
		//pointsMatrix.transposeInPlace();
		std::cout << "rows: " << pointsMatrix.rows() << std::endl;
		std::cout << "cols: " << pointsMatrix.cols() << std::endl;
		//std::cout << pointsMatrix << std::endl;

		//m_gp.train(pX, pY); // with default parameters

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