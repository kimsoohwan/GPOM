#if 1
#include <iostream>
#include <string>

#include <pcl/io/pcd_io.h>
//#include <pcl/visualization/cloud_viewer.h>	// pcl::visualization::CloudViewer

#include "GP/Mean/MeanZeroFDI.hpp"
#include "GP/Cov/CovMaternisoFDI.hpp"
#include "GP/Cov/CovSparseisoFDI.hpp"
#include "GP/Lik/LikGaussFDI.hpp"
#include "GP/Inf/InfExactFDI.hpp"

#include "util/surfaceNormals.hpp"
#include "util/int2string.hpp"
#include "GPOM.hpp"
using namespace GPOM;

typedef GaussianProcessOccupancyMap<MeanZeroFDI, CovMaterniso3FDI, LikGaussFDI, InfExactFDI> GPOMType;
//typedef GaussianProcessOccupancyMap<MeanZeroFDI, CovSparseisoFDI, LikGaussFDI, InfExactFDI> GPOMType;

int main()
{
#if 0
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
	pcl::PointCloud<pcl::Normal>::Ptr pNormals = estimateSurfaceNormals(pHitPoints, robotPosition, searchRadius);

	// Point Clouds - Robot positions
	pcl::PointXYZ		robotPosition(0.f, 0.075f, 1.0f);
	pcl::PointCloud<pcl::PointXYZ>::Ptr pRobotPositions(new pcl::PointCloud<pcl::PointXYZ>);
	pRobotPositions->push_back(robotPosition);
#else
	std::string dataPath("../../data/");

	// robot positions
	std::cout << "loading robot positions ... " << std::endl;
	pcl::PointCloud<pcl::PointXYZ>::Ptr pRobotPositions(new pcl::PointCloud<pcl::PointXYZ>);
	std::string robotPositionsFilenName = dataPath + "robot_positions.pcd";
	if (pcl::io::loadPCDFile<pcl::PointXYZ>(robotPositionsFilenName, *pRobotPositions) == -1)
	{
		PCL_ERROR("Couldn't read file!\n");
		return -1;
	}
	else
	{
		std::cout << pRobotPositions->size() << " robot positions are successfully loaded." << std::endl;
	}

	// points
	std::cout << "loading points ... " << std::endl;
	pcl::PointCloud<pcl::PointXYZ>::Ptr pHitPoints(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr pTempPoints(new pcl::PointCloud<pcl::PointXYZ>);
	for(int scan = 1; scan < 2; scan++)
	{
		std::string pointsFilenName = dataPath + to_string(scan) + "_points.pcd";
		if (pcl::io::loadPCDFile<pcl::PointXYZ>(pointsFilenName, *pTempPoints) == -1)
		{
			PCL_ERROR("Couldn't read file!\n");
			return -1;
		}
		else
		{
			(*pHitPoints) += (*pTempPoints);
			std::cout << pHitPoints->size() << " hit points are successfully loaded." << std::endl;
		}
	}

	// surface normals
	std::cout << "loading normals ... " << std::endl;
	pcl::PointCloud<pcl::Normal>::Ptr pNormals(new pcl::PointCloud<pcl::Normal>);
	pcl::PointCloud<pcl::Normal>::Ptr pTempNormals(new pcl::PointCloud<pcl::Normal>);
	for(int scan = 1; scan < 2; scan++)
	{
		std::string normalsFilenName = dataPath + to_string(scan) + "_normals.pcd";
		if (pcl::io::loadPCDFile<pcl::Normal>(normalsFilenName, *pTempNormals) == -1)
		{
			PCL_ERROR("Couldn't read file!\n");
			return -1;
		}
		else
		{
			(*pNormals) += (*pTempNormals);
			std::cout << pNormals->size() << " surface normals are successfully loaded." << std::endl;
		}
	}
#endif

	// GPOM
	const Scalar mapResolution = 0.1f;		// 10cm
	const Scalar blockSize = 2.f;				// 2m; 
	GPOMType gpom(mapResolution);
	gpom.build(pHitPoints, pNormals, pRobotPositions, blockSize);
}

#else
#include "Octree/OctreeGPOM.hpp"
//#include "Octree/OctreeGPOMViewer.hpp"

using namespace GPOM;

int main()
{
	// octree
	const Scalar mapResolution				= 0.1f; // 10cm
	OctreeGPOM octree(mapResolution);

	// set bounding box
	const float minValue = std::numeric_limits<float>::epsilon();
	pcl::PointXYZ octreeMin(-10, -10, -10);
	pcl::PointXYZ octreeMax(10, 10, 10);
	octree.defineBoundingBox(octreeMin.x, octreeMin.y, octreeMin.z, octreeMax.x, octreeMax.y, octreeMax.z);

	// merge
	octree.mergeMeanAndVarianceAtPoint(pcl::PointXYZ(1.05, 2.05, 3.05), 0.f, 1.f);
	octree.mergeMeanAndVarianceAtPoint(pcl::PointXYZ(1.05, 2.05, 3.05), 0.1f, 2.f);

	octree.mergeMeanAndVarianceAtPoint(pcl::PointXYZ(3.05, 4.05, 5.05), 0.f, 2.f);
	octree.mergeMeanAndVarianceAtPoint(pcl::PointXYZ(3.05, 4.05, 5.05), -0.1f, 2.f);

	octree.mergeMeanAndVarianceAtPoint(pcl::PointXYZ(6.05, 7.05, 8.05), -1.f, 0.1f);

	octree.mergeMeanAndVarianceAtPoint(pcl::PointXYZ(9.05, 10.05, 11.05), -1.f, 0.1f);
	std::cout << "number of leaf nodes: " << octree.getLeafCount() << std::endl;

	// iterate
	OctreeGPOM::LeafNodeIterator iter(octree);
	GaussianDistribution gaussian;
	pcl::PointXYZ center;
	Scalar occupancy;
	while(*++iter)
	{
		// query Gaussian distribution
		octree.getGaussianDistributionAtLeafNode(iter, center, gaussian);
		std::cout << "(" << center.x << ", " << center.y << ", " << center.z << "): mean = " << gaussian.getMean() << ", variance = " << gaussian.getVariance() << std::endl;

		// query occupancy
		octree.getOccupancyAtLeafNode(iter, center, occupancy);
		std::cout << "(" << center.x << ", " << center.y << ", " << center.z << "): occupancy = " << occupancy << std::endl;
	}

	// visualize
	OctreeGPOM::AlignedPointTVector voxelCenterList;
	int numOccupiedVoxels = octree.getOccupiedVoxelCenters(voxelCenterList);
	for(unsigned int i = 0; i < voxelCenterList.size(); i++)
		std::cout << voxelCenterList[i] << std::endl;
	//pcl::PointCloud<pcl::PointXYZ>::Ptr pCloud(new pcl::PointCloud<pcl::PointXYZ>());
	//OctreeGPOMViewer viewer(pCloud, octree);

	return 0;
}
#endif