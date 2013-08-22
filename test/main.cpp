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

	// points
	std::cout << "loading points ... " << std::endl;
	pcl::PointCloud<pcl::PointXYZ>::Ptr pHitPoints(new pcl::PointCloud<pcl::PointXYZ>);
	std::string pointsFilenName("../../data/1_points.pcd");
	if (pcl::io::loadPCDFile<pcl::PointXYZ>(pointsFilenName, *pHitPoints) == -1)
	{
		PCL_ERROR("Couldn't read file!\n");
		return -1;
	}
	else
	{
		std::cout << pHitPoints->size() << " points are successfully loaded." << std::endl;
	}

	// surface normals
	std::cout << "loading normals ... " << std::endl;
	pcl::PointCloud<pcl::Normal>::Ptr pNormals(new pcl::PointCloud<pcl::Normal>);
	std::string normalsFilenName("../../data/1_normals.pcd");
	if (pcl::io::loadPCDFile<pcl::Normal>(normalsFilenName, *pNormals) == -1)
	{
		PCL_ERROR("Couldn't read file!\n");
		return -1;
	}
	else
	{
		std::cout << pNormals->size() << " surface normals are successfully loaded." << std::endl;
	}

	// Point Clouds - Robot positions
	pcl::PointXYZ		robotPosition(0.f, 0.f, 0.f);
	pcl::PointCloud<pcl::PointXYZ>::Ptr pRobotPositions(new pcl::PointCloud<pcl::PointXYZ>);
	pRobotPositions->push_back(robotPosition);
#endif

	// GPOM
	const float mapResolution = 0.1f;		// 10cm
	const float octreeResolution = 2.f;	// 2m; 
	GPOMType gpom;
	gpom.build(pHitPoints, pNormals, pRobotPositions, mapResolution, octreeResolution);
	//gpom.build(pHitPoints, pNormals, mapResolution, octreeResolution);
}