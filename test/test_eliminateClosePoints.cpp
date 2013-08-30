#if 0

#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include "util/loadPointCloud.hpp"
#include "util/filters.hpp"
#include "util/visualization.hpp"

using namespace GPOM;

int main()
{
	std::string dataPath("../../data/");

	// robot positions
	std::string robotPositionsFilenName = dataPath + "robot_positions.pcd";
	pcl::PointCloud<pcl::PointXYZ>::Ptr robotPositions = loadPointCloud<pcl::PointXYZ>(robotPositionsFilenName);

	// for each scan
	const int numScans = 81;
	for(int scan = 1; scan <= robotPositions->size(); scan++)
	{
		// load points
		std::string pointsFilenNameIn = dataPath + to_string((long double) scan) + "_points.pcd";
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = loadPointCloud<pcl::PointXYZ>(pointsFilenNameIn);

		// eliminate close/far(?) points
		pcl::PointCloud<pcl::PointXYZ>::Ptr clean_cloud = rangeRemoval(cloud, (*robotPositions)[scan-1], 1.f);
		std::cout << "After elimination: " << cloud->size() - clean_cloud->size() << " points were eliminated." << std::endl;
		//compareTwoPointClouds<pcl::PointXYZ>(cloud, clean_cloud);

		// save
		std::string pointsFilenNameOut = dataPath + to_string((long double) scan) + "_clean_points.pcd";
		pcl::io::savePCDFile(pointsFilenNameOut, *clean_cloud, true);
		std::cout << std::endl;
	}
	return 0;
}

#endif