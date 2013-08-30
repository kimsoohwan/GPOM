#if 0

#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include "util/filters.hpp"
#include "util/loadPointCloud.hpp"
#include "util/visualization.hpp"

using namespace GPOM;

int main()
{
	std::string dataPath("../../data/");

	const float radius = 0.1f;

	// read origianl normals
	pcl::PointCloud<pcl::PointNormal>::Ptr pointNormals1(new pcl::PointCloud<pcl::PointNormal>());
	const int numScans = 3;
	for(int scan = 1; scan <= numScans; scan++)
	{
		// load points
		std::string filenName = dataPath + std::to_string((long double) scan) + "_clean_point_normals_" + std::to_string((long double) radius) + ".pcd";
		appendPointCloud<pcl::PointNormal>(filenName, pointNormals1);
	}

	// read refined normals
	std::string filename2 = dataPath + "clean_point_normals_" + to_string((long double) radius) + "_1_to_3_refined.pcd";
	pcl::PointCloud<pcl::PointNormal>::Ptr pointNormals2 = loadPointCloud<pcl::PointNormal>(filename2);

	// voxel grid
	const float leafSize = 0.1f;
	pointNormals1 = downSample<pcl::PointNormal>(pointNormals1, leafSize);
	pointNormals2 = downSample<pcl::PointNormal>(pointNormals2, leafSize);

	// compare
	compareTwoNormals<pcl::PointNormal>(pointNormals1, pointNormals2, "original normals", "refined normals");

	return 0;
}

#endif