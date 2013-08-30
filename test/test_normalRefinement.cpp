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
	//std::string dataPath("../../data/");
	std::string dataPath("../../data/simulation2/");

	// for read normals
	//pcl::PointCloud<pcl::PointNormal>::Ptr pointNormals(new pcl::PointCloud<pcl::PointNormal>());
	//const int numScans = 3;
	//const float radius = 0.1f;
	//for(int scan = 1; scan <= numScans; scan++)
	//{
	//	// load points
	//	std::string filenName = dataPath + std::to_string((long double) scan) + "_clean_point_normals_" + std::to_string((long double) radius) + ".pcd";
	//	appendPointCloud<pcl::PointNormal>(filenName, pointNormals);
	//}

	// save
	//std::string filename1 = dataPath + "clean_point_normals_" + to_string((long double) radius) + "_1_to_3.pcd";
	//pcl::io::savePCDFile(filename1, *pointNormals, true);

	// laod
	double radius = 2.f;
	std::string filename = dataPath +"clean_point_normals_" + to_string((long double) radius) + ".pcd";
	pcl::PointCloud<pcl::PointNormal>::Ptr pointNormals = loadPointCloud<pcl::PointNormal>(filename);

	// normal refinement
	std::cout << "refinement" << std::endl;
	const int k = 5;
	pcl::PointCloud<pcl::PointNormal>::Ptr pointNormalsRefined = normalRefinement<pcl::PointNormal>(pointNormals, k);

	// save
	//std::string filename2 = dataPath + "clean_point_normals_" + to_string((long double) radius) + "_1_to_3_refined.pcd";
	//pcl::io::savePCDFile(filename2, *pointNormalsRefined, true);

	//// voxel grid
	//std::cout << "voxel grid" << std::endl;
	//const float leafSize = 0.1f;
	//pointNormals		= downSample<pcl::PointNormal>(pointNormals, leafSize);
	//pointNormalsRefined = downSample<pcl::PointNormal>(pointNormalsRefined, leafSize);

	// compare
	std::cout << "compare" << std::endl;
	compareTwoNormals<pcl::PointNormal>(pointNormals, pointNormalsRefined, "original normals", "refined normals");

	return 0;
}

#endif