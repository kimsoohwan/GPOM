#if 0

#include <iostream>
#include <pcl/common/io.h>		// for concatenateFields
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include "util/loadPointCloud.hpp"
#include "util/filters.hpp"
#include "util/surfaceNormals.hpp"
#include "util/visualization.hpp"

using namespace GPOM;

int main()
{
	//std::string dataPath("../../data/");
	std::string dataPath("../../data/simulation2/");

	// robot positions
	std::string robotPositionsFilenName = dataPath + "robot_positions.pcd";
	pcl::PointCloud<pcl::PointXYZ>::Ptr robotPositions = loadPointCloud<pcl::PointXYZ>(robotPositionsFilenName);

	// for each scan
	int from, to;
	//std::cout << "from ";	std::cin >> from;
	//std::cout << "to ";		std::cin >> to;
	//const int numScans = 81;
	from = 1; to = 22;
	for(int scan = from; scan <= to; scan++)
	{
		// load points
		std::string pointsFilenNameIn = dataPath + std::to_string((long double) scan) + "_clean_points.pcd";
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = loadPointCloud<pcl::PointXYZ>(pointsFilenNameIn);

		double radius = 2.f;
		//for(int i = 0; i < 5; i++, radius += 0.1f)
		//{
			std::cout << "\tradius = " << radius << std::endl;

			// estimate normals
			pcl::PointCloud<pcl::Normal>::Ptr normals = estimateSurfaceNormals(cloud, (*robotPositions)[scan-1], radius);

			// concatenate
			pcl::PointCloud<pcl::PointNormal>::Ptr pointNormals(new pcl::PointCloud<pcl::PointNormal>());
			pcl::concatenateFields<pcl::PointXYZ, pcl::Normal, pcl::PointNormal>(*cloud, *normals, *pointNormals);

			// save
			std::string pointsFilenNameOut = dataPath + std::to_string((long double) scan) + "_clean_point_normals_" + to_string((long double) radius) + ".pcd";
			pcl::io::savePCDFile(pointsFilenNameOut, *pointNormals, true);
		//}
		//std::cout << std::endl;
	}

	// merge
	double radius = 2.f;
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud(new pcl::PointCloud<pcl::PointNormal>());
	for(int scan = from; scan <= to; scan++)
	{
		// load points
		std::string filename = dataPath + std::to_string((long double) scan) + "_clean_point_normals_" + std::to_string((long double) radius) + ".pcd";
		appendPointCloud<pcl::PointNormal>(filename, cloud);
	}

	// save
	std::string pointsFilenNameOut = dataPath +"clean_point_normals_" + to_string((long double) radius) + ".pcd";
	pcl::io::savePCDFile(pointsFilenNameOut, *cloud, true);

	// show
	const double scale = 1.f;
	showPointCloudNormals<pcl::PointNormal>(cloud, scale);
	return 0;
}
#endif

#if 0

#include <iostream>
#include <string>
#include <pcl/common/io.h>		// for concatenateFields
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include "util/loadPointCloud.hpp"
#include "util/visualization.hpp"

using namespace GPOM;

int main()
{
	std::string dataPath("../../data/");

	// for each scan
	const int scan = 1;
	double radius = 0.2f;
	for(int i = 0; i <= 5-1; i++, radius += 0.1f)
	{
		std::cout << "Compare radius = " << radius << " vs radius = " << radius + 0.1f << std::endl;

		// load points
		std::string pointsFilenName1 = dataPath + std::to_string((long double) scan) + "_clean_point_normals_" + std::to_string((long double) radius) + ".pcd";
		pcl::PointCloud<pcl::PointNormal>::Ptr cloud1 = loadPointCloud<pcl::PointNormal>(pointsFilenName1);
		std::string pointsFilenName2 = dataPath + std::to_string((long double) scan) + "_clean_point_normals_" + std::to_string((long double) radius + 0.1f) + ".pcd";
		pcl::PointCloud<pcl::PointNormal>::Ptr cloud2 = loadPointCloud<pcl::PointNormal>(pointsFilenName2);

		// compare normals
		compareTwoNormals<pcl::PointNormal>(cloud1, cloud2, std::to_string((long double) radius), std::to_string((long double) radius + 0.1f));
	}

	return 0;
}

#endif

#if 0

#include <iostream>
#include <string>
#include <pcl/common/io.h>		// for concatenateFields
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include "util/loadPointCloud.hpp"
#include "util/visualization.hpp"

using namespace GPOM;

int main()
{
	//std::string dataPath("../../data/");
	std::string dataPath("../../data/simulation/");

	// merge normals
	double radius = 0.1f;
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud(new pcl::PointCloud<pcl::PointNormal>());
	const int numScans = 22;
	for(int scan = 1; scan <= numScans; scan++)
	{
		// load points
		std::string filename = dataPath + std::to_string((long double) scan) + "_clean_point_normals_" + std::to_string((long double) radius) + ".pcd";
		appendPointCloud<pcl::PointNormal>(filename, cloud);
	}

	// show
	showPointCloudNormals<pcl::PointNormal>(cloud);
	return 0;
}
#endif