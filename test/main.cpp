#if 1
#include <iostream>
#include <string>

#include <pcl/io/pcd_io.h>

#include "GP/Mean/MeanZeroFDI.hpp"
#include "GP/Cov/CovMaternisoFDI.hpp"
#include "GP/Cov/CovSparseisoFDI.hpp"
#include "GP/Lik/LikGaussFDI.hpp"
#include "GP/Inf/InfExactUnstableButFastFDI.hpp"

#include "util/surfaceNormals.hpp"
#include "util/int2string.hpp"
#include "util/filters.hpp"
#include "util/loadPointCloud.hpp"
#include "util/visualization.hpp"

#include "GPOM.hpp"
using namespace GPOM;

typedef GaussianProcessOccupancyMap<MeanZeroFDI, CovMaterniso3FDI, LikGaussFDI, InfExactUnstableButFastFDI> GPOMType;
//typedef GaussianProcessOccupancyMap<MeanZeroFDI, CovSparseisoFDI, LikGaussFDI, InfExactFDI> GPOMType;

int main()
{
	// directory
	std::string dataPath("../../data/simulation2/");

	// point normals
	double radius = 2.f;
	std::string filename = dataPath +"clean_point_normals_" + to_string((long double) radius) + ".pcd";
	pcl::PointCloud<pcl::PointNormal>::Ptr pointNormals = loadPointCloud<pcl::PointNormal>(filename);

	// GPOM
	const Scalar mapResolution = 0.2f;		// 20cm
	GPOMType gpom(mapResolution);

	// build
	const Scalar blockSize = 2.f;				// 2m; 
	const Scalar pruneVarianceThreshold = 0.8;
	const Scalar pruneOccupancyThreshold = 0.3;

	// hyperparameters
	GPOMType::MeanHyp	meanLogHyp;
	GPOMType::CovHyp	covLogHyp;	covLogHyp << log(5.158820125747006f), log(1.863198796627710f);
	GPOMType::LikHyp	likLogHyp;	likLogHyp << log(0.144316317935091f), log(0.000000023020410f);
	const int numMaxIterationsForTraining = 0;
	Eigen::Vector3f min(-8, -10, 0);
	Eigen::Vector3f max(10, 6, 10);
	gpom.build(pointNormals, 
			   meanLogHyp, covLogHyp, likLogHyp, numMaxIterationsForTraining,
			   min, max,
			   pruneVarianceThreshold, pruneOccupancyThreshold);

	// show
	OctreeGPOMViewer<pcl::PointNormal> viewer(pointNormals, gpom);
}

#endif

#if 0
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

#if 0
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include "util/filters.hpp"
#include "util/visualization.hpp"

int
main (int argc, char** argv)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);


  // Fill in the cloud data
  pcl::PCDReader reader;

  // Replace the path below with the path where you saved your file
  std::string dataPath("../../data/");
  std::string pointsFilenName = dataPath + "all_clean_points_cropped.pcd";
  reader.read<pcl::PointXYZ> (pointsFilenName, *cloud);

  std::cerr << "Cloud before filtering: " << std::endl;
  std::cerr << *cloud << std::endl;

  pcl::PointCloud<pcl::PointXYZ>::Ptr inliers
	  = GPOM::statisticalOutlierRemoval<pcl::PointXYZ>(cloud, 50, 2.0);

  std::cerr << "Cloud after filtering: " << std::endl;
  std::cerr << *inliers << std::endl;

  GPOM::compareTwoPointClouds<pcl::PointXYZ>(cloud, inliers);

  return (0);
}
#endif

#if 0 // mls
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include "util/surfaceNormals.hpp"
#include "util/visualization.hpp"

int
main (int argc, char** argv)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);


  // Fill in the cloud data
  pcl::PCDReader reader;

  // Replace the path below with the path where you saved your file
  std::string dataPath("../../data/");
  std::string pointsFilenName = dataPath + "all_clean_points_downsampled_0.05_cropped.pcd";
  reader.read<pcl::PointXYZ> (pointsFilenName, *cloud);

  std::cerr << "Cloud before filtering: " << std::endl;
  std::cerr << *cloud << std::endl;

  double radius = 0.1f;
  for(int i = 0; i < 10; i++, radius += 0.1f)
  {
	std::cout << "moving least squares ... radius = " << radius << " : ";
	pcl::PointCloud<pcl::PointNormal>::Ptr	mls_normals = GPOM::smoothAndNormalEstimation(cloud, radius);
	std::cout << mls_normals->size() << " were estimated." << std::endl;

	string allPointsFilename = dataPath + "all_clean_points_downsampled_0.05_cropped_normals_" + to_string((long double) radius) + ".pcd";
	pcl::io::savePCDFile (allPointsFilename, *mls_normals, true);
  }
  return 0;

}
#endif

#if 0 // mls
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include "util/surfaceNormals.hpp"
#include "util/visualization.hpp"

int
main (int argc, char** argv)
{
  pcl::PointCloud<pcl::PointNormal>::Ptr cloud1 (new pcl::PointCloud<pcl::PointNormal>);
  pcl::PointCloud<pcl::PointNormal>::Ptr cloud2 (new pcl::PointCloud<pcl::PointNormal>);


  // Fill in the cloud data
  pcl::PCDReader reader;
  std::string dataPath("../../data/");
  std::string pointsFilenName;

  // Replace the path below with the path where you saved your file
  pointsFilenName = dataPath + "all_clean_points_downsampled_0.05_cropped_normals_" + to_string((long double) 0.3f) + ".pcd";
  reader.read<pcl::PointNormal> (pointsFilenName, *cloud1);
  pointsFilenName = dataPath + "all_clean_points_downsampled_0.05_cropped_normals_" + to_string((long double) 0.4f) + ".pcd";
  reader.read<pcl::PointNormal> (pointsFilenName, *cloud2);
  std::cerr << "Cloud1: " << *cloud1 << std::endl;
  std::cerr << "Cloud1: " << *cloud2 << std::endl;

  GPOM::compareTwoNormals<pcl::PointNormal>(cloud1, cloud2);

  return (0);
}
#endif


#if 0 // mls
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include "util/surfaceNormals.hpp"
#include "util/visualization.hpp"

int
main (int argc, char** argv)
{
  pcl::PointCloud<pcl::PointNormal>::Ptr cloud1 (new pcl::PointCloud<pcl::PointNormal>);
  pcl::PointCloud<pcl::PointNormal>::Ptr cloud2 (new pcl::PointCloud<pcl::PointNormal>);


  // Fill in the cloud data
  pcl::PCDReader reader;
  std::string dataPath("../../data/");
  std::string pointsFilenName;

  // Replace the path below with the path where you saved your file
  pointsFilenName = dataPath + "all_clean_points_downsampled_0.05_cropped_normals_" + to_string((long double) 0.3f) + ".pcd";
  reader.read<pcl::PointNormal> (pointsFilenName, *cloud1);
  pointsFilenName = dataPath + "all_clean_points_downsampled_0.05_cropped_normals_" + to_string((long double) 0.4f) + ".pcd";
  reader.read<pcl::PointNormal> (pointsFilenName, *cloud2);
  std::cerr << "Cloud1: " << *cloud1 << std::endl;
  std::cerr << "Cloud1: " << *cloud2 << std::endl;

  GPOM::compareTwoNormals<pcl::PointNormal>(cloud1, cloud2);

  return (0);
}
#endif