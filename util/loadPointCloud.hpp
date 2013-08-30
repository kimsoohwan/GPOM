#ifndef LOAD_POINT_CLOUD_HPP
#define LOAD_POINT_CLOUD_HPP

#include <iostream>
#include <string>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

namespace GPOM {

	template <typename PointT>
	typename pcl::PointCloud<PointT>::Ptr loadPointCloud(const std::string &filename)
	{
		pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
		std::cout << "loading " << filename << " ... " << std::endl;
		if (pcl::io::loadPCDFile<PointT>(filename, *cloud) == -1)
		{
			PCL_ERROR("Couldn't read file!\n");
		}
		else
		{
			std::cout << cloud->size() << " points are successfully loaded." << std::endl;
		}

		return cloud;
	}

	template <typename PointT>
	bool
	appendPointCloud(const std::string &filename, typename pcl::PointCloud<PointT>::Ptr cloud)
	{
		pcl::PointCloud<PointT>::Ptr new_cloud(new pcl::PointCloud<PointT>());
		std::cout << "loading " << filename << " ... " << std::endl;
		if (pcl::io::loadPCDFile<PointT>(filename, *new_cloud) == -1)
		{
			PCL_ERROR("Couldn't read file!\n");
			return false;
		}
		else
		{
			std::cout << cloud->size() << " + " << new_cloud->size() << " = ";
			(*cloud) += (*new_cloud);
			std::cout << cloud->size() << " points are successfully loaded." << std::endl;
			return true;
		}
	}
}

#endif