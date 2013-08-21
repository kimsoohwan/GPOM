#ifndef MESH_GRID_HPP
#define MESH_GRID_HPP

#include <pcl/point_types.h>
#include "GP/DataTypes.hpp"

namespace GPOM{

	pcl::PointCloud<pcl::PointXYZ>::Ptr meshGrid(const pcl::PointXYZ		&min,
																					const pcl::PointXYZ		&max,
																					const Scalar					gridSize)
	{
		// point cloud
		pcl::PointCloud<pcl::PointXYZ>::Ptr pPoints(new pcl::PointCloud<pcl::PointXYZ>());

		// generate mesh grid in the order of x, y, z
		for(Scalar x = min.x; x < max.x; x += gridSize)
			for(Scalar y = min.y; y < max.y; y += gridSize)
				for(Scalar z = min.z; z < max.z; z += gridSize)
					pPoints->push_back(pcl::PointXYZ(x, y, z));

		return pPoints;
	}
}

#endif