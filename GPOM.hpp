#ifndef GAUSSIAN_PROCESS_CONTINUOUS_OCCUPANCY_MAP_HPP
#define GAUSSIAN_PROCESS_CONTINUOUS_OCCUPANCY_MAP_HPP

#include <pcl/point_types.h>
#include "GP.hpp"

namespace GP{

template <class MeanFunc, class CovFunc, class LikFunc, 
				  template <class, class, class> class InfMethod>
class GaussianProcessOccupancyMap
{
	// constructor
	GaussianProcessOccupancyMap() { }

	// destructor
	virtual ~GaussianProcessOccupancyMap() { }

	// set point cloud
	void setPointCloud(pcl::PointCloud<pcl::PointXYZ> &cloud)
	{
	}

protected:
	GaussianProcess<MeanFunc, CovFunc, LikFunc, InfMethod> m_gp;
};

}

#endif