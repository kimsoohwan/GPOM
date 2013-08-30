#ifndef MESH_GRID_HPP
#define MESH_GRID_HPP

#include <pcl/point_types.h>
#include "GP/DataTypes.hpp"

namespace GPOM{

	//template <typename PoinT>
	//typename pcl::PointCloud<PoinT>::Ptr meshGrid(const PoinT		&min,
	//											  const PoinT		&max,
	//											  const Scalar		gridSize)
	//{
	//	// point cloud
	//	pcl::PointCloud<pcl::PointXYZ>::Ptr pPoints(new pcl::PointCloud<pcl::PointXYZ>());

	//	// generate mesh grid in the order of x, y, z
	//	for(Scalar x = min.x + gridSize/(static_cast<Scalar>(2.f)); x < max.x; x += gridSize)
	//		for(Scalar y = min.y + gridSize/(static_cast<Scalar>(2.f)); y < max.y; y += gridSize)
	//			for(Scalar z = min.z + gridSize/(static_cast<Scalar>(2.f)); z < max.z; z += gridSize)
	//				pPoints->push_back(pcl::PointXYZ(x, y, z));

	//	return pPoints;
	//}

	MatrixPtr meshGrid(const Eigen::Vector3f	&min,
					   const Eigen::Vector3f	&max,
					   const Scalar				gridSize)
	{
		// exclude the max points
		const int nx = floor((max.x() - min.x()) / gridSize);
		const int ny = floor((max.y() - min.y()) / gridSize);
		const int nz = floor((max.z() - min.z()) / gridSize);
		MatrixPtr pXs(new Matrix(3, nx*ny*nz));
		//std::cout << nx << ", " << ny << ", " << nz << std::endl;

		// generate mesh grid in the order of x, y, z
		int col = 0;
		Scalar x = min.x() + gridSize/(static_cast<Scalar>(2.f));
		for(int ix = 0; ix < nx; ix++, x += gridSize)
		{
			Scalar y = min.y() + gridSize/(static_cast<Scalar>(2.f));
			for(int iy = 0; iy < ny; iy++, y += gridSize)
			{
				Scalar z = min.z() + gridSize/(static_cast<Scalar>(2.f));
				for(int iz = 0; iz < nz; iz++, z += gridSize)
				{
					(*pXs)(0, col) = x;
					(*pXs)(1, col) = y;
					(*pXs)(2, col) = z;
					col++;
				}
			}
		}

		return pXs;
	}

	template <typename PointT>
	MatrixPtr copyPoints(typename pcl::PointCloud<PointT>::ConstPtr pPoints, std::vector<int> &indexVector)
	{
		const int n = indexVector.size();
		MatrixPtr pX(new Matrix(3, n));
		for(int col = 0; col < n; col++)
		{
			(*pX)(0, col) = (*pPoints)[ indexVector[col] ].x;
			(*pX)(1, col) = (*pPoints)[ indexVector[col] ].y;
			(*pX)(2, col) = (*pPoints)[ indexVector[col] ].z;
		}
		return pX;
	}

	template <typename PointT>
	MatrixPtr copyPoints(typename pcl::PointCloud<PointT>::ConstPtr pPoints)
	{
		const int n = pPoints->size();
		MatrixPtr pX(new Matrix(3, n));
		for(int col = 0; col < n; col++)
		{
			(*pX)(0, col) = (*pPoints)[ col ].x;
			(*pX)(1, col) = (*pPoints)[ col ].y;
			(*pX)(2, col) = (*pPoints)[ col ].z;
		}
		return pX;
	}
}

#endif