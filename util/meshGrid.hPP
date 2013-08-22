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

	MatrixPtr meshGrid(Eigen::Vector3f				&min,
									   Eigen::Vector3f				&max,
									   const Scalar					gridSize)
	{
		// exclude the max points
		const int nx = floor((max.x() - min.x() - EPSILON) / gridSize);
		const int ny = floor((max.y() - min.y() - EPSILON) / gridSize);
		const int nz = floor((max.z() - min.z() - EPSILON) / gridSize);
		MatrixPtr pXs(new Matrix(3, nx+ny+nz));
		//std::cout << nx << ", " << ny << ", " << nz << std::endl;

		// generate mesh grid in the order of x, y, z
		int col = 0;
		for(int ix = 0; ix < nx; ix++)
		{
			const Scalar x = min.x() + ((Scalar) ix) * gridSize;
			for(int iy = 0; iy < ny; iy++)
			{
				const Scalar y = min.y() + ((Scalar) iy) * gridSize;
				for(int iz = 0; iz < nz; iz++)
				{
					const Scalar z = min.z() + ((Scalar) iz) * gridSize;
					(*pXs)(0, col) = x;
					(*pXs)(1, col) = y;
					(*pXs)(2, col) = z;
					col++;
				}
			}
		}

		return pXs;
	}

	MatrixPtr copyPoints(pcl::PointCloud<pcl::PointXYZ>::ConstPtr pPoints, std::vector<int> &indexVector)
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

	MatrixPtr copyPoints(pcl::PointCloud<pcl::PointXYZ>::ConstPtr pPoints)
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