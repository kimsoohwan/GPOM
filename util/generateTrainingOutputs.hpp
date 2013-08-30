#ifndef GENERATE_TRAINING_OUTPUTS_HPP
#define GENERATE_TRAINING_OUTPUTS_HPP

#include <vector>
#include <pcl/point_types.h>
#include "GP/DataTypes.hpp"

namespace GPOM {

	// PCL 1.7.0 (index removed, points out)
	template <typename PointT, typename NormalT>
	typename pcl::PointCloud<PointT>::Ptr
	extractNaNFromPointCloud(const typename pcl::PointCloud<NormalT>	&normals_in, 
							 typename pcl::PointCloud<NormalT>			&normals_out)
	{
		// ponints out
		pcl::PointCloud<PointT>::Ptr	points_out(new pcl::PointCloud<PointT>());

		// If the clouds are not the same, prepare the output
		if (&normals_in != &normals_out)
		{
			normals_out.header = normals_in.header;
			normals_out.points.resize (normals_in.points.size ());
		}

		size_t j = 0;
		for (size_t i = 0; i < normals_in.points.size (); ++i)
		{
			if (!pcl_isfinite (normals_in.points[i].normal_x) || 
				!pcl_isfinite (normals_in.points[i].normal_y) || 
				!pcl_isfinite (normals_in.points[i].normal_z) ||
				normals_in.points[i].curvature <= 0)
			{
				points_out->push_back(PointT(normals_in.points[i].x,
											 normals_in.points[i].y,
											 normals_in.points[i].z));
				continue;
			}

			normals_out.points[j] = normals_in.points[i];
			j++;
		}

		if (j != normals_in.points.size ())
		{
			// Resize to the correct size
			normals_out.points.resize (j);
		}

		normals_out.height = 1;
		normals_out.width  = static_cast<uint32_t>(j);

		return points_out;
	}

	template <typename PointT, typename NormalT> 
	typename pcl::PointCloud<PointT>::Ptr
	extractNaNNormalsFromPointCloud (const typename pcl::PointCloud<NormalT>	&normals_in, 
								     typename pcl::PointCloud<NormalT>			&normals_out,
									 const std::vector<int>						&index)
	{
		// ponints out
		pcl::PointCloud<PointT>::Ptr	points_out(new pcl::PointCloud<PointT>());

		// If the clouds are not the same, prepare the output
		if (&normals_in != &normals_out)
		{
			normals_out.header = normals_in.header;
			normals_out.points.resize (index.size ());
		}

		size_t j = 0;
		for (size_t i = 0; i < index.size (); ++i)
		{
			if (!pcl_isfinite (normals_in.points[ index[i] ].normal_x) || 
				!pcl_isfinite (normals_in.points[ index[i] ].normal_y) || 
				!pcl_isfinite (normals_in.points[ index[i] ].normal_z) ||
				normals_in.points[ index[i] ].curvature <= 0)
			{
				points_out->push_back(PointT(normals_in.points[ index[i] ].x,
											 normals_in.points[ index[i] ].y,
											 normals_in.points[ index[i] ].z));
				continue;
			}

			normals_out.points[j] = normals_in.points[ index[i] ];
			j++;
		}

		if (j != index.size ())
		{
			// Resize to the correct size
			normals_out.points.resize (j);
		}

		normals_out.height = 1;
		normals_out.width  = static_cast<uint32_t>(j);

		return points_out;
	}

	template <typename PointT, typename NormalT>
	VectorPtr generateTrainingOutputs(typename pcl::PointCloud<PointT>::ConstPtr	points,
									  typename pcl::PointCloud<NormalT>::ConstPtr	clean_normals, 									  
									  const std::vector<int>						&index)
	{
		// size
		const int nd = clean_normals->size();
		const int n = points->size();
		const int d = 3;

		VectorPtr pY(new Vector(nd*(d+1) + n));
		pY->setZero();
		for(int i = 0; i < nd; i++)
		{
			//(*pY)[i] = (Scalar) 0.f;							// F1(nd),
			(*pY)[nd*1 + i] = (*clean_normals)[i].normal_x;		// D1(nd)
			(*pY)[nd*2 + i] = (*clean_normals)[i].normal_y;		// D2(nd)
			(*pY)[nd*3 + i] = (*clean_normals)[i].normal_z;		// D3(nd)
		}
		//for(int i = 0; i < n; i++)		(*pY)[nd*(d+1) + i] = (Scalar) 0.f;			// F2(n)

		return pY;
	}

	template <typename PointT, typename NormalT>
	VectorPtr generateTrainingOutputs(typename pcl::PointCloud<PointT>::ConstPtr	points,
									  typename pcl::PointCloud<NormalT>::ConstPtr	clean_normals)
	{
		// size
		const int nd = clean_normals->size();
		const int n = points->size();
		const int d = 3;

		VectorPtr pY(new Vector(nd*(d+1) + n));
		pY->setZero();
		for(int i = 0; i < nd; i++)
		{
			//(*pY)[i] = (Scalar) 0.f;							// F1(nd),
			(*pY)[nd*1 + i] = (*clean_normals)[i].normal_x;		// D1(nd)
			(*pY)[nd*2 + i] = (*clean_normals)[i].normal_y;		// D2(nd)
			(*pY)[nd*3 + i] = (*clean_normals)[i].normal_z;		// D3(nd)
		}
		//for(int i = 0; i < n; i++)		(*pY)[nd*(d+1) + i] = (Scalar) 0.f;			// F2(n)

		return pY;
	}

}

#endif