#ifndef SURFACE_NORMAL_ESTIMATION_HPP
#define SURFACE_NORMAL_ESTIMATION_HPP

#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>		// for pcl::search::KdTree
#include <pcl/features/normal_3d.h>		// for pcl::NormalEstimation
#include <pcl/surface/mls.h>					// for pcl::MovingLeastSquares

namespace GPOM{

	// Normal: float normal[3], curvature
	pcl::PointCloud<pcl::Normal>::Ptr estimateSurfaceNormals(pcl::PointCloud<pcl::PointXYZ>::ConstPtr pPoints, 
																										  const pcl::PointXYZ &sensorPosition,
																										  const float searchRadius = 0.03f)
	{
		// surface normal vectors
		// Create the normal estimation class, and pass the input dataset to it
		pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;

		// Set the input points
		ne.setInputCloud(pPoints);

		// Create an empty kdtree representation, and pass it to the normal estimation object.
		// Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
		// Use a FLANN-based KdTree to perform neighborhood searches
		pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ> ());
		ne.setSearchMethod(tree);

		// Use all neighbors in a sphere of radius
		// Specify the size of the local neighborhood to use when computing the surface normals
		ne.setRadiusSearch(searchRadius);
		//normalEstimation.setKSearch (10);

		// Set the search surface (i.e., the points that will be used when search for the input points’ neighbors)
		ne.setSearchSurface(pPoints);

		// Compute the surface normals
		pcl::PointCloud<pcl::Normal>::Ptr pNormals(new pcl::PointCloud<pcl::Normal>);
		ne.compute(*pNormals);

		// check if n is consistently oriented towards the viewpoint and flip otherwise
		// angle between Psensor - Phit and Normal should be less than 90 degrees
		// dot(Psensor - Phit, Normal) > 0
		pcl::PointCloud<pcl::PointXYZ>::const_iterator		iterPoint		= pPoints->begin();
		pcl::PointCloud<pcl::Normal>::iterator					iterNormal	= pNormals->begin();
		for(; (iterPoint != pPoints->end()) && (iterNormal != pNormals->end()); iterPoint++, iterNormal++)
		{
			if((sensorPosition.x - iterPoint->x) * iterNormal->normal_x + 
			   (sensorPosition.y - iterPoint->y) * iterNormal->normal_y + 
			   (sensorPosition.z - iterPoint->z) * iterNormal->normal_z < 0)
			{
				iterNormal->normal_x *= -1.f;
				iterNormal->normal_y *= -1.f;
				iterNormal->normal_z *= -1.f;
			}
		}

		return pNormals;
	}

	// PointNormal: float x, y, znormal[3], curvature
	void smoothAndNormalEstimation(pcl::PointCloud<pcl::PointXYZ>::Ptr pPoints, 
																 pcl::PointCloud<pcl::PointNormal>::Ptr &pPointNormals)
	{
		// Smoothing and normal estimation based on polynomial reconstruction
		// Moving Least Squares (MLS) surface reconstruction method can be used to smooth and resample noisy data

		// Create a KD-Tree
		pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);

		// Init object (second point type is for the normals, even if unused)
		pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal> mls;

		// Set parameters
		mls.setInputCloud(pPoints);
		mls.setPolynomialFit(true);
		mls.setSearchMethod(kdtree);
		mls.setSearchRadius(0.03); // 0.8

		// Reconstruct
		// PCL v1.6
#if 0
		mls.setComputeNormals (true);

		// Output has the PointNormal type in order to store the normals calculated by MLS
		pcl::PointCloud<pcl::PointNormal> mls_points;
		mls.process (mls_points);
		return mls_points;
#else
		mls.reconstruct(*pPoints);

		// Output has the PointNormal type in order to store the normals calculated by MLS
		pPointNormals = mls.getOutputNormals();
		//mls.setOutputNormals(mls_points);
#endif
	}
}

#endif 