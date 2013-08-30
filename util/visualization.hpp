#ifndef POINT_CLOUD_VISUALIZATION_HPP
#define POINT_CLOUD_VISUALIZATION_HPP

#include <string>
#include <pcl/visualization/cloud_viewer.h>	// pcl::visualization::CloudViewer
#include <boost/thread/thread.hpp>

namespace GPOM {

	template <typename PointT>
	void showPointCloud(typename pcl::PointCloud<PointT>::ConstPtr cloud)
	{
		// simple viewer
		pcl::visualization::CloudViewer simple_viewer ("Simple Cloud Viewer");
		simple_viewer.showCloud(cloud);
		while (!simple_viewer.wasStopped ())
		{
			boost::this_thread::sleep (boost::posix_time::microseconds (100000));
		}
	}

	template <typename PointT>
	void showPointCloudNormals(typename pcl::PointCloud<PointT>::ConstPtr cloud, const double scale = 0.1, const std::string &name = "normals")
	{
		// visualizer
		pcl::visualization::PCLVisualizer viewer ("3D Viewer");
		viewer.setBackgroundColor (0, 0, 0);
		viewer.addText(name, 10, 10, "text");
		pcl::visualization::PointCloudColorHandlerCustom<PointT> green_color(cloud, 0, 255, 0);
		viewer.addPointCloud<PointT> (cloud, green_color, "cloud");
		viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud"); // points
		viewer.addPointCloudNormals<PointT> (cloud, 1, scale, name);	// normals

		viewer.addCoordinateSystem (1.0);
		//viewer.initCameraParameters ();
		viewer.resetCameraViewpoint(name);

		// main loop
		while (!viewer.wasStopped ())
		{
			viewer.spinOnce (100);
			boost::this_thread::sleep (boost::posix_time::microseconds (100000));
		}
	}

	template <typename PointT>
	void compareTwoPointClouds(typename pcl::PointCloud<PointT>::ConstPtr cloud1,
							   typename pcl::PointCloud<PointT>::ConstPtr cloud2,
							   const std::string &name1 = "cloud1",
							   const std::string &name2 = "cloud2")
	{
		std::cout << "compare " << name1 << " and " << name2 << std::endl;

		// visualizer
		pcl::visualization::PCLVisualizer viewer ("3D Viewer");
		viewer.setBackgroundColor (0, 0, 0);

		// comparison
		int v1(0);
		viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
		viewer.setBackgroundColor (0, 0, 0, v1);
		viewer.addText(name1, 10, 10, "v1 text", v1);
		pcl::visualization::PointCloudColorHandlerCustom<PointT> green_color(cloud1, 0, 255, 0);
		viewer.addPointCloud<PointT> (cloud1, green_color, "cloud1", v1);
		viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud1"); // points

		int v2(0);
		viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);
		viewer.setBackgroundColor (0, 0, 0, v2);
		viewer.addText(name2, 10, 10, "v2 text", v2);
		pcl::visualization::PointCloudColorHandlerCustom<PointT> red_color(cloud2, 255, 0, 0);
		viewer.addPointCloud<PointT> (cloud2, red_color, "cloud2", v2);
		viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud2"); // points

		viewer.addCoordinateSystem (5.0);
		viewer.initCameraParameters ();

		// main loop
		while (!viewer.wasStopped ())
		{
			viewer.spinOnce (100);
			boost::this_thread::sleep (boost::posix_time::microseconds (100000));
		}
	}

	template <typename PointT>
	void compareTwoNormals(typename pcl::PointCloud<PointT>::ConstPtr cloud1,
						   typename pcl::PointCloud<PointT>::ConstPtr cloud2,
						   const std::string &name1 = "cloud1",
						   const std::string &name2 = "cloud2")
	{
		std::cout << "compare " << name1 << " and " << name2 << std::endl;

		// visualizer
		pcl::visualization::PCLVisualizer viewer ("3D Viewer");
		viewer.setBackgroundColor (0, 0, 0);

		// comparison
		int v1(0);
		viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
		viewer.setBackgroundColor (0, 0, 0, v1);
		viewer.addText(name1, 10, 10, "v1 text", v1);
		pcl::visualization::PointCloudColorHandlerCustom<PointT> green_color(cloud1, 0, 255, 0);
		viewer.addPointCloud<PointT> (cloud1, green_color, "cloud1", v1);
		viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud1"); // points
		viewer.addPointCloudNormals<PointT> (cloud1, 1, 0.05, "normals1", v1);	// normals

		int v2(0);
		viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);
		viewer.setBackgroundColor (0, 0, 0, v2);
		viewer.addText(name2, 10, 10, "v2 text", v2);
		pcl::visualization::PointCloudColorHandlerCustom<PointT> red_color(cloud2, 255, 0, 0);
		viewer.addPointCloud<PointT> (cloud2, red_color, "cloud2", v2);
		viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud2"); // points
		viewer.addPointCloudNormals<PointT> (cloud2, 1, 0.05, "normals2", v2);	// normals

		viewer.addCoordinateSystem (5.0);
		viewer.initCameraParameters ();

		// main loop
		while (!viewer.wasStopped ())
		{
			viewer.spinOnce (100);
			boost::this_thread::sleep (boost::posix_time::microseconds (100000));
		}
	}
}

#endif