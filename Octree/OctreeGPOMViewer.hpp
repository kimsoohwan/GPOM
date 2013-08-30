#ifndef OCTREE_GPOM_VIEWER_HPP
#define OCTREE_GPOM_VIEWER_HPP

#include <boost/thread/thread.hpp>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/point_cloud_handlers.h>
#include <pcl/visualization/common/common.h>

#include "Octree/OctreeGPOM.hpp"

namespace GPOM {

template <typename PointT>
class OctreeGPOMViewer
{
public:
  OctreeGPOMViewer(typename pcl::PointCloud<PointT>::ConstPtr pCloud, 
						 OctreeGPOM &octree)
	  : m_viz ("Octree visualizator"), 
		 m_pCloud(pCloud),
		 m_octree(octree),
		 m_bDisplayOctree(true),
		 m_bDisplayPoints(true), 
		 m_bWireframe(true),
		 m_occupancyThreshold(octree.getOccupancyThreshold()),
		 m_varianceThreshold(octree.getVarianceThreshold())
  {

    //register keyboard callbacks
    m_viz.registerKeyboardCallback(&OctreeGPOMViewer::keyboardEventOccurred, *this, 0);

    //key legends
    m_viz.addText("Keys:",													 0, 185, 0.0, 1.0, 0.0, "keys_t");
    m_viz.addText("i/k -> occupancy threshold ++/-- by 0.05",	10, 170, 0.0, 1.0, 0.0, "key_ik_t");
    m_viz.addText("j/l -> variance threshold ++/-- by 0.05",	10, 155, 0.0, 1.0, 0.0, "key_jl_t");
    m_viz.addText("u/d -> displayed depth ++/--",					10, 140, 0.0, 1.0, 0.0, "key_ud_t");
    m_viz.addText("o -> Show/Hide octree",							10, 125, 0.0, 1.0, 0.0, "key_o_t"); // TODO: show octree structure
    m_viz.addText("p -> Show/Hide point cloud",						10, 110, 0.0, 1.0, 0.0, "key_p_t");
    m_viz.addText("s/w -> Surface/Wireframe representation",	10,  95, 0.0, 1.0, 0.0, "key_sw_t");

    //set current level to half the maximum one
    //m_displayedDepth = static_cast<int> (floor (m_octree.getTreeDepth() / 2.0));
    //if (m_displayedDepth == 0)      m_displayedDepth = 1;
    m_displayedDepth = m_octree.getTreeDepth();

	 // update
	 update();

    //reset camera
    m_viz.resetCameraViewpoint("cloud");

    //run main loop
    run();

  }

private:
  //========================================================

  /* \brief Callback to interact with the keyboard
   *
   */
  void keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event, void *)
  {

    if (event.getKeySym() == "i" && event.keyDown())					// Increment occupancy threshold
    {
      m_occupancyThreshold += 0.05;
		m_octree.setOccupancyThreshold(m_occupancyThreshold);
		update();
    }
    else if (event.getKeySym() == "k" && event.keyDown())			// Decrement occupancy threshold
    {
      m_occupancyThreshold -= 0.05;
		m_octree.setOccupancyThreshold(m_occupancyThreshold);
		update();
    }
    else if (event.getKeySym() == "j" && event.keyDown())			// Increment variance threshold
    {
      m_varianceThreshold += 0.05;
		m_octree.setVarianceThreshold(m_varianceThreshold);
		update();
    }
    else if (event.getKeySym() == "l" && event.keyDown())			// Decrement variance threshold
    {
      m_varianceThreshold -= 0.05;
		m_octree.setVarianceThreshold(m_varianceThreshold);
		update();
    }
	 else if (event.getKeySym() == "u" && event.keyDown())			// Increment displayed depth
    {
      IncrementLevel();
    }
    else if (event.getKeySym() == "d" && event.keyDown())			// Decrement displayed depth
    {
      DecrementLevel();
    }
    else if (event.getKeySym() == "o" && event.keyDown())			// Show/Hide octree
    {
      m_bDisplayOctree = !m_bDisplayOctree;
      update();
    }
    else if (event.getKeySym() == "p" && event.keyDown())			// Show/Hide point cloud
    {
      m_bDisplayPoints = !m_bDisplayPoints;
      update();
    }
    else if (event.getKeySym() == "w" && event.keyDown())			// Surface/Wireframe representation
    {
      if(!m_bWireframe)
        m_bWireframe=true;
      update();
    }
    else if (event.getKeySym() == "s" && event.keyDown())			// Surface/Wireframe representation
    {
      if(m_bWireframe)
        m_bWireframe=false;
      update();
    }
  }

  /* \brief Graphic loop for the viewer
   *
   */
  void run()
  {
    while (!m_viz.wasStopped())
    {
      //main loop of the visualizer
      m_viz.spinOnce(100);
      boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }
  }

  /* \brief Helper function that draw info for the user on the viewer
   *
   */
  void showLegend()
  {
	 if (m_bDisplayOctree)
	 {
		 char dataDisplay[256];
		 sprintf(dataDisplay, "Displaying data as %s with occupancy threshold = %f, variance threshold = %f", (m_bWireframe) ? ("Wireframes") : ("CUBES"), m_occupancyThreshold, m_varianceThreshold);
		 m_viz.removeShape("disp_t");
		 m_viz.addText(dataDisplay, 0, 60, 1.0, 0.0, 0.0, "disp_t");
	 }

    char level[256];
    sprintf(level, "Displayed depth is %d on %d", m_displayedDepth, m_octree.getTreeDepth());
    m_viz.removeShape("level_t1");
    m_viz.addText(level, 0, 45, 1.0, 0.0, 0.0, "level_t1");

    m_viz.removeShape("level_t2");
    sprintf(level, "Voxel size: %.4fm [%zu / %zu voxels]", sqrt(m_octree.getVoxelSquaredSideLen(m_displayedDepth)),
																				m_numVoxelsOccupied, m_octree.getLeafCount());
    m_viz.addText(level, 0, 30, 1.0, 0.0, 0.0, "level_t2");

    m_viz.removeShape("org_t");
    if (m_bDisplayPoints)
      m_viz.addText("Displaying original cloud", 0, 15, 1.0, 0.0, 0.0, "org_t");
  }

  /* \brief Visual update. Create visualizations and add them to the viewer
   *
   */
  void update()
  {
    //remove existing shapes from visualizer
    clearView();

	 // show octree
    if(m_bDisplayOctree)
		 showCubes();

	 // show point cloud
	 if(m_bDisplayPoints)
	 {
        //add original cloud in visualizer
        pcl::visualization::PointCloudColorHandlerGenericField<PointT> color_handler(m_pCloud, "z");
        m_viz.addPointCloud(m_pCloud, color_handler, "cloud");
	 }

    showLegend();
  }

  /* \brief remove dynamic objects from the viewer
   *
   */
  void clearView()
  {
    //remove cubes if any
    vtkRenderer *renderer = m_viz.getRenderWindow()->GetRenderers()->GetFirstRenderer();
    while (renderer->GetActors()->GetNumberOfItems() > 0)
      renderer->RemoveActor(renderer->GetActors()->GetLastActor());
    //remove point clouds if any
    m_viz.removePointCloud("cloud");
  }

  /* \brief Create a vtkSmartPointer object containing a cube
   *
   */
  vtkSmartPointer<vtkPolyData> GetCuboid(double minX, double maxX, double minY, double maxY, double minZ, double maxZ)
  {
    vtkSmartPointer<vtkCubeSource> cube = vtkSmartPointer<vtkCubeSource>::New();
    cube->SetBounds(minX, maxX, minY, maxY, minZ, maxZ);
    return cube->GetOutput();
  }

  /* \brief display octree cubes via vtk-functions
   *
   */
  void showCubes()
  {
    //get the renderer of the visualizer object
    vtkRenderer *renderer = m_viz.getRenderWindow()->GetRenderers()->GetFirstRenderer();

    vtkSmartPointer<vtkAppendPolyData> treeWireframe = vtkSmartPointer<vtkAppendPolyData>::New();
 
#if 1
	double voxelSideLen = sqrt(m_octree.getVoxelSquaredSideLen());
	double s = voxelSideLen / 2.0;

	OctreeGPOM::LeafNodeIterator iter(m_octree);
	pcl::PointXYZ center;
	Scalar occupancy;
	while(*++iter)
	{
		// query occupancy
		if(m_octree.isVoxelOccupiedAtLeafNode(iter, center))
			treeWireframe->AddInput(GetCuboid(center.x - s, center.x + s, 
														 center.y - s, center.y + s, 
														 center.z - s, center.z + s));
	}
#else
	 // iterate the octree
    OctreeGPOM::Iterator tree_it(m_octree);
	 while(*tree_it++)
	 {
		 // skip higher nodes
      if (static_cast<int> (tree_it.getCurrentOctreeDepth ()) != m_displayedDepth)
        continue;

		// remember current node
      Eigen::Vector3f voxel_min, voxel_max;
      m_octree.getVoxelBounds(tree_it, voxel_min, voxel_max);

		// search for an occupied child
		OctreeGPOM::Iterator sub_tree_it = tree_it;
		while(*sub_tree_it++)
		{
			// if it goes to higher, stop
			if (static_cast<int> (sub_tree_it.getCurrentOctreeDepth ()) < m_displayedDepth)
				continue;
			
			// first occupied child
			if(m_octree.isVoxelOccupiedAtNode(sub_tree_it))
			{
				// draw cube
				treeWireframe->AddInput(GetCuboid(voxel_min.x(), voxel_max.x(), voxel_min.y(), voxel_max.y(), voxel_min.z(), voxel_max.z()));
				break;
			}
		}

      //we are already the desired depth, there is no reason to go deeper.
      tree_it.skipChildVoxels();
	 }
#endif

    vtkSmartPointer<vtkActor> treeActor = vtkSmartPointer<vtkActor>::New();

    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInput(treeWireframe->GetOutput());
    treeActor->SetMapper(mapper);

    treeActor->GetProperty()->SetColor(1.0, 1.0, 1.0);
    treeActor->GetProperty()->SetLineWidth(2);
    if(m_bWireframe)
    {
      treeActor->GetProperty()->SetRepresentationToWireframe();
      treeActor->GetProperty()->SetOpacity(0.35);
    }
    else
      treeActor->GetProperty()->SetRepresentationToSurface();

    renderer->AddActor(treeActor);
  }

  /* \brief Helper function to increase the octree display level by one
   *
   */
  bool IncrementLevel()
  {
    if (m_displayedDepth < static_cast<int> (m_octree.getTreeDepth ()))
    {
      m_displayedDepth++;
      update();
      return true;
    }
    else
      return false;
  }

  /* \brief Helper function to decrease the octree display level by one
   *
   */
  bool DecrementLevel()
  {
    if (m_displayedDepth > 0)
    {
      m_displayedDepth--;
      update();
      return true;
    }
    return false;
  }

  private:
  //========================================================
  // PRIVATE ATTRIBUTES
  //========================================================
  //visualizer
  pcl::PointCloud<pcl::PointXYZ>::Ptr xyz;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr xyz_rgb;

  pcl::visualization::PCLVisualizer m_viz;

  //original cloud
  typename pcl::PointCloud<PointT>::ConstPtr m_pCloud;

  //octree
  OctreeGPOM &m_octree;

  //level
  int m_displayedDepth;

  //bool to decide if we display points or cubes
  bool m_bDisplayOctree;
  bool m_bDisplayPoints;
  bool m_bWireframe;

  int m_numVoxelsOccupied;

  Scalar m_occupancyThreshold;
  Scalar m_varianceThreshold;

};

}

#endif