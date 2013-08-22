#ifndef GAUSSIAN_PROCESS_CONTINUOUS_OCCUPANCY_MAP_HPP
#define GAUSSIAN_PROCESS_CONTINUOUS_OCCUPANCY_MAP_HPP

//#include <Eigen/Core>

#include <pcl/point_types.h>							
#include <pcl/octree/octree.h>
#include <pcl/octree/octree_impl.h> 

#include "GP/GP.hpp"
#include "util/meshGrid.hpp"
#include "util/generateTrainingOutputs.hpp"

namespace GPOM{

template <class MeanFunc, class CovFunc, class LikFunc, 
				  template <class, class, class> class InfMethod>
class GaussianProcessOccupancyMap
{
public:
	// constructor
	GaussianProcessOccupancyMap() { }

	// destructor
	virtual ~GaussianProcessOccupancyMap() { }

	// set point cloud
	//void setPointCloud(pcl::PointCloud<pcl::PointXYZ> &cloud)
	//{
	//}

	void build(pcl::PointCloud<pcl::PointXYZ>::ConstPtr			pHitPoints, 
					  pcl::PointCloud<pcl::Normal>::ConstPtr				pNormals,
					  pcl::PointCloud<pcl::PointXYZ>::ConstPtr			pRobotPositions, 
					  const float																	mapResolution,
					  const float																	octreeResolution)
	{
		// min, max
		pcl::PointXYZ min, max;
		pcl::getMinMax3D (*pHitPoints, min, max);
		std::cout << "min = " << min << std::endl;
		std::cout << "max = " << max << std::endl;

		// bounding box
		pcl::PointXYZ octreeMin, octreeMax;
		octreeMin.x = octreeResolution * floor(min.x / octreeResolution);
		octreeMin.y = octreeResolution * floor(min.y / octreeResolution);
		octreeMin.z = octreeResolution * floor(min.z / octreeResolution);
		octreeMax.x = octreeResolution * ceil(max.x / octreeResolution);
		octreeMax.y = octreeResolution * ceil(max.y / octreeResolution);
		octreeMax.z = octreeResolution * ceil(max.z / octreeResolution);
		std::cout << "octreeMin = " << octreeMin << std::endl;
		std::cout << "octreeMax = " << octreeMax << std::endl;

		// build
		build(pHitPoints, pNormals, pRobotPositions, octreeMin, octreeMax, mapResolution, octreeResolution);
	}

	void build(pcl::PointCloud<pcl::PointXYZ>::ConstPtr			pHitPoints, 
					  pcl::PointCloud<pcl::Normal>::ConstPtr				pNormals,
					  pcl::PointCloud<pcl::PointXYZ>::ConstPtr			pRobotPositions, 
					  const pcl::PointXYZ													&octreeMin,
					  const pcl::PointXYZ													&octreeMax,
					  const float																	mapResolution,
					  const float																	octreeResolution)
	{
		// training inputs
		//Matrix Xd		= pHitPoints->getMatrixXfMap(3, 4, 0);
		//Matrix X			= pRobotPositions->getMatrixXfMap(3, 4, 0);
		//MatrixPtr pXd(&Xd);
		//MatrixPtr pX(&X);
		MatrixPtr pX = copyPoints(pRobotPositions);
		std::cout << "X= " << std::endl << *pX << std::endl << std::endl;
		//MatrixPtr pXd = boost::make_shared(pHitPoints->getMatrixXfMap(3, 4, 0));
		//MatrixPtr pX	= boost::make_shared(pRobotPositions->getMatrixXfMap(3, 4, 0));

		// training outputs
		//const int nd		= pXd->cols();
		//const int n		= pX->cols();
		//const int d		= 3;
		//std::cout << "number of surface normals = " << nd << std::endl;
		//std::cout << "nnumber of points = " << n << std::endl;
		//std::cout << "dimensions = " << d << std::endl;
		//VectorPtr pY(new Vector(nd*(d+1) + n));
		//pY->setZero();
		//for(int i = 0; i < nd; i++)
		//{
		//	//(*pY)[i] = (Scalar) 0.f;											// F1(nd),
		//	(*pY)[nd*1 + i] = (*pNormals)[i].normal_x;			// D1(nd)
		//	(*pY)[nd*2 + i] = (*pNormals)[i].normal_y;			// D2(nd)
		//	(*pY)[nd*3 + i] = (*pNormals)[i].normal_z;			// D3(nd)
		//}
		////for(int i = 0; i < n; i++)		(*pY)[nd*(d+1) + i] = (Scalar) 0.f;			// F2(n)

		// octree
		pcl::octree::OctreePointCloudPointVector<pcl::PointXYZ> octree(octreeResolution);
		octree.defineBoundingBox(octreeMin.x, octreeMin.y, octreeMin.z, octreeMax.x, octreeMax.y, octreeMax.z);
		//pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree(octreeResolution);
		octree.setInputCloud(pHitPoints);
		octree.addPointsFromInputCloud();
		std::cout << "number of leaf nodes: " << octree.getLeafCount() << std::endl;

		// for each leaf node
		pcl::octree::OctreePointCloudPointVector<pcl::PointXYZ>::LeafNodeIterator iter(octree);
		//pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>::LeafNodeIterator iter(octree);
		unsigned int totalNumPoints = 0;
		unsigned int numLeafNodes = 0;
		unsigned int maxNumPoints = 0;
		unsigned int minNumPoints = 1000000;
		while (*++iter)
		{
			//std::cout << "# leaf node: " << numLeafNodes << std::endl;

			// points in the leaf node
			std::vector<int> indexVector;
			iter.getData(indexVector);
			const int n = indexVector.size();
			if(n <= 0) continue;
			std::cout << "# leaf node: " << numLeafNodes << std::endl;
			std::cout << "# hit points: " << n << std::endl;
			numLeafNodes += 1;
			totalNumPoints += n;
			if(n > maxNumPoints)		maxNumPoints = n;
			if(n < minNumPoints)		minNumPoints = n;

			// training inputs
			//pcl::PointCloud<pcl::PointXYZ>::Ptr pHitPointsInLeafNode(new pcl::PointCloud<pcl::PointXYZ>(*pHitPoints, indexVector));
			//Matrix Xd = pHitPointsInLeafNode->getMatrixXfMap(3, 4, 0);
			//MatrixPtr pXd(&Xd);
			MatrixPtr pXd = copyPoints(pHitPoints, indexVector);
			std::cout << "Xd = " << std::endl << *pXd << std::endl << std::endl;

			// training outputs
			VectorPtr pY = generateTrainingOutputs(pNormals, pRobotPositions, indexVector);
			std::cout << "Y = " << std::endl << *pY << std::endl << std::endl;

			// set training data
			m_gp.setTrainingData(pXd, pX, pY);

			// hyperparameters
			MeanFunc::Hyp		meanLogHyp;
			CovFunc::Hyp		covLogHyp;
			LikFunc::Hyp			likLogHyp;

			// default values
			covLogHyp << log(1.f), log(1.f);
			likLogHyp << log(1.f), log(1.f);
			//covLogHyp << log(42.7804f), log(0.0228842f);
			//likLogHyp << log(0.0133948f), log(0.403304f);

			// train
			//std::cout << "training ... " << std::endl;
			m_gp.train<BFGS, DeltaFunc>(meanLogHyp, covLogHyp, likLogHyp, 10);
			//std::cout << "done in seconds" << std::endl;

			std::cout << "Mean: " << std::endl << meanLogHyp.array().exp() << std::endl;
			std::cout << "Cov: " << std::endl << covLogHyp.array().exp() << std::endl;
			std::cout << "Lik: " << std::endl << likLogHyp.array().exp() << std::endl;

			// test points
			Eigen::Vector3f nodeMin, nodeMax;
			octree.getVoxelBounds(iter, nodeMin, nodeMax);
			//std::cout << "nodeMin = " << std::endl << nodeMin << std::endl << std::endl;
			//std::cout << "nodeMax = " << std::endl << nodeMax << std::endl << std::endl;
			//pcl::PointCloud<pcl::PointXYZ>::Ptr pTestPoints = meshGrid(min, max, mapResolution);
			MatrixPtr pXs = meshGrid(nodeMin, nodeMax, mapResolution);
			//Matrix Xs		= pTestPoints->getMatrixXfMap(3, 4, 0);
			//MatrixPtr pXs(&Xs);
			////MatrixPtr pXs= boost::make_shared(pTestPoints->getMatrixXfMap(3, 4, 0));

			//// predict
			VectorPtr	pMu;
			MatrixPtr		pSigma;
			m_gp.predict(meanLogHyp, covLogHyp, likLogHyp, pXs, 
									pMu, pSigma);
			*/
		}
		std::cout << "total: " << numLeafNodes << " leaf nodes." << std::endl;
		std::cout << "total: " << totalNumPoints << " points." << std::endl;
		std::cout << "max: " << maxNumPoints << std::endl;
		std::cout << "min: " << minNumPoints << std::endl;
		std::cout << "avg: " << (float) totalNumPoints / (float) numLeafNodes << std::endl;
	

		//// set training data
		//m_gp.setTrainingData(pXd, pX, pY);

		//// hyperparameters
		//MeanFunc::Hyp		meanLogHyp;
		//CovFunc::Hyp		covLogHyp;
		//LikFunc::Hyp			likLogHyp;

		//// default values
		//covLogHyp << log(1.f), log(1.f);
		//likLogHyp << log(1.f), log(1.f);

		//// train
		//std::cout << "training ... " << std::endl;
		//m_gp.train<BFGS, DeltaFunc>(meanLogHyp, covLogHyp, likLogHyp, 10);
		//std::cout << "done in seconds" << std::endl;

		//std::cout << "Mean: " << std::endl << meanLogHyp << std::endl;
		//std::cout << "Cov: " << std::endl << covLogHyp << std::endl;
		//std::cout << "Lik: " << std::endl << likLogHyp << std::endl;

		//// test points
		//pcl::PointCloud<pcl::PointXYZ>::Ptr pTestPoints = meshGrid(min, max, mapResolution);
		//Matrix Xs		= pTestPoints->getMatrixXfMap(3, 4, 0);
		//MatrixPtr pXs(&Xs);
		////MatrixPtr pXs= boost::make_shared(pTestPoints->getMatrixXfMap(3, 4, 0));

		//// predict
		//VectorPtr	pMu;
		//MatrixPtr		pSigma;
		//m_gp.predict(meanLogHyp, covLogHyp, likLogHyp, pXs, 
		//						pMu, pSigma);
	}

	// create a continuous occupancy map
	void build(pcl::PointCloud<pcl::PointXYZ>::ConstPtr			pPoints, 
					  pcl::PointCloud<pcl::Normal>::ConstPtr				pNormals,
					  const float																	mapResolution, 
					  const float																	octreeResolution)
	{
		// store in an octree
		pcl::octree::OctreePointCloud<pcl::PointXYZ> octree(octreeResolution);
		octree.setInputCloud(pPoints);
		octree.addPointsFromInputCloud();

		// instantiate iterator for octreeA
		pcl::octree::OctreePointCloud<pcl::PointXYZ>::LeafNodeIterator leafNodeIter(octree);

		// for each leaf node
		std::vector<int> indexVector;
		unsigned int totalNumPoints = 0;
		while (*++leafNodeIter)
		{
			leafNodeIter.getData (indexVector);
			std::cout << indexVector.size() << std::endl;
			totalNumPoints += indexVector.size();
			indexVector.clear();
		}
		std::cout << "In total, " << totalNumPoints << " points" << std::endl;

		// occupied voxels
		pcl::octree::OctreePointCloud<pcl::PointXYZ>::AlignedPointTVector voxelCenterList;
		octree.getOccupiedVoxelCenters(voxelCenterList);
		pcl::octree::OctreePointCloud<pcl::PointXYZ>::AlignedPointTVector::const_iterator voxelCenterIter;
		for(voxelCenterIter = voxelCenterList.begin(); voxelCenterIter != voxelCenterList.end(); voxelCenterIter++)
		{
			if(octree.isVoxelOccupiedAtPoint (*voxelCenterIter))		std::cout << "occupied: ";
			else																							std::cout << "empty: ";
			std::cout << *voxelCenterIter << std::endl;
		}
	}


protected:
	GaussianProcess<MeanFunc, CovFunc, LikFunc, InfMethod> m_gp;
};

}

#endif