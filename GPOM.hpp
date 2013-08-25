#ifndef GAUSSIAN_PROCESS_CONTINUOUS_OCCUPANCY_MAP_HPP
#define GAUSSIAN_PROCESS_CONTINUOUS_OCCUPANCY_MAP_HPP

#include <fstream>

#include <pcl/point_types.h>							

#include "GP/GP.hpp"
#include "util/meshGrid.hpp"
#include "util/generateTrainingOutputs.hpp"
#include "Octree/OctreeGPOM.hpp"
#include "Octree/OctreeGPOMViewer.hpp"

namespace GPOM{

template <class MeanFunc, class CovFunc, class LikFunc, 
				  template <class, class, class> class InfMethod>
class GaussianProcessOccupancyMap : public OctreeGPOM
{
public:
	// constructor
	GaussianProcessOccupancyMap(const Scalar mapResolution)
		: OctreeGPOM(mapResolution)
	{
	}

	// destructor
	virtual ~GaussianProcessOccupancyMap() { }

	void build(pcl::PointCloud<pcl::PointXYZ>::ConstPtr			pHitPoints, 
				  pcl::PointCloud<pcl::Normal>::ConstPtr				pNormals,
				  pcl::PointCloud<pcl::PointXYZ>::ConstPtr			pRobotPositions, 
				  const float													blockSize,		  
				  const Scalar													pruneVarianceThreshold, 
				  const Scalar													pruneOccupancyThreshold)
	{
		// training inputs
		//Matrix Xd		= pHitPoints->getMatrixXfMap(3, 4, 0);
		//Matrix X			= pRobotPositions->getMatrixXfMap(3, 4, 0);
		//MatrixPtr pXd(&Xd);
		//MatrixPtr pX(&X);
		MatrixPtr pX = copyPoints(pRobotPositions);
		//std::cout << "X= " << std::endl << *pX << std::endl << std::endl;
		//MatrixPtr pXd = boost::make_shared(pHitPoints->getMatrixXfMap(3, 4, 0));
		//MatrixPtr pX	= boost::make_shared(pRobotPositions->getMatrixXfMap(3, 4, 0));

		// octree
		pcl::octree::OctreePointCloudPointVector<pcl::PointXYZ> octree(blockSize);
		//pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree(octreeResolution);

		// min, max
		pcl::PointXYZ min, max;
		pcl::getMinMax3D (*pHitPoints, min, max);

		// bounding box
		pcl::PointXYZ octreeMin, octreeMax;
		octreeMin.x = resolution_ * floor(min.x / resolution_);
		octreeMin.y = resolution_ * floor(min.y / resolution_);
		octreeMin.z = resolution_ * floor(min.z / resolution_);
		octreeMax.x = resolution_ * ceil(max.x / resolution_);
		octreeMax.y = resolution_ * ceil(max.y / resolution_);
		octreeMax.z = resolution_ * ceil(max.z / resolution_);
		octree.defineBoundingBox(octreeMin.x, octreeMin.y, octreeMin.z, octreeMax.x, octreeMax.y, octreeMax.z);

		// set point clouds
		octree.setInputCloud(pHitPoints);
		octree.addPointsFromInputCloud();
		std::cout << "total number of leaf nodes: " << octree.getLeafCount() << std::endl;

		// for each leaf node
		pcl::octree::OctreePointCloudPointVector<pcl::PointXYZ>::LeafNodeIterator iter(octree);
		//pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>::LeafNodeIterator iter(octree);

		unsigned int totalNumPoints = 0;
		unsigned int numLeafNodes = 0;
		unsigned int maxNumPoints = 0;
		unsigned int minNumPoints = 1000000;
		const int MIN_HIT_POINTS_TO_CONSIDER = 10;
		//std::ofstream fout("numPoints.txt");
		while (*++iter)
		{
			//std::cout << "# leaf node: " << numLeafNodes << std::endl;
			numLeafNodes += 1;
			totalNumPoints += n;
			if(numLeafNodes < 400) continue;
			if(numLeafNodes > 500) break;

			// points in the leaf node
			std::vector<int> indexVector;
			iter.getData(indexVector);
			const int n = indexVector.size();
			if(n <= MIN_HIT_POINTS_TO_CONSIDER) continue;
			if(n > maxNumPoints)		maxNumPoints = n;
			if(n < minNumPoints)		minNumPoints = n;
			std::cout << "[ " << numLeafNodes << " ]: " << n << std::endl;
			//fout << numLeafNodes << "\t" << n << std::endl;

			// training inputs
			//pcl::PointCloud<pcl::PointXYZ>::Ptr pHitPointsInLeafNode(new pcl::PointCloud<pcl::PointXYZ>(*pHitPoints, indexVector));
			//Matrix Xd = pHitPointsInLeafNode->getMatrixXfMap(3, 4, 0);
			//MatrixPtr pXd(&Xd);
			MatrixPtr pXd = copyPoints(pHitPoints, indexVector);
			//std::cout << "Xd = " << std::endl << *pXd << std::endl << std::endl;

			// training outputs
			VectorPtr pY = generateTrainingOutputs(pNormals, pRobotPositions, indexVector);
			//std::cout << "Y = " << std::endl << *pY << std::endl << std::endl;

			// set training data
			m_gp.setTrainingData(pXd, pX, pY);

			// hyperparameters
			MeanFunc::Hyp		meanLogHyp;
			CovFunc::Hyp		covLogHyp;
			LikFunc::Hyp		likLogHyp;

			// default values
			covLogHyp << log(5.158820125747006f), log(1.863198796627710f);
			likLogHyp << log(0.144316317935091f), log(0.000000023020410f);
			//covLogHyp << log(42.7804f), log(0.0228842f);
			//likLogHyp << log(0.0133948f), log(0.403304f);

			// train
			//std::cout << "training ... " << std::endl;
			//m_gp.train<BFGS, DeltaFunc>(meanLogHyp, covLogHyp, likLogHyp, 10);
			//std::cout << "done in seconds" << std::endl;
			//std::cout << "Mean: " << std::endl << meanLogHyp.array().exp() << std::endl;
			//std::cout << "Cov: " << std::endl << covLogHyp.array().exp() << std::endl;
			//std::cout << "Lik: " << std::endl << likLogHyp.array().exp() << std::endl;

			// test points
			Eigen::Vector3f nodeMin, nodeMax;
			octree.getVoxelBounds(iter, nodeMin, nodeMax);
			//std::cout << "nodeMin = " << std::endl << nodeMin << std::endl << std::endl;
			//std::cout << "nodeMax = " << std::endl << nodeMax << std::endl << std::endl;
			//pcl::PointCloud<pcl::PointXYZ>::Ptr pTestPoints = meshGrid(min, max, mapResolution);
			MatrixPtr pXs = meshGrid(nodeMin, nodeMax, resolution_);
			//std::cout << "number of test points: " << pXs->cols() << std::endl; // 8000
			//Matrix Xs		= pTestPoints->getMatrixXfMap(3, 4, 0);
			//MatrixPtr pXs(&Xs);
			////MatrixPtr pXs= boost::make_shared(pTestPoints->getMatrixXfMap(3, 4, 0));

			// predict
			VectorPtr	pMu;
			MatrixPtr	pSigma;
			m_gp.predict(meanLogHyp, covLogHyp, likLogHyp, pXs, 
									pMu, pSigma);

			if(pMu->hasNaN())		std::cout << "Error: Mu has NaN!" << std::endl;
			if(pSigma->hasNaN()) std::cout << "Error: Sigma has NaN!" << std::endl;

			// merge
			for(unsigned int i = 0; i < pXs->cols(); i++)
			{
				mergeMeanAndVarianceAtPoint((*pXs)(0, i), (*pXs)(1, i), (*pXs)(2, i),
													 (*pMu)(i, 0), (*pSigma)(i, 0));
			}

			// show
			//OctreeGPOMViewer viewer(pHitPoints, *this);
		}
		std::cout << "total: " << numLeafNodes << " leaf nodes." << std::endl;
		std::cout << "total: " << totalNumPoints << " points." << std::endl;
		std::cout << "max: " << maxNumPoints << std::endl;
		std::cout << "min: " << minNumPoints << std::endl;
		std::cout << "avg: " << (float) totalNumPoints / (float) numLeafNodes << std::endl;

		// prune
		std::cout << "before prune: " << getLeafCount() << " leaf nodes." << std::endl;
		unsigned int numPrunedLeafNodes = pruneCertainUnoccupiedNodes(pruneVarianceThreshold, pruneOccupancyThreshold);
		std::cout << "after prune: " << getLeafCount() << " leaf nodes. ( " << numPrunedLeafNodes << " were pruned. )" << std::endl;

		// show
		OctreeGPOMViewer viewer(pHitPoints, *this);
	

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