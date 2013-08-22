#ifndef GENERATE_TRAINING_OUTPUTS_HPP
#define GENERATE_TRAINING_OUTPUTS_HPP

#include <vector>
#include <pcl/point_types.h>
#include "GP/DataTypes.hpp"

namespace GPOM {

	VectorPtr generateTrainingOutputs(pcl::PointCloud<pcl::Normal>::ConstPtr				pNormals, 
																 pcl::PointCloud<pcl::PointXYZ>::ConstPtr			pRobotPositions,
																 const std::vector<int>												&indexVector)
	{
		// size
		const int nd = indexVector.size();
		const int n = pRobotPositions->size();
		const int d = 3;

		VectorPtr pY(new Vector(nd*(d+1) + n));
		pY->setZero();
		for(int i = 0; i < nd; i++)
		{
			//(*pY)[i] = (Scalar) 0.f;											// F1(nd),
			(*pY)[nd*1 + i] = (*pNormals)[ indexVector[i] ].normal_x;			// D1(nd)
			(*pY)[nd*2 + i] = (*pNormals)[ indexVector[i] ].normal_y;			// D2(nd)
			(*pY)[nd*3 + i] = (*pNormals)[ indexVector[i] ].normal_z;			// D3(nd)
		}
		//for(int i = 0; i < n; i++)		(*pY)[nd*(d+1) + i] = (Scalar) 0.f;			// F2(n)

		return pY;
	}

}

#endif