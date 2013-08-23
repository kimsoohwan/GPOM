#ifndef OCTREE_FOR_GAUSSIAN_PROCESS
#define OCTREE_FOR_GAUSSIAN_PROCESS

#include <pcl/octree/octree_base.h>
#include <pcl/octree/octree2buf_base.h>
#include <pcl/octree/octree_lowmemory_base.h>

#include <pcl/point_types.h>

#include <pcl/octree/octree_nodes.h>
#include <pcl/octree/octree_iterator.h>

namespace GPOM {

	class GaussianDistribution
	{
	public:
		GaussianDistribution()
			: mean_((Scalar) 0.f), 
			  inverseVariance_((Scalar) 0.f)
		{
		}

		GaussianDistribution(const Scalar mean, const Scalar inverseVariance)
			: mean_(mean), 
			  inverseVariance_(inverseVariance)
		{
		}

		GaussianDistribution& operator=(const GaussianDistribution &rhs)
		{
			// Only do assignment if RHS is a different object from this.
			if (this != &rhs)
			{
				mean_						= rhs.mean_;
				inverseVariance_		= rhs.inverseVariance_;
			}
			return *this;
		}

		// merge
		GaussianDistribution& operator+=(const GaussianDistribution &rhs)
		{
			mean_						= inverseVariance_ * mean_ + rhs.inverseVariance_ * rhs.mean_;
			inverseVariance_		+= rhs.inverseVariance_;
			mean_						/= inverseVariance_;
			return *this;
		}

		Scalar			mean_;
		Scalar			inverseVariance_;
	};

	class OctreeGPOMLeaf : public pcl::octree::OctreeLeafAbstract<GaussianDistribution>
    {
    public:
      /** \brief Class initialization. */
      OctreeGPOMLeaf ()
      {
      }

      /** \brief Empty class deconstructor. */
      ~OctreeGPOMLeaf ()
      {
      }

      /** \brief deep copy function */
      virtual OctreeNode *
      deepCopy () const
      {
        return (OctreeNode*) new OctreeGPOMLeaf (*this);
      }

      /** \brief Read input data. Only an internal counter is increased.
       * /param Gaussian_arg: input Gaussian distribution
       *  */
      virtual void
      setData (const DataT& data_arg)
      {
		  m_Gaussian += data_arg;
      }

      /** \brief Returns a null pointer as this leaf node does not store any data.
       *  \param data_arg: reference to return pointer of leaf node DataT element (will be set to 0).
       */
      virtual void
      getData (const DataT*& data_arg) const
      {
        data_arg = m_Gaussian;
      }

      /** \brief Empty getData data vector implementation as this leaf node does not store any data. \
       *  \param dataVector_arg: reference to dummy DataT vector that is extended with leaf node DataT elements.
       */
      virtual void
      getData (std::vector<DataT>& dataVector_arg) const
      {
      }

      /** \brief Return mean.
       *  \return mean
       * */
      Scalar
      getMean() const
      {
        return m_Gaussian.mean_;
      }

      /** \brief Return inverse variance.
       *  \return sigma
       * */
      Scalar
      getInverseVariance() const
      {
        return m_Gaussian.inverseVariance_;
      }

      /** \brief Empty reset leaf node implementation as this leaf node does not store any data. */
      virtual void
      reset ()
      {
		m_Gaussian.mean_						= (Scalar) 0.f;
		m_Gaussian.inverseVariance_	= (Scalar) 0.f;
      }

    private:
		GaussianDistribution	m_Gaussian;
    };

	template<typename PointT, typename LeafT = OctreeGPOMLeaf, typename OctreeT = pcl::octree::OctreeBase<int, LeafT> >
    class OctreeGPOM : public OctreeT
    {
      // iterators are friends
      friend class pcl::octree::OctreeIteratorBase<int, LeafT, OctreeT> ;
      friend class pcl::octree::OctreeDepthFirstIterator<int, LeafT, OctreeT> ;
      friend class pcl::octree::OctreeBreadthFirstIterator<int, LeafT, OctreeT> ;
      friend class pcl::octree::OctreeLeafNodeIterator<int, LeafT, OctreeT> ;

      public:
        typedef OctreeT																												Base;
        typedef typename OctreeT::OctreeLeaf																		OctreeLeaf;

        // Octree iterators
        typedef pcl::octree::OctreeDepthFirstIterator<int, LeafT, OctreeT>						Iterator;
        typedef const pcl::octree::OctreeDepthFirstIterator<int, LeafT, OctreeT>				ConstIterator;

        typedef pcl::octree::OctreeLeafNodeIterator<int, LeafT, OctreeT>							LeafNodeIterator;
        typedef const pcl::octree::OctreeLeafNodeIterator<int, LeafT, OctreeT>				ConstLeafNodeIterator;

        typedef pcl::octree::OctreeDepthFirstIterator<int, LeafT, OctreeT>						DepthFirstIterator;
        typedef const pcl::octree::OctreeDepthFirstIterator<int, LeafT, OctreeT>				ConstDepthFirstIterator;
        typedef pcl::octree::OctreeBreadthFirstIterator<int, LeafT, OctreeT>						BreadthFirstIterator;
        typedef const pcl::octree::OctreeBreadthFirstIterator<int, LeafT, OctreeT>			ConstBreadthFirstIterator;

        /** \brief Octree pointcloud constructor.
          * \param[in] resolution_arg octree resolution at lowest octree level
          */
        OctreeGPOM (const double resolution_arg)
			: OctreeT (), resolution_ (resolution), 
			  minX_ (0.0f), maxX_ (resolution), 
			  minY_ (0.0f), maxY_ (resolution), 
			  minZ_ (0.0f), maxZ_ (resolution), 
			  maxKeys_ (1), boundingBoxDefined_ (false).
			  inverseVarianceThreshold_((Scalar) 0.1f),
			  occuapancyThreshold_((Scalar) 0.5f),
			  PLSC_mean_((Scalar) 0.05f),
			  PLSC_variance_((Scalar) 0.0001f)
		{
			assert ( resolution > 0.0f );
			input_ = PointCloudConstPtr ();
		}


        /** \brief Empty deconstructor. */
        virtual
        ~OctreeGPOM ()
		{
		}

        // public typedefs for single/double buffering
		typedef OctreeGPOM<PointT, LeafT, pcl::octree::OctreeBase<int, LeafT> >						SingleBuffer;
        typedef OctreeGPOM<PointT, LeafT, pcl::octree::Octree2BufBase<int, LeafT> >				DoubleBuffer;
        typedef OctreeGPOM<PointT, LeafT, pcl::octree::OctreeLowMemBase<int, LeafT> >		LowMem;

        // Boost shared pointers
        typedef boost::shared_ptr<OctreeGPOM<PointT, LeafT, OctreeT> >									Ptr;
        typedef boost::shared_ptr<const OctreeGPOM<PointT, LeafT, OctreeT> >						ConstPtr;

        // Eigen aligned allocator
        typedef std::vector<PointT, Eigen::aligned_allocator<PointT> >									AlignedPointTVector;

        /** \brief Set/change the octree voxel resolution
          * \param[in] resolution_arg side length of voxels at lowest tree level
          */
        inline void
        setResolution (double resolution_arg)
        {
			// octree needs to be empty to change its resolution
			assert( this->leafCount_ == 0 );

			resolution_ = resolution_arg;

			getKeyBitSize();
        }

        /** \brief Get octree voxel resolution
          * \return voxel resolution at lowest tree level
          */
        inline double
        getResolution () const
        {
			return (resolution_);
        }

        /** \brief Get the maximum depth of the octree.
         *  \return depth_arg: maximum depth of octree
         * */
        inline unsigned int
        getTreeDepth () const
        {
			return this->octreeDepth_;
        }

        /** \brief Set/change the inverse variance threshold
          * \param[in] inverseVarianceThreshold_arg inverse variance threshold
          */
        inline void
        setInverseVarianceThreshold(const Scalar inverseVarianceThreshold_arg)
        {
			inverseVarianceThreshold_ = inverseVarianceThreshold_arg;
        }

        /** \brief Get the inverse variance threshold
          * \return inverse variance threshold
          */
        inline Scalar
        getInverseVarianceThreshold() const
        {
			return inverseVarianceThreshold_;
        }

       /** \brief Set/change the inverse variance threshold
          * \param[in] variance_arg variance threshold
          */
        inline void
        setOccupancyThreshold(const Scalar occupancyThreshold_arg)
        {
			occuapancyThreshold_ = occupancyThreshold_arg;
        }

        /** \brief Get the variance threshold
          * \return variance threshold
          */
        inline Scalar
        getOccupancyThreshold() const
        {
			return occuapancyThreshold_;
        }

       /** \brief Set/change the probabilistic least squre classification parameters
          * \param[in] PLSC_mean_arg mean
          * \param[in] PLSC_variance_arg variance
          */
        inline void
        setPLSCParameters(const Scalar PLSC_mean_arg, const Scalar PLSC_variance_arg)
        {
			PLSC_mean_		= PLSC_mean_arg;
			PLSC_variance_	= PLSC_variance_arg;
        }

        /** \brief Get the probabilistic least squre classification parameters
          * \param[out] PLSC_mean_arg mean
          * \param[out] PLSC_variance_arg variance
          */
        inline void
        getOccupancyThreshold(Scalar &PLSC_mean_arg, Scalar &PLSC_variance_arg) const
        {
			PLSC_mean_arg			= PLSC_mean_;
			PLSC_variance_arg		= PLSC_variance_;
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Merge
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        /** \brief Merge mean and inverse variance at a given point. If there is no leaf node at the point, create one.
          * \param[in] point_arg position
          * \param[in] mean mean
          * \param[in] inverseVariance inverse variance
          */
        void
		mergeMeanAndInverseVarianceAtPoint(const PointT &point_arg, const Scalar mean, const Scalar inverseVariance)
		{
			// make sure bounding box is big enough
			adoptBoundingBoxToPoint (point_arg);

			// generate key
			OctreeKey key;
			genOctreeKeyforPoint (point, key);

			// merge the gaussian process to octree at key
			this->add(key, GaussianDistribution(mean, inverseVariance));

			// prune
			if(LeafT* pLeaf = this->findLeaf(key) && pLeaf->getInverseVariance() < inverseVarianceThreshold_)
			{
				removeLeaf (key);
			}
		}


        /** \brief Check if voxel at given point exist.
          * \param[in] point_arg point to be checked
          * \return "true" if voxel exist; "false" otherwise
          */
        bool
        isVoxelOccupiedAtPoint (const PointT& point_arg) const
		{
			OctreeKey key;

			// generate key for point
			this->genOctreeKeyforPoint (point_arg, key);

			return isVoxelOccupiedAtKey(key);
		}


        /** \brief Check if voxel at given point coordinates exist.
          * \param[in] pointX_arg X coordinate of point to be checked
          * \param[in] pointY_arg Y coordinate of point to be checked
          * \param[in] pointZ_arg Z coordinate of point to be checked
          * \return "true" if voxel exist; "false" otherwise
          */
        bool
        isVoxelOccupiedAtPoint (const double pointX_arg, const double pointY_arg, const double pointZ_arg) const
		{
			OctreeKey key;

			// generate key for point
			this->genOctreeKeyforPoint (pointX_arg, pointY_arg, pointZ_arg, key);

			return isVoxelOccupiedAtKey(key);
		}


        /** \brief Get a PointT vector of centers of all occupied voxels.
          * \param[out] voxelCenterList_arg results are written to this vector of PointT elements
          * \return number of occupied voxels
          */
        int
        getOccupiedVoxelCenters (AlignedPointTVector &voxelCenterList_arg) const
		{
			OctreeKey key;
			key.x = key.y = key.z = 0;

			voxelCenterList_arg.clear ();

			return getOccupiedVoxelCentersRecursive (this->rootNode_, key, voxelCenterList_arg);
		}

        /** \brief Get a PointT vector of centers of voxels intersected by a line segment.
          * This returns a approximation of the actual intersected voxels by walking
          * along the line with small steps. Voxels are ordered, from closest to
          * furthest w.r.t. the origin.
          * \param[in] origin origin of the line segment
          * \param[in] end end of the line segment
          * \param[out] voxel_center_list results are written to this vector of PointT elements
          * \param[in] precision determines the size of the steps: step_size = octree_resolution x precision
          * \return number of intersected voxels
          */
        int
        getApproxIntersectedVoxelCentersBySegment (const Eigen::Vector3f& origin, const Eigen::Vector3f& end,
																						   AlignedPointTVector &voxel_center_list,
																						   float precision = 0.2)
		{
			Eigen::Vector3f direction = end - origin;
			float norm = direction.norm ();
			direction.normalize ();

			const float step_size = (const float)resolution_ * precision;

			// Ensure we get at least one step for the first voxel.
			const int nsteps = std::max (1, (int) (norm / step_size));

			OctreeKey prev_key;
			prev_key.x = prev_key.y = prev_key.z = -1;

			// Walk along the line segment with small steps.
			for (int i = 0; i < nsteps; ++i)
			{
				Eigen::Vector3f p = origin + (direction * step_size * (const float)i);

				PointT octree_p;
				octree_p.x = p.x ();
				octree_p.y = p.y ();
				octree_p.z = p.z ();

				OctreeKey key;
				this->genOctreeKeyforPoint (octree_p, key);

				// Not a new key, still the same voxel.
				if (key == prev_key)
				  continue;

				prev_key = key;

				// check if it is occupied
				if(isVoxelOccupiedAtKey(key))
				{
					PointT center;
					genLeafNodeCenterFromOctreeKey (key, center);
					voxel_center_list.push_back (center);
				}
			}

			OctreeKey end_key;
			PointT end_p;
			end_p.x = end.x ();
			end_p.y = end.y ();
			end_p.z = end.z ();
			this->genOctreeKeyforPoint (end_p, end_key);
			if (!(end_key == prev_key))
			{
				// check if it is occupied
				if(isVoxelOccupiedAtKey(end_key))
				{
					PointT center;
					genLeafNodeCenterFromOctreeKey (end_key, center);
					voxel_center_list.push_back (center);
				}
			}

			return ((int)voxel_center_list.size ());
		}


        /** \brief Delete leaf node / voxel at given point
          * \param[in] point_arg point addressing the voxel to be deleted.
          */
        void
        deleteVoxelAtPoint (const PointT& point_arg)
		{
			OctreeKey key;

			// generate key for point
			this->genOctreeKeyforPoint (point_arg, key);

			this->removeLeaf (key);
		}


        /** \brief Delete the octree structure and its leaf nodes. */
        void
        deleteTree ()
        {
          // reset bounding box
          minX_ = minY_ = maxY_ = minZ_ = maxZ_ = 0;
          maxKeys_ = 1;
          this->boundingBoxDefined_ = false;

          OctreeT::deleteTree ();
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Bounding box methods
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        /** \brief Define bounding box for octree
          * \note Bounding box cannot be changed once the octree contains elements.
          * \param[in] minX_arg X coordinate of lower bounding box corner
          * \param[in] minY_arg Y coordinate of lower bounding box corner
          * \param[in] minZ_arg Z coordinate of lower bounding box corner
          * \param[in] maxX_arg X coordinate of upper bounding box corner
          * \param[in] maxY_arg Y coordinate of upper bounding box corner
          * \param[in] maxZ_arg Z coordinate of upper bounding box corner
          */
        void
        defineBoundingBox (const double minX_arg, const double minY_arg, const double minZ_arg, 
											const double maxX_arg, const double maxY_arg, const double maxZ_arg)
		{
			// bounding box cannot be changed once the octree contains elements
			assert (this->leafCount_ == 0);

			assert (maxX_arg >= minX_arg);
			assert (maxY_arg >= minY_arg);
			assert (maxZ_arg >= minZ_arg);

			minX_ = minX_arg;
			maxX_ = maxX_arg;

			minY_ = minY_arg;
			maxY_ = maxY_arg;

			minZ_ = minZ_arg;
			maxZ_ = maxZ_arg;

			minX_ = min (minX_, maxX_);
			minY_ = min (minY_, maxY_);
			minZ_ = min (minZ_, maxZ_);

			maxX_ = max (minX_, maxX_);
			maxY_ = max (minY_, maxY_);
			maxZ_ = max (minZ_, maxZ_);

			// generate bit masks for octree
			getKeyBitSize ();

			boundingBoxDefined_ = true;
		}

        /** \brief Define bounding box for octree
          * \note Lower bounding box point is set to (0, 0, 0)
          * \note Bounding box cannot be changed once the octree contains elements.
          * \param[in] maxX_arg X coordinate of upper bounding box corner
          * \param[in] maxY_arg Y coordinate of upper bounding box corner
          * \param[in] maxZ_arg Z coordinate of upper bounding box corner
          */
        void
        defineBoundingBox (const double maxX_arg, const double maxY_arg, const double maxZ_arg)
		{
			// bounding box cannot be changed once the octree contains elements
			assert (this->leafCount_ == 0);

			assert (maxX_arg >= 0.0f);
			assert (maxY_arg >= 0.0f);
			assert (maxZ_arg >= 0.0f);

			minX_ = 0.0f;
			maxX_ = maxX_arg;

			minY_ = 0.0f;
			maxY_ = maxY_arg;

			minZ_ = 0.0f;
			maxZ_ = maxZ_arg;

			minX_ = min (minX_, maxX_);
			minY_ = min (minY_, maxY_);
			minZ_ = min (minZ_, maxZ_);

			maxX_ = max (minX_, maxX_);
			maxY_ = max (minY_, maxY_);
			maxZ_ = max (minZ_, maxZ_);

			// generate bit masks for octree
			getKeyBitSize ();

			boundingBoxDefined_ = true;
		}

        /** \brief Define bounding box cube for octree
          * \note Lower bounding box corner is set to (0, 0, 0)
          * \note Bounding box cannot be changed once the octree contains elements.
          * \param[in] cubeLen_arg side length of bounding box cube.
          */
        void
        defineBoundingBox (const double cubeLen_arg)
		{
			// bounding box cannot be changed once the octree contains elements
			assert (this->leafCount_ == 0);

			assert (cubeLen_arg >= 0.0f);

			minX_ = 0.0f;
			maxX_ = cubeLen_arg;

			minY_ = 0.0f;
			maxY_ = cubeLen_arg;

			minZ_ = 0.0f;
			maxZ_ = cubeLen_arg;

			minX_ = min (minX_, maxX_);
			minY_ = min (minY_, maxY_);
			minZ_ = min (minZ_, maxZ_);

			maxX_ = max (minX_, maxX_);
			maxY_ = max (minY_, maxY_);
			maxZ_ = max (minZ_, maxZ_);

			// generate bit masks for octree
			getKeyBitSize ();

			boundingBoxDefined_ = true;
		}

        /** \brief Get bounding box for octree
          * \note Bounding box cannot be changed once the octree contains elements.
          * \param[in] minX_arg X coordinate of lower bounding box corner
          * \param[in] minY_arg Y coordinate of lower bounding box corner
          * \param[in] minZ_arg Z coordinate of lower bounding box corner
          * \param[in] maxX_arg X coordinate of upper bounding box corner
          * \param[in] maxY_arg Y coordinate of upper bounding box corner
          * \param[in] maxZ_arg Z coordinate of upper bounding box corner
          */
        void
        getBoundingBox (double& minX_arg, double& minY_arg, double& minZ_arg, 
									   double& maxX_arg, double& maxY_arg, double& maxZ_arg) const
		{
			minX_arg = minX_;
			minY_arg = minY_;
			minZ_arg = minZ_;

			maxX_arg = maxX_;
			maxY_arg = maxY_;
			maxZ_arg = maxZ_;
		}

        /** \brief Calculates the squared diameter of a voxel at given tree depth
          * \param[in] treeDepth_arg depth/level in octree
          * \return squared diameter
          */
        double
        getVoxelSquaredDiameter (unsigned int treeDepth_arg) const
		{
			// return the squared side length of the voxel cube as a function of the octree depth
			return (getVoxelSquaredSideLen (treeDepth_arg) * 3);
		}

        /** \brief Calculates the squared diameter of a voxel at leaf depth
          * \return squared diameter
          */
        inline double
        getVoxelSquaredDiameter () const
        {
			return getVoxelSquaredDiameter (this->octreeDepth_);
        }

        /** \brief Calculates the squared voxel cube side length at given tree depth
          * \param[in] treeDepth_arg depth/level in octree
          * \return squared voxel cube side length
          */
        double
        getVoxelSquaredSideLen (unsigned int treeDepth_arg) const
		{
			double sideLen;

			// side length of the voxel cube increases exponentially with the octree depth
			sideLen = this->resolution_ * (double)(1 << (this->octreeDepth_ - treeDepth_arg));

			// squared voxel side length
			sideLen *= sideLen;

			return (sideLen);
		}

        /** \brief Calculates the squared voxel cube side length at leaf level
          * \return squared voxel cube side length
          */
        inline double
        getVoxelSquaredSideLen () const
        {
			return getVoxelSquaredSideLen (this->octreeDepth_);
        }


        /** \brief Generate bounds of the current voxel of an octree iterator
         * \param[in] iterator: octree iterator
         * \param[out] min_pt lower bound of voxel
         * \param[out] max_pt upper bound of voxel
         */
        inline void
        getVoxelBounds (OctreeIteratorBase<int,LeafT,OctreeT>& iterator, Eigen::Vector3f &min_pt, Eigen::Vector3f &max_pt)
        {
			this->genVoxelBoundsFromOctreeKey (iterator.getCurrentOctreeKey(), iterator.getCurrentOctreeDepth(), min_pt, max_pt);
        }


      protected:

        /** \brief Check if voxel at given octree key.
          * \param[in] key_arg key to be checked
          * \return "true" if voxel exist; "false" otherwise
          */
        bool
        isVoxelOccupiedAtKey(const OctreeKey &key_arg) const
		{
			// search for key in octree
			if(LeafT* pLeaf = this->findLeaf(key))
			{
				// mean and variance
				Scalar mean						= pLeaf->getMean();
				Scalar variance					= ((Scalar) 1.f) / pLeaf->getInverseVariance();

				// probabilistic least square classification
				// int \Phi(y*(x-m)/v) \mathcal{N}(x; mu, s2) dx = \Phi(y*(mu - m)/(v*sqrt(1+s2/v^2)))
				Scalar occupancy = normcdf(((Scalar) -1.f) * (mean - PLSC_mean_) / sqrt(variance + PLSC_variance_));

				return occupancy > occupancyThreshold_;
			}

			return false;
		}


        /** \brief Find octree leaf node at a given point
          * \param[in] point_arg query point
          * \return pointer to leaf node. If leaf node does not exist, pointer is 0.
          */
        LeafT*
        findLeafAtPoint (const PointT& point_arg) const 
		{
			OctreeKey key;

			// generate key for point
			this->genOctreeKeyforPoint (point, key);

			return (this->findLeaf (key));
		}

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Protected octree methods based on octree keys
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        typedef typename OctreeT::OctreeKey				OctreeKey;
        typedef typename OctreeT::OctreeBranch			OctreeBranch;

        /** \brief Define octree key setting and octree depth based on defined bounding box. */
        void
        getKeyBitSize ()
		{
			unsigned int maxVoxels;

			unsigned int maxKeyX;
			unsigned int maxKeyY;
			unsigned int maxKeyZ;

			double octreeSideLen;

			const float minValue = std::numeric_limits<float>::epsilon();

			// find maximum key values for x, y, z
			maxKeyX = (unsigned int)((maxX_ - minX_) / resolution_);
			maxKeyY = (unsigned int)((maxY_ - minY_) / resolution_);
			maxKeyZ = (unsigned int)((maxZ_ - minZ_) / resolution_);

			// find maximum amount of keys
			maxVoxels = max (max (max (maxKeyX, maxKeyY), maxKeyZ), (unsigned int)2);


			// tree depth == amount of bits of maxVoxels
			this->octreeDepth_ = max ((min ((unsigned int)OCT_MAXTREEDEPTH, 
																	 (unsigned int)ceil (this->Log2 (maxVoxels)-minValue))), (unsigned int)0);

			maxKeys_ = (1 << this->octreeDepth_);

			octreeSideLen = (double)maxKeys_ * resolution_-minValue;

			if (this->leafCount_ == 0)
			{
				double octreeOversizeX;
				double octreeOversizeY;
				double octreeOversizeZ;

				octreeOversizeX = (octreeSideLen - (maxX_ - minX_)) / 2.0;
				octreeOversizeY = (octreeSideLen - (maxY_ - minY_)) / 2.0;
				octreeOversizeZ = (octreeSideLen - (maxZ_ - minZ_)) / 2.0;

				minX_ -= octreeOversizeX;
				minY_ -= octreeOversizeY;
				minZ_ -= octreeOversizeZ;

				maxX_ += octreeOversizeX;
				maxY_ += octreeOversizeY;
				maxZ_ += octreeOversizeZ;
			}
			else
			{
				maxX_ = minX_ + octreeSideLen;
				maxY_ = minY_ + octreeSideLen;
				maxZ_ = minZ_ + octreeSideLen;
			}

			// configure tree depth of octree
			this->setTreeDepth (this->octreeDepth_);
		}

        /** \brief Grow the bounding box/octree until point fits
          * \param[in] pointIdx_arg point that should be within bounding box;
          */
        void
        adoptBoundingBoxToPoint (const PointT& pointIdx_arg)
		{
			const float minValue = std::numeric_limits<float>::epsilon();

			// increase octree size until point fits into bounding box
			while (true)
			{
				bool bLowerBoundViolationX = (pointIdx_arg.x < minX_);
				bool bLowerBoundViolationY = (pointIdx_arg.y < minY_);
				bool bLowerBoundViolationZ = (pointIdx_arg.z < minZ_);

				bool bUpperBoundViolationX = (pointIdx_arg.x >= maxX_);
				bool bUpperBoundViolationY = (pointIdx_arg.y >= maxY_);
				bool bUpperBoundViolationZ = (pointIdx_arg.z >= maxZ_);

				// do we violate any bounds?
				if (bLowerBoundViolationX || bLowerBoundViolationY || bLowerBoundViolationZ || 
					bUpperBoundViolationX	|| bUpperBoundViolationY || bUpperBoundViolationZ || 
					(!boundingBoxDefined_))
				{
					double octreeSideLen;
					unsigned char childIdx;

					if (this->leafCount_ > 0)
					{
						// octree not empty - we add another tree level and thus increase its size by a factor of 2*2*2
						childIdx = ((!bUpperBoundViolationX) << 2) | ((!bUpperBoundViolationY) << 1) | ((!bUpperBoundViolationZ));

						OctreeBranch* newRootBranch;

						this->createBranch (newRootBranch);
						this->branchCount_++;

						this->setBranchChild (*newRootBranch, childIdx, this->rootNode_);

						this->rootNode_ = newRootBranch;

						octreeSideLen = (double)maxKeys_ * resolution_ ;

						if (!bUpperBoundViolationX)				  minX_ -= octreeSideLen;
						if (!bUpperBoundViolationY)				  minY_ -= octreeSideLen;
						if (!bUpperBoundViolationZ)				  minZ_ -= octreeSideLen;

						// configure tree depth of octree
						this->octreeDepth_ ++;
						this->setTreeDepth (this->octreeDepth_);
						maxKeys_ = (1 << this->octreeDepth_);

						// recalculate bounding box width
						octreeSideLen = (double)maxKeys_ * resolution_ - minValue;

						// increase octree bounding box
						maxX_ = minX_ + octreeSideLen;
						maxY_ = minY_ + octreeSideLen;
						maxZ_ = minZ_ + octreeSideLen;
					}
					else
					{
						// octree is empty - we set the center of the bounding box to our first pixel
						this->minX_ = pointIdx_arg.x - this->resolution_ / 2;
						this->minY_ = pointIdx_arg.y - this->resolution_ / 2;
						this->minZ_ = pointIdx_arg.z - this->resolution_ / 2;

						this->maxX_ = pointIdx_arg.x + this->resolution_ / 2;
						this->maxY_ = pointIdx_arg.y + this->resolution_ / 2;
						this->maxZ_ = pointIdx_arg.z + this->resolution_ / 2;

						getKeyBitSize();
					}

					boundingBoxDefined_ = true;
				}
				else
					// no bound violations anymore - leave while loop
					break;
			}
		}

        /** \brief Generate octree key for voxel at a given point
          * \param[in] point_arg the point addressing a voxel
          * \param[out] key_arg write octree key to this reference
          */
        void
        genOctreeKeyforPoint (const PointT & point_arg, OctreeKey &key_arg) const
		{
			// calculate integer key for point coordinates
			key_arg.x = min ((unsigned int)((point_arg.x - this->minX_) / this->resolution_), maxKeys_ - 1);
			key_arg.y = min ((unsigned int)((point_arg.y - this->minY_) / this->resolution_), maxKeys_ - 1);
			key_arg.z = min ((unsigned int)((point_arg.z - this->minZ_) / this->resolution_), maxKeys_ - 1);
		}

        /** \brief Generate octree key for voxel at a given point
          * \param[in] pointX_arg X coordinate of point addressing a voxel
          * \param[in] pointY_arg Y coordinate of point addressing a voxel
          * \param[in] pointZ_arg Z coordinate of point addressing a voxel
          * \param[out] key_arg write octree key to this reference
          */
        void
        genOctreeKeyforPoint (const double pointX_arg, const double pointY_arg, const double pointZ_arg,
												 OctreeKey & key_arg) const
		{
			PointT tempPoint;

			tempPoint.x = (float)pointX_arg;
			tempPoint.y = (float)pointY_arg;
			tempPoint.z = (float)pointZ_arg;

			// generate key for point
			genOctreeKeyforPoint (tempPoint, key_arg);
		}

        /** \brief Virtual method for generating octree key for a given point index.
          * \note This method enables to assign indices to leaf nodes during octree deserialization.
          * \param[in] data_arg index value representing a point in the dataset given by \a setInputCloud
          * \param[out] key_arg write octree key to this reference
          * \return "true" - octree keys are assignable
          */
        virtual bool
        genOctreeKeyForDataT (const int& data_arg, OctreeKey & key_arg) const
		{
			const PointT tempPoint = getPointByIndex (data_arg);

			// generate key for point
			genOctreeKeyforPoint (tempPoint, key_arg);

			return (true);
		}

        /** \brief Generate a point at center of leaf node voxel
          * \param[in] key_arg octree key addressing a leaf node.
          * \param[out] point_arg write leaf node voxel center to this point reference
          */
        void
        genLeafNodeCenterFromOctreeKey (const OctreeKey & key_arg, PointT& point_arg) const
		{
			// define point to leaf node voxel center
			point.x = (float)(((double)key.x + 0.5f) * this->resolution_ + this->minX_);
			point.y = (float)(((double)key.y + 0.5f) * this->resolution_ + this->minY_);
			point.z = (float)(((double)key.z + 0.5f) * this->resolution_ + this->minZ_);
		}

        /** \brief Generate a point at center of octree voxel at given tree level
          * \param[in] key_arg octree key addressing an octree node.
          * \param[in] treeDepth_arg octree depth of query voxel
          * \param[out] point_arg write leaf node center point to this reference
          */
        void
        genVoxelCenterFromOctreeKey (const OctreeKey & key_arg, unsigned int treeDepth_arg, PointT& point_arg) const
		{
			// generate point for voxel center defined by treedepth (bitLen) and key
			point_arg.x = (float)(((double)(key_arg.x) + 0.5f) * (this->resolution_ * (double)(1 << (this->octreeDepth_  - treeDepth_arg))) + this->minX_);
			point_arg.y = (float)(((double)(key_arg.y) + 0.5f) * (this->resolution_ * (double)(1 << (this->octreeDepth_  - treeDepth_arg))) + this->minY_);
			point_arg.z = (float)(((double)(key_arg.z) + 0.5f) * (this->resolution_ * (double)(1 << (this->octreeDepth_  - treeDepth_arg))) + this->minZ_);
		}

        /** \brief Generate bounds of an octree voxel using octree key and tree depth arguments
          * \param[in] key_arg octree key addressing an octree node.
          * \param[in] treeDepth_arg octree depth of query voxel
          * \param[out] min_pt lower bound of voxel
          * \param[out] max_pt upper bound of voxel
          */
        void
        genVoxelBoundsFromOctreeKey (const OctreeKey & key_arg, unsigned int treeDepth_arg, Eigen::Vector3f &min_pt,
																	Eigen::Vector3f &max_pt) const
		{
			// calculate voxel size of current tree depth
			double voxel_side_len = this->resolution_ * (double)(1 << (this->octreeDepth_ - treeDepth_arg));

			// calculate voxel bounds
			min_pt (0) = (float)((double)(key_arg.x) * voxel_side_len + this->minX_);
			min_pt (1) = (float)((double)(key_arg.y) * voxel_side_len + this->minY_);
			min_pt (2) = (float)((double)(key_arg.z) * voxel_side_len + this->minZ_);

			max_pt (0) = (float)((double)(key_arg.x + 1) * voxel_side_len + this->minX_);
			max_pt (1) = (float)((double)(key_arg.y + 1) * voxel_side_len + this->minY_);
			max_pt (2) = (float)((double)(key_arg.z + 1) * voxel_side_len + this->minZ_);
		}

        /** \brief Recursively search the tree for all leaf nodes and return a vector of voxel centers.
          * \param[in] node_arg current octree node to be explored
          * \param[in] key_arg octree key addressing a leaf node.
          * \param[out] voxelCenterList_arg results are written to this vector of PointT elements
          * \return number of voxels found
          */
        int
        getOccupiedVoxelCentersRecursive (const OctreeBranch* node_arg, const OctreeKey& key_arg,
																		  std::vector<PointT, Eigen::aligned_allocator<PointT> > &voxelCenterList_arg) const
		{
			// child iterator
			unsigned char childIdx;

			int voxelCount = 0;

			// iterate over all children
			for (childIdx = 0; childIdx < 8; childIdx++)
			{
				if (!this->branchHasChild (*node_arg, childIdx))
					continue;

				const OctreeNode * childNode;
				childNode = this->getBranchChild (*node_arg, childIdx);

				// generate new key for current branch voxel
				OctreeKey newKey;
				newKey.x = (key_arg.x << 1) | (!!(childIdx & (1 << 2)));
				newKey.y = (key_arg.y << 1) | (!!(childIdx & (1 << 1)));
				newKey.z = (key_arg.z << 1) | (!!(childIdx & (1 << 0)));

				switch (childNode->getNodeType ())
				{
					case BRANCH_NODE:
					{
						// recursively proceed with indexed child branch
						voxelCount += getOccupiedVoxelCentersRecursive ((OctreeBranch*)childNode, newKey, voxelCenterList_arg);
						break;
					}
					case LEAF_NODE:
					{
						// check if it is occupied
						if(isVoxelOccupiedAtKey(newKey))
						{
							PointT newPoint;

							genLeafNodeCenterFromOctreeKey (newKey, newPoint);
							voxelCenterList_arg.push_back (newPoint);

							voxelCount++;
							}
						break;
					}
					default:
						break;
				}
			}

			return (voxelCount);
		}

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Globals
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /** \brief Pointer to input point cloud dataset. */
        //PointCloudConstPtr input_;

        /** \brief A pointer to the vector of point indices to use. */
        //IndicesConstPtr indices_;

        /** \brief Epsilon precision (error bound) for nearest neighbors searches. */
        //double epsilon_;

        /** \brief Octree resolution. */
        double resolution_;

        // Octree bounding box coordinates
        double minX_;
        double maxX_;

        double minY_;
        double maxY_;

        double minZ_;
        double maxZ_;

        /** \brief Maximum amount of keys available in octree. */
        unsigned int maxKeys_;

        /** \brief Flag indicating if octree has defined bounding box. */
        bool boundingBoxDefined_;

        /** \brief If an inverse variance is less than this threshold, then the node will be pruned. */
        Scalar inverseVarianceThreshold_;

        /** \brief If an occupancy is greater than this threshold, then the node will be thought of as being occupied. */
        Scalar occuapancyThreshold_;

        /** \brief mean for PLSC. */
        Scalar PLSC_mean_;

        /** \brief variance for PLSC. */
        Scalar PLSC_variance_;
    };
}


//#if 0
////#include <pcl/octree/octree.h>
////#include <pcl/octree/octree_impl.h> 
//#include <pcl/octree/octree_nodes.h>
//#include <pcl/octree/octree_base.h>
//
//#include "GP/DataTypes.hpp"
//
//namespace GPOM {
//
//	template<typename DataT>
//	class OctreePointCloudGPLeaf : public pcl::octree::OctreeLeafAbstract<DataT>
//    {
//    public:
//      /** \brief Class initialization. */
//      OctreePointCloudGPLeaf ()
//		  : m_mean((Scalar) 0.f), m_inverseVariance((Scalar) 0.f)
//      {
//      }
//
//      /** \brief Empty class deconstructor. */
//      ~OctreePointCloudGPLeaf ()
//      {
//      }
//
//      /** \brief deep copy function */
//      virtual OctreeNode *
//      deepCopy () const
//      {
//        return (OctreeNode*) new OctreePointCloudGPLeaf (*this);
//      }
//
//      /** \brief Read input data. Only an internal counter is increased.
//       * /param point_arg: input point - this argument is ignored
//       *  */
//      virtual void
//      setData (const DataT& point_arg)
//      {
//      }
//
//      /** \brief Returns a null pointer as this leaf node does not store any data.
//       *  \param data_arg: reference to return pointer of leaf node DataT element (will be set to 0).
//       */
//      virtual void
//      getData (const DataT*& data_arg) const
//      {
//        data_arg = 0;
//      }
//
//      /** \brief Empty getData data vector implementation as this leaf node does not store any data. \
//       *  \param dataVector_arg: reference to dummy DataT vector that is extended with leaf node DataT elements.
//       */
//      virtual void
//      getData (std::vector<DataT>& dataVector_arg) const
//      {
//      }
//
//      /** \brief Return mean.
//       *  \return mean
//       * */
//      Scalar
//      getMean()
//      {
//        return m_mean;
//      }
//
//      /** \brief Return inverse variance.
//       *  \return sigma
//       * */
//      Scalar
//      getInverseVariance()
//      {
//        return m_inverseVariance;
//      }
//
//      /** \brief Merge predictions.
//       *  \param dataVector_arg: reference to dummy DataT vector that is extended with leaf node DataT elements.
//       * */
//      void
//      merge(const Scalar mean, const Scalar inverseVariance)
//      {
//		  m_mean						= m_inverseVariance * m_mean + inverseVariance * mean;
//		  m_inverseVariance		+= inverseVariance;
//		  m_mean						/= m_inverseVariance;
//      }
//
//      /** \brief Empty reset leaf node implementation as this leaf node does not store any data. */
//      virtual void
//      reset ()
//      {
//		m_inverseVariance = (Scalar) 0.f;
//		m_mean = (Scalar) 0.f;
//      }
//
//    private:
//		Scalar m_inverseVariance;
//		Scalar m_mean;
//    };
//
//    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    /** \brief @b Octree pointcloud GP class
//     *  \note This class generate an octrees from a point cloud (zero-copy). Only the amount of points that fall into the leaf node voxel are stored.
//     *  \note The octree pointcloud is initialized with its voxel resolution. Its bounding box is automatically adjusted or can be predefined.
//     *  \note
//     *  \note typename: PointT: type of point used in pointcloud
//     *  \ingroup octree
//     *  \author Julius Kammerl (julius@kammerl.de)
//     */
//    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    template<typename PointT, 
//				     typename LeafT = OctreePointCloudGPLeaf<int>, 
//					 typename OctreeT = pcl::octree::OctreeBase<int, LeafT> >
//	class OctreePointCloudGP : public pcl::octree::OctreePointCloud<PointT, LeafT, OctreeT>
//      {
//		public:
//			// public typedefs for single/double buffering
//			//typedef OctreePointCloudGP<PointT, LeafT, pcl::octree::OctreeBase<int, LeafT> > SingleBuffer;
//			//typedef OctreePointCloudGP<PointT, LeafT, pcl::octree::Octree2BufBase<int, LeafT> > DoubleBuffer;
//
//		  // iterators are friends
//		  friend class pcl::octree::OctreeIteratorBase<int, LeafT, OctreeT> ;
//		  friend class pcl::octree::OctreeDepthFirstIterator<int, LeafT, OctreeT> ;
//		  friend class pcl::octree::OctreeBreadthFirstIterator<int, LeafT, OctreeT> ;
//		  friend class pcl::octree::OctreeLeafNodeIterator<int, LeafT, OctreeT> ;
//
//			// Octree iterators
//			typedef pcl::octree::OctreeDepthFirstIterator<int, LeafT, OctreeT> Iterator;
//			typedef const pcl::octree::OctreeDepthFirstIterator<int, LeafT, OctreeT> ConstIterator;
//
//			typedef pcl::octree::OctreeLeafNodeIterator<int, LeafT, OctreeT> LeafNodeIterator;
//			typedef const pcl::octree::OctreeLeafNodeIterator<int, LeafT, OctreeT> ConstLeafNodeIterator;
//
//			typedef pcl::octree::OctreeDepthFirstIterator<int, LeafT, OctreeT> DepthFirstIterator;
//			typedef const pcl::octree::OctreeDepthFirstIterator<int, LeafT, OctreeT> ConstDepthFirstIterator;
//			typedef pcl::octree::OctreeBreadthFirstIterator<int, LeafT, OctreeT> BreadthFirstIterator;
//			typedef const pcl::octree::OctreeBreadthFirstIterator<int, LeafT, OctreeT> ConstBreadthFirstIterator;
//
//      /** \brief OctreePointCloudDensity class constructor.
//       *  \param resolution_arg:  octree resolution at lowest octree level
//       * */
//      OctreePointCloudGP (const double resolution_arg)
//		  : OctreePointCloud<PointT, LeafT, OctreeT> (resolution_arg)
//      {
//      }
//
//      /** \brief Empty class deconstructor. */
//      virtual
//      ~OctreePointCloudGP ()
//      {
//      }
//
//      /** \brief Get the mean and variance within a leaf node voxel which is addressed by a point
//       *  \param point_arg: a point addressing a voxel
//       *  \param mean: mean
//       *  \param variance: variance
//       * */
//		bool
//		getMeanAndInverseVarianceAtPoint (const PointT			&point_arg,
//																		 Scalar					&mean, 
//																		 Scalar					&inverseVariance) const
//		{
//			mean						= (Scalar) 0.f;
//			inverseVariance	= (Scalar) 0.f;
//
//			OctreePointCloudGPLeaf<int>* leaf = this->findLeafAtPoint (point_arg);
//
//			if (leaf)
//			{
//				mean						= leaf->getMean();
//				inverseVariance	= leaf->getInverseVariance();
//				return true;
//			}
//
//			return false;
//		}
//
//      /** \brief Merge the mean and variance within a leaf node voxel which is addressed by a point
//       *  \param point_arg: a point addressing a voxel
//       *  \param mean: mean
//       *  \param variance: variance
//       * */
//		bool
//		mergeMeanAndInverseVarianceAtPoint (const PointT			&point_arg,
//																			   const Scalar			mean, 
//																			   const Scalar			inverseVariance) const
//		{
//			OctreePointCloudGPLeaf<int>* leaf = this->findLeafAtPoint (point_arg);
//
//			if (leaf)
//			{
//				leaf->merge(mean, inverseVariance);
//
//				// prune node
//				return true;
//			}
//			else
//			{
//				// create leaf node
//			}
//
//			return false;
//		}
//
//      };
//}
//#endif

#endif