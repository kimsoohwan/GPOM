#ifndef OCTREE_FOR_GAUSSIAN_PROCESS
#define OCTREE_FOR_GAUSSIAN_PROCESS

#include <assert.h>
#include <algorithm>
using std::min;
using std::max;

#include <pcl/octree/octree_base.h>
#include <pcl/octree/octree2buf_base.h>

#include <pcl/point_types.h>

#include <pcl/octree/octree_nodes.h>
#include <pcl/octree/octree_iterator.h>

#include "util/normcdf.hpp"

namespace GPOM {

	class GaussianDistribution
	{
	public:
		GaussianDistribution()
			: m_mean((Scalar) 0.f), 
			  m_inverseVariance((Scalar) 0.f)
		{
		}

		GaussianDistribution(const Scalar mean, const Scalar variance)
			: m_mean(mean), 
			  m_inverseVariance(((Scalar) 1.f) / variance)
		{
		}

		GaussianDistribution& operator=(const GaussianDistribution &rhs)
		{
			// Only do assignment if RHS is a different object from this.
			if (this != &rhs)
			{
				m_mean						= rhs.m_mean;
				m_inverseVariance	= rhs.m_inverseVariance;
			}
			return *this;
		}

		// merge
		GaussianDistribution& operator+=(const GaussianDistribution &rhs)
		{
			m_mean						= m_inverseVariance * m_mean + rhs.m_inverseVariance * rhs.m_mean;
			m_inverseVariance			+= rhs.m_inverseVariance;
			m_mean						/= m_inverseVariance;
			return *this;
		}

		/** \brief Return mean.
		*  \return mean
		* */
		Scalar
		getMean() const
		{
			return m_mean;
		}

		/** \brief Return variance.
		*  \return variance
		* */
		Scalar
		getVariance() const
		{
			return ((Scalar) 1.f) / m_inverseVariance;
		}

		/** \brief Return inverse variance.
		*  \return inverse variance
		* */
		Scalar
		getInverseVariance() const
		{
			return m_inverseVariance;
		}

		/** \brief reset */
		void
		reset ()
		{
			m_mean					= (Scalar) 0.f;
			m_inverseVariance		= (Scalar) 0.f;
		}

	protected:
		Scalar			m_mean;
		Scalar			m_inverseVariance;
	};

	template <typename DataT>
	class MergeContainer : public DataT
    {
    public:

      /** \brief Class initialization. */
      MergeContainer ()
      {
      }

      /** \brief Empty class deconstructor. */
      ~MergeContainer ()
      {
      }

      /** \brief deep copy function */
      virtual MergeContainer*
      deepCopy () const
      {
        return new MergeContainer (*this);
      }

		/** \brief Get size of container (number of DataT objects)
			* \return number of DataT elements in leaf node container.
			*/
		size_t
		getSize () const
		{
			return 0;
		}

      /** \brief Read input data. Only an internal counter is increased.
       * /param data: input Gaussian distribution
       *  */
      void
      setData (const DataT& data)
      {
		  m_data += data;
      }

      /** \brief Returns a null pointer as this leaf node does not store any data.
       *  \param data: reference to return pointer of leaf node DataT element (will be set to 0).
       */
      //void
      ////getData (const DataT*& data) const
      //getData (DataT &data) const
      //{
      //  data = &m_Gaussian;
      //}
		void
      getData (DataT &data) const
      {
        data = m_data;
      }

		const DataT&
      getData () const
      {
        return m_data;
      }

      /** \brief Empty getData data vector implementation as this leaf node does not store any data. \
       *  \param dataVector: reference to dummy DataT vector that is extended with leaf node DataT elements.
       */
      void
      getData (std::vector<DataT>& dataVector) const
      {
      }

      /** \brief Empty reset leaf node implementation as this leaf node does not store any data. */
      void
      reset ()
      {
			m_data.reset();
      }

    private:
		DataT	m_data;
    };

	//template<typename LeafT = MergeContainer<GaussianDistribution>,
	//			typename BranchT = pcl::octree::OctreeContainerEmpty<GaussianDistribution>,
	//		   typename OctreeT = pcl::octree::OctreeBase<GaussianDistribution, LeafT, BranchT> >
	//class OctreeGPOM : public OctreeT
	class OctreeGPOM : public pcl::octree::OctreeBase< GaussianDistribution, 
																	   MergeContainer<GaussianDistribution>, 
																	   pcl::octree::OctreeContainerEmpty<GaussianDistribution> >
    {
	 public:
			typedef MergeContainer<GaussianDistribution>																		LeafT;
			typedef pcl::octree::OctreeContainerEmpty<GaussianDistribution>											BranchT;
			typedef pcl::octree::OctreeBase< GaussianDistribution, 
														MergeContainer<GaussianDistribution>, 
														pcl::octree::OctreeContainerEmpty<GaussianDistribution> >		OctreeT;

			// iterators are friends
			friend class pcl::octree::OctreeIteratorBase<GaussianDistribution, OctreeT> ;
			friend class pcl::octree::OctreeDepthFirstIterator<GaussianDistribution, OctreeT> ;
			friend class pcl::octree::OctreeBreadthFirstIterator<GaussianDistribution, OctreeT> ;
			friend class pcl::octree::OctreeLeafNodeIterator<GaussianDistribution, OctreeT> ;

      public:
			typedef OctreeT						Base;
			typedef OctreeT::LeafNode			LeafNode;
			typedef OctreeT::BranchNode		BranchNode;

			// Octree iterators
			typedef pcl::octree::OctreeDepthFirstIterator<GaussianDistribution, OctreeT>					Iterator;
			typedef const pcl::octree::OctreeDepthFirstIterator<GaussianDistribution, OctreeT>			ConstIterator;

			typedef pcl::octree::OctreeLeafNodeIterator<GaussianDistribution, OctreeT>						LeafNodeIterator;
			typedef const pcl::octree::OctreeLeafNodeIterator<GaussianDistribution, OctreeT>				ConstLeafNodeIterator;

			typedef pcl::octree::OctreeDepthFirstIterator<GaussianDistribution, OctreeT>					DepthFirstIterator;
			typedef const pcl::octree::OctreeDepthFirstIterator<GaussianDistribution, OctreeT>			ConstDepthFirstIterator;
			typedef pcl::octree::OctreeBreadthFirstIterator<GaussianDistribution, OctreeT>					BreadthFirstIterator;
			typedef const pcl::octree::OctreeBreadthFirstIterator<GaussianDistribution, OctreeT>			ConstBreadthFirstIterator;

			// public typedefs for single/double buffering
			//typedef OctreeGPOM<LeafT, pcl::octree::OctreeBase<GaussianDistribution, LeafT> >				SingleBuffer;
			//typedef OctreeGPOM<LeafT, pcl::octree::Octree2BufBase<GaussianDistribution, LeafT> >			DoubleBuffer;

			// Boost shared pointers
			//typedef boost::shared_ptr<OctreeGPOM<GaussianDistribution, LeafT, OctreeT> >					Ptr;
			//typedef boost::shared_ptr<const OctreeGPOM<GaussianDistribution, LeafT, OctreeT> >			ConstPtr;
			typedef boost::shared_ptr<OctreeGPOM>					Ptr;
			typedef boost::shared_ptr<const OctreeGPOM>			ConstPtr;

			// Eigen aligned allocator
			typedef std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ> >											AlignedPointTVector;


			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			// Merge
			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	 public:
			/** \brief Merge mean and inverse variance at a given point. If there is no leaf node at the point, create one.
			  * \param[in] point position
			  * \param[in] mean mean
			  * \param[in] inverseVariance inverse variance
			  */
			void
			mergeMeanAndVarianceAtPoint(const pcl::PointXYZ &point, const Scalar mean, const Scalar variance)
			{
				// make sure bounding box is big enough
				adoptBoundingBoxToPoint (point);

				// generate key
				pcl::octree::OctreeKey key;
				genOctreeKeyforPoint (point, key);

				// merge the gaussian process to octree at key
				this->addData(key, GaussianDistribution(mean, variance));
			}


			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			// Prune
			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

			/** \brief Prune nodes that have high uncertainty
			  * \param[in] varianceThreshold variance threshold
			  */
			unsigned int
			pruneUncertainNodes(const Scalar varianceThreshold)
			{
				// variables
				GaussianDistribution gaussian;
				const Scalar inverseVarianceThreshold = ((Scalar) 1.f) / varianceThreshold;
				unsigned int numNodeRemoved = 0; 

				// iterator
				LeafNodeIterator iter(*this);
				const GaussianDistribution* pGaussian;
				while(*++iter)
				{
					pGaussian = dynamic_cast<const GaussianDistribution*> (*iter);
					if(pGaussian->getInverseVariance() < inverseVarianceThreshold)
					{
						removeLeaf(iter.getCurrentOctreeKey());
						numNodeRemoved++;
					}
				}
			}

        /** \brief Prune nodes that have low occupancy
          * \param[in] occupancyThreshold occupancy threshold
          */
			unsigned int
			pruneUnoccupiedNodes(const Scalar occupancyThreshold)
			{
				// variables
				GaussianDistribution gaussian;
				unsigned int numNodeRemoved = 0; 

				// iterator
				LeafNodeIterator iter(*this);
				const GaussianDistribution* pGaussian;
				while(*++iter)
				{
					pGaussian = dynamic_cast<const GaussianDistribution*> (*iter);
					if(PLSC(*pGaussian) < occupancyThreshold)
					{
						removeLeaf(iter.getCurrentOctreeKey());
						numNodeRemoved++;
					}
				}
			}

			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			// Check Occupied
			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

			/** \brief Check if voxel at given point exist.
				* \param[in] point point to be checked
				* \return "true" if voxel exist; "false" otherwise
				*/
			bool
			isVoxelOccupiedAtPoint (const pcl::PointXYZ& point) const
			{
				pcl::octree::OctreeKey key;

				// generate key for point
				this->genOctreeKeyforPoint (point, key);

				return (isPointWithinBoundingBox (point) && isVoxelOccupiedAtKey(key));
			}

			/** \brief Check if voxel at given point coordinates exist.
				* \param[in] pointX X coordinate of point to be checked
				* \param[in] pointY Y coordinate of point to be checked
				* \param[in] pointZ Z coordinate of point to be checked
				* \return "true" if voxel exist; "false" otherwise
				*/
			bool
			isVoxelOccupiedAtPoint (const double pointX, const double pointY, const double pointZ) const
			{
				pcl::octree::OctreeKey key;

				// generate key for point
				this->genOctreeKeyforPoint (pointX, pointY, pointZ, key);

				return isVoxelOccupiedAtKey(key);
			}

		protected:
			/** \brief Check if voxel at given octree key.
				* \param[in] key key to be checked
				* \return "true" if voxel exist; "false" otherwise
				*/
			bool
			isVoxelOccupiedAtKey(const pcl::octree::OctreeKey &key) const
			{
				// search for key in octree
				if(LeafNode* pLeaf = this->findLeaf(key))
				{
					return PLSC(pLeaf->getData()) > occupancyThreshold_;
				}
				return false;
			}

		public:
			/** \brief Get the mean and the inverse variance at a given leaf node iterator
				* \param[in] iter leaf node iterator
				* \param[in] center node center
				* \param[out] pGaussian Gaussian distribution
				* \return "true" if voxel exist; "false" otherwise
			*/
			bool
			isVoxelOccupiedAtLeafNode(ConstLeafNodeIterator &iter, 
											  pcl::PointXYZ &center) const
			{
				pcl::octree::OctreeKey key = iter.getCurrentOctreeKey();

				// retreive data
				if(isVoxelOccupiedAtKey(key))
				{
					genLeafNodeCenterFromOctreeKey(key, center);
					return true;
				}

				return false;
			}

			bool
			isVoxelOccupiedAtNode(const pcl::octree::OctreeIteratorBase &iter) const
			{
				if(!iter.isLeafNode()) return false;

				pcl::octree::OctreeKey key = iter.getCurrentOctreeKey();

				// retreive data
				return isVoxelOccupiedAtKey(key);
			}

			/** \brief Get a pcl::PointXYZ vector of centers of all occupied voxels.
				* \param[out] voxelCenterList results are written to this vector of pcl::PointXYZ elements
				* \return number of occupied voxels
				*/
			int
			getOccupiedVoxelCenters(AlignedPointTVector &voxelCenterList) const
			{
				pcl::octree::OctreeKey key;
				key.x = key.y = key.z = 0;

				voxelCenterList.clear ();

				return getOccupiedVoxelCentersRecursive(this->rootNode_, key, voxelCenterList);
			}

			/** \brief Get a pcl::PointXYZ vector of centers of all occupied voxels.
				* \param[in] occupancyThreshod occupancy threshold
				* \param[out] voxelCenterList results are written to this vector of pcl::PointXYZ elements
				* \return number of occupied voxels
				*/
			int
			getOccupiedVoxelCenters(const Scalar occupancyThreshod, AlignedPointTVector &voxelCenterList) const
			{
				setOccupancyThreshold(occupancyThreshod);

				pcl::octree::OctreeKey key;
				key.x = key.y = key.z = 0;

				voxelCenterList.clear ();

				return getOccupiedVoxelCentersRecursive(this->rootNode_, key, voxelCenterList);
			}

			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			// Get Gaussian Distribution
			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			/** \brief Get the mean and the inverse variance at a given point.
				* \param[in] point point to be checked
				* \param[out] pGaussian Gaussian distribution
				* \return "true" if voxel exist; "false" otherwise
			*/
			bool
			getGaussianDistribution (const pcl::PointXYZ& point, GaussianDistribution gaussian) const
			{
				pcl::octree::OctreeKey key;

				// generate key for point
				this->genOctreeKeyforPoint (point, key);

				return (isPointWithinBoundingBox (point) && getGaussianDistribution(key, gaussian));
			}

			/** \brief Get the mean and the inverse variance at a given point.
				* \param[in] pointX X coordinate of point to be checked
				* \param[in] pointY Y coordinate of point to be checked
				* \param[in] pointZ Z coordinate of point to be checked
				* \param[out] pGaussian Gaussian distribution
				* \return "true" if voxel exist; "false" otherwise
			*/
			bool
			getGaussianDistribution(const double pointX, const double pointY, const double pointZ,
											GaussianDistribution gaussian) const
			{
				pcl::octree::OctreeKey key;

				// generate key for point
				this->genOctreeKeyforPoint (pointX, pointY, pointZ, key);

				return getGaussianDistribution(key, gaussian);
			}

		protected:
			/** \brief Get the mean and the inverse variance at a given octree key.
				* \param[in] key key to be checked
				* \param[out] pGaussian Gaussian distribution
				*/
			bool
			getGaussianDistribution(const pcl::octree::OctreeKey &key, GaussianDistribution &gaussian) const
			{
				// search for key in octree
				if(LeafT* pLeaf = this->findLeaf(key))
				{
					gaussian = pLeaf->getData();
					return true;
				}
				return false;
			}

		public:
			/** \brief Get the mean and the inverse variance at a given leaf node iterator
				* \param[in] iter leaf node iterator
				* \param[in] center node center
				* \param[out] pGaussian Gaussian distribution
				* \return "true" if voxel exist; "false" otherwise
			*/
			bool
			getGaussianDistributionAtLeafNode(ConstLeafNodeIterator &iter, 
														 pcl::PointXYZ &center, GaussianDistribution &gaussian) const
			{
				pcl::octree::OctreeKey key = iter.getCurrentOctreeKey();

				// retreive data
				if(getGaussianDistribution(key, gaussian))
				{
					genLeafNodeCenterFromOctreeKey(key, center);
					return true;
				}

				return false;
			}

			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			// Get occupancy
			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

			/** \brief Get the occupancy at a given point.
				* \param[in] point point to be checked
				* \param[out] occupancy occupancy
				* \return "true" if voxel exist; "false" otherwise
				*/
			bool
			getOccupancy (const pcl::PointXYZ& point, Scalar &occupancy) const
			{
				pcl::octree::OctreeKey key;

				// generate key for point
				this->genOctreeKeyforPoint (point, key);

				return (isPointWithinBoundingBox (point) && getOccupancy(key, occupancy));
			}

			/** \brief Get the occupancy at a given point.
				* \param[in] pointX X coordinate of point to be checked
				* \param[in] pointY Y coordinate of point to be checked
				* \param[in] pointZ Z coordinate of point to be checked
				* \param[out] occupancy occupancy
				* \return "true" if voxel exist; "false" otherwise
			*/
			bool
			getOccupancy(const double pointX, const double pointY, const double pointZ,
							 Scalar &occupancy) const
			{
				pcl::octree::OctreeKey key;

				// generate key for point
				this->genOctreeKeyforPoint (pointX, pointY, pointZ, key);

				return getOccupancy(key, occupancy);
			}

		protected:
			/** \brief Get the occupancy at a given point.
			  * \param[in] point point to be checked
			  * \param[out] occupancy occupancy
			  * \return "true" if voxel exist; "false" otherwise
			  */
			bool
			getOccupancy (const pcl::octree::OctreeKey &key, Scalar &occupancy) const
			{
				// search for key in octree
				if(LeafT* pLeaf = this->findLeaf(key))
				{
					occupancy = PLSC(pLeaf->getData());
					return true;
				}
				return false;
			}

		public:
			/** \brief Get the occupancy at a given leaf node iterator
				* \param[in] iter leaf node iterator
				* \param[in] center node center
				* \param[out] occupancy occupancy
				*/
			bool
			getOccupancyAtLeafNode(ConstLeafNodeIterator &iter, 
											pcl::PointXYZ &center, Scalar &occupancy) const
			{
				pcl::octree::OctreeKey key = iter.getCurrentOctreeKey();

				// retreive data
				if(getOccupancy(key, occupancy))
				{
					genLeafNodeCenterFromOctreeKey(key, center);
					return true;
				}

				return false;
			}

			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			// Probabilistic Least Square Classification
			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      protected:
			/** \brief Calculate occupancy by probabilistic least square classification
				* \param[in] pGaussian Gaussian distribution
				* \return occupancy
				*/
			inline Scalar
			PLSC(const GaussianDistribution &gaussian) const
			{
				return phi((PLSC_mean_ - gaussian.getMean()) / sqrt(gaussian.getVariance() + PLSC_variance_));
			}

			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			// Constructor
			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		public:
        /** \brief Octree pointcloud constructor.
          * \param[in] resolution octree resolution at lowest octree level
          */
        OctreeGPOM (const double resolution, 
								  const Scalar PLSC_mean		= (Scalar) 0.05f,
								  const Scalar PLSC_variance	= (Scalar) 0.0001f)
			: OctreeT (), resolution_ (resolution), 
			  minX_ (0.0f), maxX_ (resolution), 
			  minY_ (0.0f), maxY_ (resolution), 
			  minZ_ (0.0f), maxZ_ (resolution),
			  boundingBoxDefined_ (false),
			  occupancyThreshold_ (0.8f),
			  PLSC_mean_(PLSC_mean),
			  PLSC_variance_(PLSC_variance)
		{
			assert ( resolution > 0.0f );
		}


        /** \brief Empty deconstructor. */
        virtual
        ~OctreeGPOM ()
		{
		}

        /** \brief Set/change the octree voxel resolution
          * \param[in] resolution side length of voxels at lowest tree level
          */
        inline void
        setResolution (double resolution)
        {
			// octree needs to be empty to change its resolution
			assert( this->leafCount_ == 0 );

			resolution_ = resolution;

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
         *  \return depth: maximum depth of octree
         * */
        inline unsigned int
        getTreeDepth () const
        {
			return this->octreeDepth_;
        }

        /** \brief Set/change the occupancy threshold
          * \param[in] occupancyThreshold
          */
        inline void
        setOccupancyThreshold(const Scalar occupancyThreshold)
        {
			  occupancyThreshold_ = occupancyThreshold;
        }

        /** \brief Get the occupancy threshold
          * \return occupancy threshold
          */
        inline Scalar
        getOccupancyThreshold() const
        {
			  return occupancyThreshold_;
        }

       /** \brief Set/change the probabilistic least squre classification parameters
          * \param[in] PLSC_mean mean
          * \param[in] PLSC_variance variance
          */
        inline void
        setPLSCParameters(const Scalar PLSC_mean, const Scalar PLSC_variance)
        {
			PLSC_mean_		= PLSC_mean;
			PLSC_variance_	= PLSC_variance;
        }

        /** \brief Get the probabilistic least squre classification parameters
          * \param[out] PLSC_mean mean
          * \param[out] PLSC_variance variance
          */
        inline void
        getOccupancyThreshold(Scalar &PLSC_mean, Scalar &PLSC_variance) const
        {
			PLSC_mean			= PLSC_mean_;
			PLSC_variance		= PLSC_variance_;
        }


        /** \brief Get a pcl::PointXYZ vector of centers of voxels intersected by a line segment.
          * This returns a approximation of the actual intersected voxels by walking
          * along the line with small steps. Voxels are ordered, from closest to
          * furthest w.r.t. the origin.
          * \param[in] origin origin of the line segment
          * \param[in] end end of the line segment
          * \param[out] voxel_center_list results are written to this vector of pcl::PointXYZ elements
          * \param[in] precision determines the size of the steps: step_size = octree_resolution x precision
          * \return number of intersected voxels
          */
        int
        getApproxIntersectedVoxelCentersBySegment (const Eigen::Vector3f& origin, 
																	const Eigen::Vector3f& end,
																	AlignedPointTVector &voxel_center_list,
																	float precision)
		{
			Eigen::Vector3f direction = end - origin;
			float norm = direction.norm ();
			direction.normalize ();

			const float step_size = static_cast<const float> (resolution_) * precision;

			// Ensure we get at least one step for the first voxel.
			const int nsteps = std::max (1, static_cast<int> (norm / step_size));

			pcl::octree::OctreeKey prev_key;

			bool bkeyDefined = false;

			// Walk along the line segment with small steps.
			for (int i = 0; i < nsteps; ++i)
			{
				Eigen::Vector3f p = origin + (direction * step_size * static_cast<const float> (i));

				pcl::PointXYZ octree_p;
				octree_p.x = p.x ();
				octree_p.y = p.y ();
				octree_p.z = p.z ();

				pcl::octree::OctreeKey key;
				this->genOctreeKeyforPoint (octree_p, key);

				// Not a new key, still the same voxel.
				if ((key == prev_key) && (bkeyDefined) )
				  continue;

				prev_key = key;
				bkeyDefined = true;

				// check if it is occupied
				if(isVoxelOccupiedAtKey(key))
				{
					pcl::PointXYZ center;
					genLeafNodeCenterFromOctreeKey (key, center);
					voxel_center_list.push_back (center);
				}
			}

			pcl::octree::OctreeKey end_key;
			pcl::PointXYZ end_p;
			end_p.x = end.x ();
			end_p.y = end.y ();
			end_p.z = end.z ();
			this->genOctreeKeyforPoint (end_p, end_key);
			if (!(end_key == prev_key))
			{
				// check if it is occupied
				if(isVoxelOccupiedAtKey(end_key))
				{
					pcl::PointXYZ center;
					genLeafNodeCenterFromOctreeKey (end_key, center);
					voxel_center_list.push_back (center);
				}
			}
			
			return (static_cast<int> (voxel_center_list.size ()));
		}


        /** \brief Delete leaf node / voxel at given point
          * \param[in] point point addressing the voxel to be deleted.
          */
        void
        deleteVoxelAtPoint (const pcl::PointXYZ& point)
		{
			pcl::octree::OctreeKey key;

			// generate key for point
			this->genOctreeKeyforPoint (point, key);

			this->removeLeaf (key);
		}


        /** \brief Delete the octree structure and its leaf nodes. */
        void
        deleteTree ()
        {
          // reset bounding box
          minX_ = minY_ = maxY_ = minZ_ = maxZ_ = 0;
          this->boundingBoxDefined_ = false;

          OctreeT::deleteTree ();
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Bounding box methods
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        /** \brief Define bounding box for octree
          * \note Bounding box cannot be changed once the octree contains elements.
          * \param[in] minX X coordinate of lower bounding box corner
          * \param[in] minY Y coordinate of lower bounding box corner
          * \param[in] minZ Z coordinate of lower bounding box corner
          * \param[in] maxX X coordinate of upper bounding box corner
          * \param[in] maxY Y coordinate of upper bounding box corner
          * \param[in] maxZ Z coordinate of upper bounding box corner
          */
        void
        defineBoundingBox (const double minX, const double minY, const double minZ, 
									const double maxX, const double maxY, const double maxZ)
		{
			// bounding box cannot be changed once the octree contains elements
			assert (this->leafCount_ == 0);

			assert (maxX >= minX);
			assert (maxY >= minY);
			assert (maxZ >= minZ);

			minX_ = minX;
			maxX_ = maxX;

			minY_ = minY;
			maxY_ = maxY;

			minZ_ = minZ;
			maxZ_ = maxZ;

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
          * \param[in] maxX X coordinate of upper bounding box corner
          * \param[in] maxY Y coordinate of upper bounding box corner
          * \param[in] maxZ Z coordinate of upper bounding box corner
          */
        void
        defineBoundingBox (const double maxX, const double maxY, const double maxZ)
		{
			// bounding box cannot be changed once the octree contains elements
			assert (this->leafCount_ == 0);

			assert (maxX >= 0.0f);
			assert (maxY >= 0.0f);
			assert (maxZ >= 0.0f);

			minX_ = 0.0f;
			maxX_ = maxX;

			minY_ = 0.0f;
			maxY_ = maxY;

			minZ_ = 0.0f;
			maxZ_ = maxZ;

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
          * \param[in] cubeLen side length of bounding box cube.
          */
        void
        defineBoundingBox (const double cubeLen)
		{
			// bounding box cannot be changed once the octree contains elements
			assert (this->leafCount_ == 0);

			assert (cubeLen >= 0.0f);

			minX_ = 0.0f;
			maxX_ = cubeLen;

			minY_ = 0.0f;
			maxY_ = cubeLen;

			minZ_ = 0.0f;
			maxZ_ = cubeLen;

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
          * \param[in] minX X coordinate of lower bounding box corner
          * \param[in] minY Y coordinate of lower bounding box corner
          * \param[in] minZ Z coordinate of lower bounding box corner
          * \param[in] maxX X coordinate of upper bounding box corner
          * \param[in] maxY Y coordinate of upper bounding box corner
          * \param[in] maxZ Z coordinate of upper bounding box corner
          */
        void
        getBoundingBox (double& minX, double& minY, double& minZ, 
									   double& maxX, double& maxY, double& maxZ) const
		{
			minX = minX_;
			minY = minY_;
			minZ = minZ_;

			maxX = maxX_;
			maxY = maxY_;
			maxZ = maxZ_;
		}

        /** \brief Calculates the squared diameter of a voxel at given tree depth
          * \param[in] treeDepth depth/level in octree
          * \return squared diameter
          */
        double
        getVoxelSquaredDiameter (unsigned int treeDepth) const
		{
			// return the squared side length of the voxel cube as a function of the octree depth
			return (getVoxelSquaredSideLen (treeDepth) * 3);
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
          * \param[in] treeDepth depth/level in octree
          * \return squared voxel cube side length
          */
        double
        getVoxelSquaredSideLen (unsigned int treeDepth) const
		{
			double sideLen;

			// side length of the voxel cube increases exponentially with the octree depth
			sideLen = this->resolution_ * static_cast<double>(1 << (this->octreeDepth_ - treeDepth));

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
			getVoxelBounds (pcl::octree::OctreeIteratorBase<GaussianDistribution, OctreeT>& iterator, 
								 Eigen::Vector3f &min_pt, 
								 Eigen::Vector3f &max_pt)
        {
			  this->genVoxelBoundsFromOctreeKey(iterator.getCurrentOctreeKey(), 
															iterator.getCurrentOctreeDepth(), 
															min_pt, max_pt);
        }


      protected:
        /** \brief Find octree leaf node at a given point
          * \param[in] point query point
          * \return pointer to leaf node. If leaf node does not exist, pointer is 0.
          */
        LeafT*
        findLeafAtPoint (const pcl::PointXYZ& point) const 
		{
			pcl::octree::OctreeKey key;

			// generate key for point
			this->genOctreeKeyforPoint (point, key);

			return (this->findLeaf (key));
		}

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Protected octree methods based on octree keys
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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
			maxKeyX = static_cast<unsigned int> ((maxX_ - minX_) / resolution_);
			maxKeyY = static_cast<unsigned int> ((maxY_ - minY_) / resolution_);
			maxKeyZ = static_cast<unsigned int> ((maxZ_ - minZ_) / resolution_);

			// find maximum amount of keys
			maxVoxels = max (max (max (maxKeyX, maxKeyY), maxKeyZ), static_cast<unsigned int> (2));


			// tree depth == amount of bits of maxVoxels
			this->octreeDepth_ = max ((min (static_cast<unsigned int> (OCT_MAXTREEDEPTH), 
													  static_cast<unsigned int> (ceil (this->Log2 (maxVoxels)-minValue)))),
												static_cast<unsigned int> (0));

			octreeSideLen = static_cast<double> (1 << this->octreeDepth_) * resolution_-minValue;

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

        /** \brief Checks if given point is within the bounding box of the octree
         * \param[in] pointIdx_arg point to be checked for bounding box violations
         * \return "true" - no bound violation
         */
        inline bool isPointWithinBoundingBox (const pcl::PointXYZ& point) const
        {
          return (! ( (point.x <  minX_) || (point.y <  minY_) || (point.z <  minZ_) ||
				          (point.x >= maxX_) || (point.y >= maxY_) || (point.z >= maxZ_)));
        }

        /** \brief Grow the bounding box/octree until point fits
          * \param[in] point point that should be within bounding box;
          */
        void
        adoptBoundingBoxToPoint (const pcl::PointXYZ& point)
		{
			const float minValue = std::numeric_limits<float>::epsilon();

			// increase octree size until point fits into bounding box
			while (true)
			{
				bool bLowerBoundViolationX = (point.x < minX_);
				bool bLowerBoundViolationY = (point.y < minY_);
				bool bLowerBoundViolationZ = (point.z < minZ_);

				bool bUpperBoundViolationX = (point.x >= maxX_);
				bool bUpperBoundViolationY = (point.y >= maxY_);
				bool bUpperBoundViolationZ = (point.z >= maxZ_);

				// do we violate any bounds?
				if (bLowerBoundViolationX || bLowerBoundViolationY || bLowerBoundViolationZ || 
					bUpperBoundViolationX	|| bUpperBoundViolationY || bUpperBoundViolationZ)
				{
					if (boundingBoxDefined_)
					{
						double octreeSideLen;
						unsigned char childIdx;

						// octree not empty - we add another tree level and thus increase its size by a factor of 2*2*2
						childIdx = static_cast<unsigned char> (((!bUpperBoundViolationX) << 2) | ((!bUpperBoundViolationY) << 1) | ((!bUpperBoundViolationZ)));

						BranchNode* newRootBranch;

						newRootBranch = this->branchNodePool_.popNode();
						this->branchCount_++;

						this->setBranchChildPtr (*newRootBranch, childIdx, this->rootNode_);

						this->rootNode_ = newRootBranch;

						octreeSideLen = static_cast<double> (1 << this->octreeDepth_) * resolution_;

						if (!bUpperBoundViolationX)				  minX_ -= octreeSideLen;
						if (!bUpperBoundViolationY)				  minY_ -= octreeSideLen;
						if (!bUpperBoundViolationZ)				  minZ_ -= octreeSideLen;

						// configure tree depth of octree
						this->octreeDepth_ ++;
						this->setTreeDepth (this->octreeDepth_);

						// recalculate bounding box width
						octreeSideLen = static_cast<double> (1 << this->octreeDepth_) * resolution_ - minValue;

						// increase octree bounding box
						maxX_ = minX_ + octreeSideLen;
						maxY_ = minY_ + octreeSideLen;
						maxZ_ = minZ_ + octreeSideLen;
					}
					// bounding box is not defined - set it to point position
					else
					{
						// octree is empty - we set the center of the bounding box to our first pixel
						this->minX_ = point.x - this->resolution_ / 2;
						this->minY_ = point.y - this->resolution_ / 2;
						this->minZ_ = point.z - this->resolution_ / 2;

						this->maxX_ = point.x + this->resolution_ / 2;
						this->maxY_ = point.y + this->resolution_ / 2;
						this->maxZ_ = point.z + this->resolution_ / 2;

						getKeyBitSize();
						
						boundingBoxDefined_ = true;
					}

				}
				else
					// no bound violations anymore - leave while loop
					break;
			}
		}

        /** \brief Generate octree key for voxel at a given point
          * \param[in] point the point addressing a voxel
          * \param[out] key write octree key to this reference
          */
        void
        genOctreeKeyforPoint (const pcl::PointXYZ & point, pcl::octree::OctreeKey &key) const
		{
			// calculate integer key for point coordinates
			key.x = static_cast<unsigned int> ((point.x - this->minX_) / this->resolution_);
			key.y = static_cast<unsigned int> ((point.y - this->minY_) / this->resolution_);
			key.z = static_cast<unsigned int> ((point.z - this->minZ_) / this->resolution_);
		}

        /** \brief Generate octree key for voxel at a given point
          * \param[in] pointX X coordinate of point addressing a voxel
          * \param[in] pointY Y coordinate of point addressing a voxel
          * \param[in] pointZ Z coordinate of point addressing a voxel
          * \param[out] key write octree key to this reference
          */
        void
        genOctreeKeyforPoint (const double pointX, const double pointY, const double pointZ,
												 pcl::octree::OctreeKey & key) const
		{
			pcl::PointXYZ tempPoint;

			tempPoint.x = static_cast<float> (pointX);
			tempPoint.y = static_cast<float> (pointY);
			tempPoint.z = static_cast<float> (pointZ);

			// generate key for point
			genOctreeKeyforPoint (tempPoint, key);
		}


        /** \brief Generate a point at center of leaf node voxel
          * \param[in] key octree key addressing a leaf node.
          * \param[out] point write leaf node voxel center to this point reference
          */
        void
        genLeafNodeCenterFromOctreeKey (const pcl::octree::OctreeKey & key, pcl::PointXYZ& point) const
		{
			// define point to leaf node voxel center
			point.x = static_cast<float> ((static_cast<double> (key.x) + 0.5f) * this->resolution_ + this->minX_);
			point.y = static_cast<float> ((static_cast<double> (key.y) + 0.5f) * this->resolution_ + this->minY_);
			point.z = static_cast<float> ((static_cast<double> (key.z) + 0.5f) * this->resolution_ + this->minZ_);
		}

        /** \brief Generate a point at center of octree voxel at given tree level
          * \param[in] key octree key addressing an octree node.
          * \param[in] treeDepth octree depth of query voxel
          * \param[out] point write leaf node center point to this reference
          */
        void
        genVoxelCenterFromOctreeKey (const pcl::octree::OctreeKey & key, unsigned int treeDepth, pcl::PointXYZ& point) const
		{
			// generate point for voxel center defined by treedepth (bitLen) and key
			point.x = static_cast<float> ((static_cast <double> (key.x) + 0.5f) * (this->resolution_ * static_cast<double> (1 << (this->octreeDepth_ - treeDepth))) + this->minX_);
			point.y = static_cast<float> ((static_cast <double> (key.y) + 0.5f) * (this->resolution_ * static_cast<double> (1 << (this->octreeDepth_ - treeDepth))) + this->minY_);
			point.z = static_cast<float> ((static_cast <double> (key.z) + 0.5f) * (this->resolution_ * static_cast<double> (1 << (this->octreeDepth_ - treeDepth))) + this->minZ_);
		}

        /** \brief Generate bounds of an octree voxel using octree key and tree depth arguments
          * \param[in] key octree key addressing an octree node.
          * \param[in] treeDepth octree depth of query voxel
          * \param[out] min_pt lower bound of voxel
          * \param[out] max_pt upper bound of voxel
          */
        void
        genVoxelBoundsFromOctreeKey (const pcl::octree::OctreeKey & key, unsigned int treeDepth, Eigen::Vector3f &min_pt,
																	Eigen::Vector3f &max_pt) const
		{
			// calculate voxel size of current tree depth
			double voxel_side_len = this->resolution_ * static_cast<double> (1 << (this->octreeDepth_ - treeDepth));

			// calculate voxel bounds
			min_pt (0) = static_cast<float> (static_cast<double> (key.x) * voxel_side_len + this->minX_);
			min_pt (1) = static_cast<float> (static_cast<double> (key.y) * voxel_side_len + this->minY_);
			min_pt (2) = static_cast<float> (static_cast<double> (key.z) * voxel_side_len + this->minZ_);

			max_pt (0) = static_cast<float> (static_cast<double> (key.x + 1) * voxel_side_len + this->minX_);
			max_pt (1) = static_cast<float> (static_cast<double> (key.y + 1) * voxel_side_len + this->minY_);
			max_pt (2) = static_cast<float> (static_cast<double> (key.z + 1) * voxel_side_len + this->minZ_);
		}

        /** \brief Recursively search the tree for all leaf nodes and return a vector of voxel centers.
          * \param[in] node current octree node to be explored
          * \param[in] key octree key addressing a leaf node.
          * \param[out] voxelCenterList results are written to this vector of pcl::PointXYZ elements
          * \return number of voxels found
          */
        int
        getOccupiedVoxelCentersRecursive (const BranchNode* node, const pcl::octree::OctreeKey& key,
														AlignedPointTVector &voxelCenterList) const
		{
			// child iterator
			unsigned char childIdx;

			int voxelCount = 0;

			// iterate over all children
			for (childIdx = 0; childIdx < 8; childIdx++)
			{
				if (!this->branchHasChild (*node, childIdx))
					continue;

				const pcl::octree::OctreeNode * childNode;
				childNode = this->getBranchChildPtr (*node, childIdx);

				// generate new key for current branch voxel
				pcl::octree::OctreeKey newKey;
				newKey.x = (key.x << 1) | (!!(childIdx & (1 << 2)));
				newKey.y = (key.y << 1) | (!!(childIdx & (1 << 1)));
				newKey.z = (key.z << 1) | (!!(childIdx & (1 << 0)));

				switch (childNode->getNodeType ())
				{
					case pcl::octree::BRANCH_NODE:
					{
						// recursively proceed with indexed child branch
						voxelCount += getOccupiedVoxelCentersRecursive (static_cast<const BranchNode*> (childNode), newKey, voxelCenterList);
						break;
					}
					case pcl::octree::LEAF_NODE:
					{
						// check if it is occupied
						if(isVoxelOccupiedAtKey(newKey))
						{
							pcl::PointXYZ newPoint;

							genLeafNodeCenterFromOctreeKey (newKey, newPoint);
							voxelCenterList.push_back (newPoint);

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

        /** \brief Flag indicating if octree has defined bounding box. */
        bool boundingBoxDefined_;

        /** \brief if the occupancy is greater than this threshold, it is thought of being occupied */
		  Scalar occupancyThreshold_; // TODO: varianceThreshold_ ?

        /** \brief mean for PLSC. */
        Scalar PLSC_mean_;

        /** \brief variance for PLSC. */
        Scalar PLSC_variance_;
    };
}

#endif