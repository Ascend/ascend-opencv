#ifndef OPENCV_ACL_MAT_HPP
#define OPENCV_ACL_MAT_HPP

#include "acl/acl.h"
#include "acl_init.hpp"
#include "acl_type.hpp"
#include "opencv2/core.hpp"

namespace cv {
namespace acl {
//////////////////////////////// aclMat ////////////////////////////////
class CV_EXPORTS aclMat {
 public:
  aclMat();
  //! constructs aclMatrix of the specified size and type (_type is CV_8UC1,
  //! CV_16FC1 etc.)
  aclMat(int rows, int cols, int type, aclCxt *acl_context,
         ALIGNMENT config = MEMORY_UNALIGNED,
         MemMallocPolicy policy = MALLOC_NORMAL_ONLY);
  aclMat(Size size, int type, aclCxt *acl_context,
         ALIGNMENT config = MEMORY_UNALIGNED,
         MemMallocPolicy policy = MALLOC_NORMAL_ONLY);
  //! copy constructor
  aclMat(const aclMat &m);
  //! constructor for aclMatrix headers pointing to user-allocated data
  aclMat(int rows, int cols, int type, void *data, aclCxt *acl_context,
         ALIGNMENT config = MEMORY_UNALIGNED, size_t step = Mat::AUTO_STEP);
  aclMat(Size size, int type, void *data, aclCxt *acl_context,
         ALIGNMENT config = MEMORY_UNALIGNED, size_t step = Mat::AUTO_STEP);
  //! creates a matrix header for a part of the bigger matrix
  aclMat(const aclMat &m, const Range &rowRange,
         const Range &colRange = Range::all());
  aclMat(const aclMat &m, const Rect &roi);
  //! builds aclMat from Mat. Perfom blocking upload to device.
  aclMat(const Mat &m, aclCxt *acl_context, ALIGNMENT config = MEMORY_UNALIGNED,
         MemMallocPolicy policy = MALLOC_NORMAL_ONLY);
  //! destructor - calls release()
  ~aclMat();

  //! assignment operators shallow copy
  aclMat &operator=(const aclMat &m);
  //! assignment operator. Perfom blocking upload to device.
  aclMat &operator=(const Mat &m);

  //! pefroms blocking upload data to aclMat.
  void upload(const Mat &m, ALIGNMENT config = MEMORY_UNALIGNED);
  void upload(const Mat &m, aclStream stream,
              ALIGNMENT config = MEMORY_UNALIGNED);
  //! downloads data from device to host memory. Blocking calls.
  void download(Mat &m, ALIGNMENT config = MEMORY_UNALIGNED) const;
  void download(Mat &m, aclStream stream,
                ALIGNMENT config = MEMORY_UNALIGNED) const;

  operator Mat() const;
  aclMat clone() const;
  void copyTo(aclMat &dest) const;

  //! returns a new aclMatrix header for the specified row
  aclMat row(int y) const;
  //! returns a new aclMatrix header for the specified column
  aclMat col(int x) const;
  //! ... for the specified row span
  aclMat rowRange(int startrow, int endrow) const;
  aclMat rowRange(const Range &r) const;
  //! ... for the specified column span
  aclMat colRange(int startcol, int endcol) const;
  aclMat colRange(const Range &r) const;

  //! locates aclMatrix header within a parent aclMatrix. See below
  void locateROI(Size &wholeSize, Point &ofs) const;
  //! moves/resizes the current aclMatrix ROI inside the parent aclMatrix.
  aclMat &adjustROI(int dtop, int dbottom, int dleft, int dright);

  //! allocates new aclMatrix data unless the aclMatrix already has specified
  //! size and type.
  // previous data is unreferenced if needed.
  void create(int rows, int cols, int type, ALIGNMENT config = MEMORY_UNALIGNED,
              MemMallocPolicy policy = MALLOC_NORMAL_ONLY);
  void create(Size size, int type, ALIGNMENT config = MEMORY_UNALIGNED,
              MemMallocPolicy policy = MALLOC_NORMAL_ONLY);

  //! allocates new aclMatrix with specified device memory type.
  void createEx(int rows, int cols, int type,
                ALIGNMENT config = MEMORY_UNALIGNED,
                MemMallocPolicy policy = MALLOC_NORMAL_ONLY);
  void createEx(Size size, int type, ALIGNMENT config = MEMORY_UNALIGNED,
                MemMallocPolicy policy = MALLOC_NORMAL_ONLY);

  //! decreases reference counter;
  // deallocate the data when reference counter reaches 0.
  void release();

  //! swaps with other smart pointer
  void swap(aclMat &mat);

  //! extracts a rectangular sub-aclMatrix
  // (this is a generalized form of row, rowRange etc.)
  aclMat operator()(Range rowRange, Range colRange) const;
  aclMat operator()(const Rect &roi) const;

  aclMat &operator+=(const aclMat &m);
  aclMat &operator-=(const aclMat &m);
  aclMat &operator/=(const aclMat &m);
  aclMat &operator*=(const aclMat &m);

  //! returns true if the aclMatrix data is continuous
  // (i.e. when there are no gaps between successive rows).
  // similar to CV_IS_aclMat_CONT(cvaclMat->type)
  bool isContinuous() const;

  //! returns element size in bytes,
  // similar to CV_ELEM_SIZE(cvMat->type)
  size_t elemSize() const;
  //! returns the size of element channel in bytes.
  size_t elemSize1() const;

  //! returns element type, similar to CV_MAT_TYPE(cvMat->type)
  int type() const;
  //! returns element type, i.e. 8UC3 returns 8UC4 because in acl
  //! 3 channels element actually use 4 channel space
  int acltype() const;
  //! returns element type, similar to CV_MAT_DEPTH(cvMat->type)
  int depth() const;

  //! returns element type, similar to CV_MAT_CN(cvMat->type)
  int channels() const;
  //! returns element type, return 4 for 3 channels element,
  //! becuase 3 channels element actually use 4 channel space
  int aclchannels() const;

  //! returns step/elemSize1()
  size_t step1() const;
  //! returns aclMatrix size:
  // width == number of columns, height == number of rows
  Size size() const;
  //! returns true if aclMatrix data is NULL
  bool empty() const;

  friend void swap(aclMat &a, aclMat &b);
  friend void ensureSizeIsEnough(int rows, int cols, int type, aclMat &m,
                                 ALIGNMENT config = MEMORY_UNALIGNED);
  friend void ensureSizeIsEnough(Size size, int type, aclMat &m,
                                 ALIGNMENT config = MEMORY_UNALIGNED);

  /*! includes several bit-fields:
    - the magic signature
    - continuity flag
    - depth
    - number of channels
    */
  int flags;
  //! the number of rows and columns
  int rows, cols;
  //! a distance between successive rows in bytes; includes the gap if any
  size_t step;

  //! pointer to the data(ACL memory object)
  void *data;  

  //! pointer to the reference counter;
  // when aclMatrix points to user-allocated data, the pointer is NULL
  int *refcount;

  //! helper fields used in locateROI and adjustROI
  // datastart and dataend are not used in current version
  uchar *datastart;
  uchar *dataend;

  // add offset for handle ROI, calculated in byte
  int offset;
  // add wholerows and wholecols for the whole matrix, datastart and dataend are
  // no longer used
  int wholerows;
  int wholecols;

  aclCxt *acl_context;
  size_t totalSize;
};
} /* end of namespace acl */

} /* end of namespace cv */

#endif