/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __OPENCV_ACL_TYPE_HPP__
#define __OPENCV_ACL_TYPE_HPP__

#include <memory>
#include <vector>

#include "acl/acl.h"
#include "acl/acl_op_compiler.h"
#include "operator_desc.hpp"

#include "opencv2/core/core.hpp"
using namespace std;

namespace cv
{
    namespace acl 
    {
        /* Memory alignment */
        enum ALIGNMENT { MEMORY_UNALIGNED = 0, MEMORY_ALIGN = 1};
        enum { MAGIC_VAL  = 0x42FF0000, AUTO_STEP = 0, CONTINUOUS_FLAG = CV_MAT_CONT_FLAG, SUBMATRIX_FLAG = CV_SUBMAT_FLAG };
        enum { MAGIC_MASK = 0xFFFF0000, TYPE_MASK = 0x00000FFF, DEPTH_MASK = 7 };

        enum DeviceType
        {
            ACL_DEVICE_TYPE_DEFAULT     = (1 << 0),
            ACL_DEVICE_TYPE_200         = (1 << 1),
            ACL_DEVICE_TYPE_ACCELERATOR = (1 << 3),
        };

        #define aclStream aclrtStream

        //////////////////////////////// aclEnv ////////////////////////////////
        class CV_EXPORTS aclEnv
        {
            public:
                aclEnv();
                aclEnv(const char* config_path);
                static aclEnv* get_acl_env(const char* config_path);
                int get_device_count();
                int *refcount;
                ~aclEnv();

            private:
                uint32_t _device_count;
        };

        //////////////////////////////// aclCxt ////////////////////////////////
        class CV_EXPORTS aclCxt
        {
            public:
                aclCxt();
                aclCxt(int device_id);

                CV_EXPORTS aclrtContext* get_context();
                CV_EXPORTS void set_current_context();

                CV_EXPORTS void create_stream(int count = 1);
                CV_EXPORTS aclStream get_stream(const size_t index = 0);

                ~aclCxt();

            private:
                int32_t _device_id;
                aclrtContext* _context;
                std::vector<aclStream> _acl_streams;
        };

        //////////////////////////////// device ////////////////////////////////
        CV_EXPORTS aclCxt *set_device(const char* config_path, int device_id = 0, int stream_count = 1);
        CV_EXPORTS void release_device(aclCxt* context);


        class CV_EXPORTS aclMatExpr;
        //////////////////////////////// aclMat ////////////////////////////////
        class CV_EXPORTS aclMat
        {
            public:
                //! default constructor
                aclMat();
                //! constructs aclMatrix of the specified size and type (_type is CV_8UC1,  CV_16FC1 etc.)
                aclMat(int rows, int cols, int type, aclCxt *acl_context, ALIGNMENT config = MEMORY_UNALIGNED, aclrtMemMallocPolicy policy = ACL_MEM_MALLOC_HUGE_FIRST);
                aclMat(Size size, int type, aclCxt *acl_context, ALIGNMENT config = MEMORY_UNALIGNED, aclrtMemMallocPolicy policy = ACL_MEM_MALLOC_HUGE_FIRST);
                //! constucts aclMatrix and fills it with the specified value _s.
                //aclMat(int rows, int cols, int type, const Scalar &s, aclCxt *acl_context, aclrtMemMallocPolicy policy = ACL_MEM_MALLOC_HUGE_FIRST);
                //aclMat(Size size, int type, const Scalar &s);
                //! copy constructor
                aclMat(const aclMat &m);
                //! constructor for aclMatrix headers pointing to user-allocated data
                aclMat(int rows, int cols, int type, void *data, aclCxt* acl_context, ALIGNMENT config = MEMORY_UNALIGNED, size_t step = Mat::AUTO_STEP);
                aclMat(Size size, int type, void *data, aclCxt* acl_context, ALIGNMENT config = MEMORY_UNALIGNED, size_t step = Mat::AUTO_STEP);

                //! creates a matrix header for a part of the bigger matrix
                aclMat(const aclMat &m, const Range &rowRange, const Range &colRange = Range::all());
                aclMat(const aclMat &m, const Rect &roi);

                //! builds aclMat from Mat. Perfom blocking upload to device.
                aclMat (const Mat &m, aclCxt* acl_context, ALIGNMENT config = MEMORY_UNALIGNED, aclrtMemMallocPolicy policy = ACL_MEM_MALLOC_HUGE_FIRST);
                //! destructor - calls release()
                ~aclMat();
                //! assignment operators shallow copy
                aclMat &operator=(const aclMat &m);
                //! assignment operator. Perfom blocking upload to device.
                aclMat &operator=(const Mat &m);
                aclMat &operator=(const aclMatExpr& expr);
        
                //! pefroms blocking upload data to aclMat.
                CV_EXPORTS void upload(const Mat &m, ALIGNMENT config = MEMORY_UNALIGNED);
                CV_EXPORTS void upload(const Mat &m, aclStream stream, ALIGNMENT config = MEMORY_UNALIGNED);

                CV_EXPORTS void download(Mat &m, ALIGNMENT config = MEMORY_UNALIGNED) const;
                CV_EXPORTS void download(Mat &m, aclStream stream, ALIGNMENT config = MEMORY_UNALIGNED) const;
               
                //! downloads data from device to host memory. Blocking calls.
                operator Mat() const;

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
/*
                //! returns deep copy of the aclMatrix, i.e. the data is copied
                aclMat clone() const;

                //! copies those aclMatrix elements to "m" that are marked with non-zero mask elements.
                // It calls m.create(this->size(), this->type()).
                // It supports any data type
                void copyTo( aclMat &m, const aclMat &mask = aclMat()) const;

                //! converts aclMatrix to another datatype with optional scalng. See cvConvertScale.
                void convertTo( aclMat &m, int rtype, double alpha = 1, double beta = 0 ) const;

                void assignTo( aclMat &m, int type = -1 ) const;

                //! sets every aclMatrix element to s
                aclMat& operator = (const Scalar &s);
                //! sets some of the aclMatrix elements to s, according to the mask
                aclMat& setTo(const Scalar &s, const aclMat &mask = aclMat());
                //! creates alternative aclMatrix header for the same data, with different
                // number of channels and/or different number of rows. see cvReshape.
                aclMat reshape(int cn, int rows = 0) const;
*/
                //! allocates new aclMatrix data unless the aclMatrix already has specified size and type.
                // previous data is unreferenced if needed.
                void create(int rows, int cols, int type, ALIGNMENT config = MEMORY_UNALIGNED, aclrtMemMallocPolicy policy = ACL_MEM_MALLOC_HUGE_FIRST);
                void create(Size size, int type, ALIGNMENT config = MEMORY_UNALIGNED, aclrtMemMallocPolicy policy = ACL_MEM_MALLOC_HUGE_FIRST);

                //! allocates new aclMatrix with specified device memory type.
                void createEx(int rows, int cols, int type, ALIGNMENT config = MEMORY_UNALIGNED, aclrtMemMallocPolicy policy = ACL_MEM_MALLOC_HUGE_FIRST);
                void createEx(Size size, int type, ALIGNMENT config = MEMORY_UNALIGNED, aclrtMemMallocPolicy policy = ACL_MEM_MALLOC_HUGE_FIRST);

                //! decreases reference counter;
                // deallocate the data when reference counter reaches 0.
                void release();
                //! swaps with other smart pointer
                void swap(aclMat &mat);

                //! extracts a rectangular sub-aclMatrix
                // (this is a generalized form of row, rowRange etc.)
                aclMat operator()( Range rowRange, Range colRange ) const;
                aclMat operator()( const Rect &roi ) const;


                aclMat& operator+=( const aclMat& m );
                aclMat& operator-=( const aclMat& m );
                aclMat& operator/=( const aclMat& m );
                aclMat& operator*=( const aclMat& m );
                aclMat& abs();
            /*
            */
                //! returns true if the aclMatrix data is continuous
                // (i.e. when there are no gaps between successive rows).
                // similar to CV_IS_aclMat_CONT(cvaclMat->type)
                bool isContinuous() const;
                //! returns element size in bytes,
                // similar to CV_ELEM_SIZE(cvMat->type)
                CV_EXPORTS size_t elemSize() const;
                //! returns the size of element channel in bytes.
                size_t elemSize1() const;
                //! returns element type, similar to CV_MAT_TYPE(cvMat->type)
                CV_EXPORTS int type() const;
                //! returns element type, i.e. 8UC3 returns 8UC4 because in acl
                //! 3 channels element actually use 4 channel space
                int acltype() const;
                //! returns element type, similar to CV_MAT_DEPTH(cvMat->type)
                int depth() const;
                //! returns element type, similar to CV_MAT_CN(cvMat->type)
                CV_EXPORTS int channels() const;
                //! returns element type, return 4 for 3 channels element,
                //!becuase 3 channels element actually use 4 channel space
                CV_EXPORTS int aclchannels() const;
                //! returns step/elemSize1()
                size_t step1() const;
                //! returns aclMatrix size:
                // width == number of columns, height == number of rows
                Size size() const;
                //! returns true if aclMatrix data is NULL
                bool empty() const;

                friend void swap(aclMat &a, aclMat &b);
                friend void ensureSizeIsEnough(int rows, int cols, int type, aclMat &m, ALIGNMENT config = MEMORY_UNALIGNED);
                friend void ensureSizeIsEnough(Size size, int type, aclMat &m, ALIGNMENT config = MEMORY_UNALIGNED);
                //! matrix transposition
                //aclMat t() const;

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
                //uchar *data;

                //! OpenCL context associated with the aclMat object.
                void *data; // TODO 

                //! pointer to the reference counter;
                // when aclMatrix points to user-allocated data, the pointer is NULL
                int *refcount;

                //! helper fields used in locateROI and adjustROI
                //datastart and dataend are not used in current version
                uchar *datastart;
                uchar *dataend;

                //add offset for handle ROI, calculated in byte
                int offset;
                //add wholerows and wholecols for the whole matrix, datastart and dataend are no longer used
                int wholerows;
                int wholecols;

                aclCxt *acl_context;
                size_t totalSize;

        };

    /* operator */

    OperatorDesc CreateOpDesc(const string opType, vector<aclMat>& input_Mat, vector<aclMat>& output_Mat, aclFormat format = ACL_FORMAT_NHWC);
    void compileAndRunop(OperatorDesc& opDesc, vector<aclDataBuffer *>& inputBuffers_, vector<aclDataBuffer *>& outputBuffers_, aclCxt *acl_context);
    void OneInAndOneOut(const aclMat& input, aclMat& output, const string opType);
    void TwoInAndOneOut(const aclMat& inputMat, const aclMat& inputMatOther, aclMat& outputMat, const string opType);
    aclMat& Runop(vector<aclMat>& input, vector<aclMat>& output, const string opType);
    }
}


#endif /* __OPENCV_ACL_HPP__ */
