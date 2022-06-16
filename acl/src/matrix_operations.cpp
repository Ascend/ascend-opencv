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
#include "precomp.hpp"


#define ALIGN 64
//#define GPU_MATRIX_MALLOC_STEP(step) (((step) + ALIGN - 1) / ALIGN) * ALIGN

// helper routines
namespace cv
{
    namespace acl
    {

        /* Memory alignment */
        static inline size_t alignSize(size_t sz, int n = ALIGN);
        
        void aclMat::upload(const Mat &m, ALIGNMENT config)
        {
            CV_Assert((config == ALIGNMENT::MEMORY_ALIGN) || (config == ALIGNMENT::MEMORY_UNALIGNED));
            if (config == ALIGNMENT::MEMORY_UNALIGNED)
            {
                CV_Assert(m.data && (this->step == m.step) && (this->rows == m.rows) && (this->cols == m.cols) && (this->type() == m.type()));
                aclrtMemcpy((void *)this->data, (m.step * m.rows), (void *)m.data, (m.step * m.rows), ACL_MEMCPY_HOST_TO_DEVICE);
            }
            else if (config == ALIGNMENT::MEMORY_ALIGN)
            {
                CV_Assert(m.data && (this->rows == m.rows) && (this->cols == m.cols) && (this->type() == m.type()));
                aclrtMemcpy2d((void *)this->data, this->step, (void *)m.data, m.step, m.cols * m.elemSize(), m.rows, ACL_MEMCPY_HOST_TO_DEVICE);
            }
        }

        void aclMat::upload(const Mat &m, aclStream stream, ALIGNMENT config)
        {
            CV_Assert((config == ALIGNMENT::MEMORY_ALIGN) || (config == ALIGNMENT::MEMORY_UNALIGNED));
            if (config == ALIGNMENT::MEMORY_UNALIGNED)
            {
                CV_Assert(m.data && (this->step == m.step) && (this->rows == m.rows) && (this->cols == m.cols) && (this->type() == m.type()));
                aclrtMemcpyAsync((void *)this->data, this->totalSize, (void *)m.data, (m.step * m.rows), ACL_MEMCPY_HOST_TO_DEVICE, stream);
            }
            else if (config == ALIGNMENT::MEMORY_ALIGN)
            {
                CV_Assert(m.data && (this->rows == m.rows) && (this->cols == m.cols) && (this->type() == m.type()));
                aclrtMemcpy2dAsync((void *)this->data, this->step, (void *)m.data, m.step, m.cols * m.elemSize(), m.rows, ACL_MEMCPY_HOST_TO_DEVICE, stream);
            }
	        AclSafeCall(aclrtSynchronizeStream(stream));
        }


        void aclMat::download(Mat &m, ALIGNMENT config) const
        {
            CV_Assert((config == ALIGNMENT::MEMORY_ALIGN) || (config == ALIGNMENT::MEMORY_UNALIGNED));
            if (config == ALIGNMENT::MEMORY_UNALIGNED)
            {
                CV_Assert(m.data && (this->step == m.step) && (this->rows == m.rows) && (this->cols == m.cols) && (this->type() == m.type()));
                aclrtMemcpy((void *)m.data, (m.step * m.rows), (void *)(this->data), (m.step * m.rows), ACL_MEMCPY_DEVICE_TO_HOST);
            }
            else if (config == ALIGNMENT::MEMORY_ALIGN)
            {
                CV_Assert(m.data && (this->rows == m.rows) && (this->cols == m.cols) && (this->type() == m.type()));
                aclrtMemcpy2d((void *)m.data, m.step, (void *)(this->data), this->step, this->cols * this->elemSize(), this->rows, ACL_MEMCPY_DEVICE_TO_HOST);
            }
            return;
        }

        void aclMat::download(Mat &m, aclStream stream, ALIGNMENT config) const
        {
            CV_Assert((config == ALIGNMENT::MEMORY_ALIGN) || (config == ALIGNMENT::MEMORY_UNALIGNED));
            if (config == ALIGNMENT::MEMORY_UNALIGNED)
            {
                CV_Assert(m.data && (this->step == m.step) && (this->rows == m.rows) && (this->cols == m.cols) && (this->type() == m.type()));
                aclrtMemcpyAsync((void *)m.data, (m.step * m.rows), (void *)(this->data), this->totalSize, ACL_MEMCPY_DEVICE_TO_HOST, stream);
            }
            else if (config == ALIGNMENT::MEMORY_ALIGN)
            {
                CV_Assert(m.data && (this->rows == m.rows) && (this->cols == m.cols) && (this->type() == m.type()));
                aclrtMemcpy2dAsync((void *)m.data, m.step, (void *)(this->data), this->step, this->cols * this->elemSize(), this->rows, ACL_MEMCPY_DEVICE_TO_HOST, stream);
            }
	        AclSafeCall(aclrtSynchronizeStream(stream));
            return;
        }

        void aclMat::create(int _rows, int _cols, int _type, ALIGNMENT config, aclrtMemMallocPolicy policy)
        {
            createEx(_rows, _cols, _type, config, policy);
        }

        void aclMat::create(Size size, int type, ALIGNMENT config, aclrtMemMallocPolicy policy)
        {
            createEx(size, type, config, policy);
        }

        inline size_t alignSize(size_t sz, int n)
        {
            return (sz + n - 1) & -n;
        }

        /* core logic */
        void aclMat::createEx(int _rows, int _cols, int _type, ALIGNMENT config, aclrtMemMallocPolicy policy)
        {
            /* TO ENSURE */
            //_type &= CV_MAT_TYPE_MASK;
            _type &= TYPE_MASK;
            if (rows == _rows && cols == _cols && type() == _type && data)
                return;

            if (data)
                release();
            
            CV_DbgAssert(_rows >= 0 && _cols >= 0);

            if (_rows > 0 && _cols > 0)
            {
                /* TO ENSURE */
                //flags = (_type & CV_MAT_TYPE_MASK) | MAGIC_VAL;
                flags = Mat::MAGIC_VAL + _type;
                rows = _rows;
                cols = _cols;
                wholerows = _rows;
                wholecols = _cols;
                size_t esz = elemSize();

                void *dev_ptr;
                if (config == ALIGNMENT::MEMORY_ALIGN)
                    step = alignSize(cols * esz);
                else 
                    step = cols * esz;
                totalSize = step * rows;

                AclSafeCall(aclrtMalloc(&dev_ptr, totalSize, policy));

                data = dev_ptr;
                datastart = static_cast<uchar *>(data);
                dataend = static_cast<uchar *>(data) + totalSize;
                refcount = static_cast<int *>(fastMalloc(sizeof(*refcount)));
                *refcount = 0;
                CV_XADD(refcount, 1);
                flags |= Mat::CONTINUOUS_FLAG;
            }
        }

        void aclMat::createEx(Size size, int type, ALIGNMENT config, aclrtMemMallocPolicy policy)
        {
            createEx(size.height, size.width, type, config, policy);
        }

        void aclMat::release()
        {
            CV_XADD(refcount, -1);
            if (data && (*refcount == 0))
            {
                aclrtFree(data);
            }
            data = nullptr;
            datastart = nullptr;
            dataend = nullptr;
        }

        aclMat &aclMat::operator+=(const aclMat &m)
        {
            TwoInAndOneOut(*this, m, *this, "Add"); 
            return *this;
        }

        aclMat &aclMat::operator-=(const aclMat &m)
        {
            TwoInAndOneOut(*this, m, *this, "Sub"); 
            return *this;
        }

        aclMat &aclMat::operator/=(const aclMat &m)
        {
            TwoInAndOneOut(*this, m, *this, "Div"); 
            return *this;
        }

        aclMat &aclMat::operator*=(const aclMat &m)
        {
            vector<aclMat> input_Mat;
            vector<aclMat> output_Mat;
            vector<aclDataBuffer *> inputBuffers_;
            vector<aclDataBuffer *> outputBuffers_;
            aclMat newMat{this->rows, m.cols, this->type(), this->acl_context};

            input_Mat.emplace_back(*this);
            input_Mat.emplace_back(m);
            output_Mat.emplace_back(newMat);

            inputBuffers_.emplace_back(aclCreateDataBuffer(this->data, this->totalSize));
            inputBuffers_.emplace_back(aclCreateDataBuffer(m.data, m.totalSize));
            inputBuffers_.emplace_back(aclCreateDataBuffer(nullptr, 0));
            outputBuffers_.emplace_back(aclCreateDataBuffer(newMat.data, newMat.totalSize));

            OperatorDesc opDesc = CreateOpDesc("MatMul", input_Mat, output_Mat);
            opDesc.AddInputTensorDesc(ACL_DT_UNDEFINED, 0, nullptr, ACL_FORMAT_UNDEFINED);
            opDesc.AddTensorAttr("transpose_x1", OP_BOOL, false);
            opDesc.AddTensorAttr("transpose_x2", OP_BOOL, false);
            compileAndRunop(opDesc, inputBuffers_, outputBuffers_, this->acl_context);

            newMat.data = aclGetDataBufferAddr(outputBuffers_[0]);
            *this = newMat;

            for (size_t i = 0; i < inputBuffers_.size(); i++)
                AclSafeCall(aclDestroyDataBuffer(inputBuffers_[i]));
            for (size_t i = 0; i < outputBuffers_.size(); i++)
                AclSafeCall(aclDestroyDataBuffer(outputBuffers_[i]));

            return *this;
        }

        aclMat& aclMat::abs() 
        {
            OneInAndOneOut(*this, *this, "Abs");
            return *this;    
        }
    }
}





/*
///////////////////////////////////////////////////////////////////////////
////////////////////////////////// CopyTo /////////////////////////////////
///////////////////////////////////////////////////////////////////////////
static void copy_to_with_mask(const aclMat &src, aclMat &dst, const aclMat &mask, string kernelName)
{
}

void cv::acl::aclMat::copyTo( aclMat &mat, const aclMat &mask) const
{
}

///////////////////////////////////////////////////////////////////////////
//////////////////////////////// ConvertTo ////////////////////////////////
///////////////////////////////////////////////////////////////////////////

void cv::acl::aclMat::convertTo( aclMat &dst, int rtype, double alpha, double beta ) const
{
}

///////////////////////////////////////////////////////////////////////////
//////////////////////////////// setTo ////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

aclMat &cv::acl::aclMat::operator = (const Scalar &s)
{
    setTo(s);
    return *this;
}


static void set_to_withoutmask_run(const aclMat &dst, const Scalar &scalar, string kernelName)
{
}

static void set_to_withmask_run(const aclMat &dst, const Scalar &scalar, const aclMat &mask, string kernelName)
{
}

aclMat &cv::acl::aclMat::setTo(const Scalar &scalar, const aclMat &mask)
{
}

aclMat cv::acl::aclMat::reshape(int new_cn, int new_rows) const
{
    if( new_rows != 0 && new_rows != rows)
        CV_Error( CV_StsBadFunc, "aclMat's number of rows can not be changed for current version" );

    aclMat hdr = *this;

    int cn = aclchannels();
    if (new_cn == 0)
        new_cn = cn;

    int total_width = cols * cn;
    if ((new_cn > total_width || total_width % new_cn != 0) && new_rows == 0)
        new_rows = rows * total_width / new_cn;

    if (new_rows != 0 && new_rows != rows)
    {
        int total_size = total_width * rows;

        if (!isContinuous())
            CV_Error(CV_BadStep, "The matrix is not continuous, thus its number of rows can not be changed");

        if ((unsigned)new_rows > (unsigned)total_size)
            CV_Error(CV_StsOutOfRange, "Bad new number of rows");

        total_width = total_size / new_rows;
        if (total_width * new_rows != total_size)
            CV_Error(CV_StsBadArg, "The total number of matrix elements is not divisible by the new number of rows");

        hdr.rows = new_rows;
        hdr.step = total_width * elemSize1();
    }

    int new_width = total_width / new_cn;
    if (new_width * new_cn != total_width)
        CV_Error(CV_BadNumChannels, "The total width is not divisible by the new number of channels");

    hdr.cols = new_width;
    hdr.wholecols = new_width;
    hdr.flags = (hdr.flags & ~CV_MAT_CN_MASK) | ((new_cn - 1) << CV_CN_SHIFT);
    return hdr;

}


*/

#if 0



/*
aclMat& cv::acl::aclMat::operator+=( const aclMat& m )
{
    add(*this, m, *this);
    return *this;
}

aclMat& cv::acl::aclMat::operator-=( const aclMat& m )
{
    subtract(*this, m, *this);
    return *this;
}

aclMat& cv::acl::aclMat::operator*=( const aclMat& m )
{
    multiply(*this, m, *this);
    return *this;
}

aclMat& cv::acl::aclMat::operator/=( const aclMat& m )
{
    divide(*this, m, *this);
    return *this;
}
*/
#endif