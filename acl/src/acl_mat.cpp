#include "precomp.hpp"


#define ALIGN 64

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

        void aclMat::create(int _rows, int _cols, int _type, ALIGNMENT config, MemMallocPolicy policy)
        {
            createEx(_rows, _cols, _type, config, policy);
        }

        void aclMat::create(Size size, int type, ALIGNMENT config, MemMallocPolicy policy)
        {
            createEx(size, type, config, policy);
        }

        inline size_t alignSize(size_t sz, int n)
        {
            return (((sz) + n - 1) / n ) * n;
        }

        /* core logic */
        void aclMat::createEx(int _rows, int _cols, int _type, ALIGNMENT config, MemMallocPolicy policy)
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
                {
                    if (channels() == 3)
                        step = alignSize(cols * esz, ALIGN * channels());
                    else   
                        step = alignSize(cols * esz);
                }
                else 
                    step = cols * esz;
                totalSize = step * rows;

                AclSafeCall(aclrtMalloc(&dev_ptr, totalSize, type_transition(policy)));

                data = dev_ptr;
                datastart = static_cast<uchar *>(data);
                dataend = static_cast<uchar *>(data) + totalSize;
                refcount = static_cast<int *>(fastMalloc(sizeof(*refcount)));
                *refcount = 0;
                CV_XADD(refcount, 1);
                flags |= Mat::CONTINUOUS_FLAG;
            }
        }

        void aclMat::createEx(Size size, int type, ALIGNMENT config, MemMallocPolicy policy)
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
            CV_Assert(this->rows == m.rows && this->cols == m.cols && this->type() == m.type());
            TwoInAndOneOut(*this, m, *this, "Add"); 
            return *this;
        }

        aclMat &aclMat::operator-=(const aclMat &m)
        {
            CV_Assert(this->rows == m.rows && this->cols == m.cols && this->type() == m.type());
            TwoInAndOneOut(*this, m, *this, "Sub"); 
            return *this;
        }

        aclMat &aclMat::operator/=(const aclMat &m)
        {
            CV_Assert(this->rows == m.rows && this->cols == m.cols && this->type() == m.type());
            TwoInAndOneOut(*this, m, *this, "Div"); 
            return *this;
        }

        aclMat &aclMat::operator*=(const aclMat &m)
        {
            CV_Assert(this->cols == m.rows && this->type() == m.type());
            vector<aclMat> input_Mat;
            vector<aclMat> output_Mat;
            vector<aclDataBuffer *> inputBuffers_;
            vector<aclDataBuffer *> outputBuffers_;
            aclMat newMat{this->rows, m.cols, this->type(), this->acl_context};

            input_Mat.emplace_back(*this);
            input_Mat.emplace_back(m);
            output_Mat.emplace_back(newMat);

            OperatorDesc opDesc = CreateOpDesc("MatMul", input_Mat, output_Mat, ACL_FORMAT_NHWC, TWO_DIMS);
            opDesc.AddInputTensorDesc(ACL_DT_UNDEFINED, 0, nullptr, ACL_FORMAT_UNDEFINED);
            opDesc.AddTensorAttr("transpose_x1", OP_BOOL, false);
            opDesc.AddTensorAttr("transpose_x2", OP_BOOL, false);

            inputBuffers_.emplace_back(aclCreateDataBuffer(this->data, this->totalSize));
            inputBuffers_.emplace_back(aclCreateDataBuffer(m.data, m.totalSize));
            inputBuffers_.emplace_back(aclCreateDataBuffer(nullptr, 0));
            outputBuffers_.emplace_back(aclCreateDataBuffer(newMat.data, newMat.totalSize));

            compileAndRunop(opDesc, inputBuffers_, outputBuffers_, this->acl_context, 0);

            *this = newMat;

            for (size_t i = 0; i < inputBuffers_.size(); i++)
                AclSafeCall(aclDestroyDataBuffer(inputBuffers_[i]));
            for (size_t i = 0; i < outputBuffers_.size(); i++)
                AclSafeCall(aclDestroyDataBuffer(outputBuffers_[i]));

            return *this;
        }

    } /* end of namespace acl */

} /* end of namespace cv */
