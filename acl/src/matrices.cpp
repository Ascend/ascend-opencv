#include "precomp.hpp"

namespace cv
{
    namespace acl
    {
/*
        //disable
        void lookUpTable(const aclMat& src, const aclMat& lut, aclMat& dest)
        {
            bool is_correct;
            is_correct = ((src.depth() == CV_8U) || (src.depth() == CV_8S));
            is_correct &= ((lut.depth() == CV_8U) || (lut.depth() == CV_8S));
            is_correct &= (lut.totalSize == 256);
            CV_Assert(is_correct);

            vector<aclMat> input_Mat;
            vector<aclMat> output_Mat;
            vector<aclDataBuffer *> inputBuffers_;
            vector<aclDataBuffer *> outputBuffers_;

            uchar keyValue[256];
            for (int i = 0; i < 256; ++i)
                keyValue[i] = i;
            aclMat key(1, 256, src.type(), keyValue, src.acl_context);

            input_Mat.emplace_back(src);
            input_Mat.emplace_back(key);
            input_Mat.emplace_back(lut);

            inputBuffers_.emplace_back(aclCreateDataBuffer(src.data, src.totalSize));
            inputBuffers_.emplace_back(aclCreateDataBuffer(key.data, key.totalSize));
            inputBuffers_.emplace_back(aclCreateDataBuffer(lut.data, lut.totalSize));

            aclDataType dataType = type_transition(input_Mat[0].depth());
            aclFormat format = ACL_FORMAT_NHWC;

            OperatorDesc opDesc("LookupTableImport");
            vector<int64_t> shape1{src.rows, src.cols * src.channels()};
            vector<int64_t> shape2{lut.rows, lut.cols * lut.channels()};
            vector<int64_t> shape3{dest.rows, dest.cols * dest.channels()};
            opDesc.AddInputTensorDesc(dataType, shape1.size(), shape1.data(), format);
            opDesc.AddInputTensorDesc(dataType, shape2.size(), shape2.data(), format);
            opDesc.AddInputTensorDesc(dataType, shape3.size(), shape3.data(), format);

            compileAndRunop(opDesc, inputBuffers_, outputBuffers_, dest.acl_context);

            dest.data = aclGetDataBufferAddr(inputBuffers_[0]);

            for (size_t i = 0; i < inputBuffers_.size(); i++)
                AclSafeCall(aclDestroyDataBuffer(inputBuffers_[i]));
            for (size_t i = 0; i < outputBuffers_.size(); i++)
                AclSafeCall(aclDestroyDataBuffer(outputBuffers_[i]));
        }
*/

/*
        void merge(const vector<aclMat>& mv, aclMat& dest)
        {
            vector<aclDataBuffer *> inputBuffers_;
            vector<aclDataBuffer *> outputBuffers_;

            OperatorDesc opDesc("ConcatD");
            aclDataType dataType = type_transition(mv[0].depth());

            for (size_t i = 0; i < mv.size(); ++i)
            {
                int cols = mv[i].step/mv[i].elemSize();
                vector<int64_t> inputShape{1, mv[i].rows, cols, mv[i].channels()};
                opDesc.AddInputTensorDesc(dataType, inputShape.size(), inputShape.data(), ACL_FORMAT_ND);
            }
            int cols = dest.step/dest.elemSize();
            vector<int64_t> outputShape{1, dest.rows, cols, dest.channels()};
            opDesc.AddOutputTensorDesc(dataType, outputShape.size(), outputShape.data(), ACL_FORMAT_ND);

            for (size_t i = 0; i < opDesc.inputDesc.size(); ++i)
            {
                inputBuffers_.emplace_back(aclCreateDataBuffer(mv[i].data, mv[i].totalSize));
            }
            outputBuffers_.emplace_back(aclCreateDataBuffer(dest.data, dest.totalSize));

            aclopSetAttrInt(opDesc.opAttr, "N", mv.size());
            aclopSetAttrInt(opDesc.opAttr, "concat_dim", 3);

            compileAndRunop(opDesc, inputBuffers_, outputBuffers_, dest.acl_context);

            for (size_t i = 0; i < inputBuffers_.size(); i++)
                AclSafeCall(aclDestroyDataBuffer(inputBuffers_[i]));
            for (size_t i = 0; i < outputBuffers_.size(); i++)
                AclSafeCall(aclDestroyDataBuffer(outputBuffers_[i]));
        }
*/


        void merge(const vector<aclMat>& mv, aclMat& dest)
        {
            vector<aclDataBuffer *> inputBuffers_;
            vector<aclDataBuffer *> outputBuffers_;

            OperatorDesc opDesc("Concat");
            aclDataType dataType = type_transition(mv[0].depth());

            vector<int64_t> inputShape{};
            opDesc.AddInputTensorDesc(ACL_INT32, inputShape.size(), inputShape.data(), ACL_FORMAT_ND);

            for (size_t i = 0; i < mv.size(); ++i)
            {
                int cols = mv[i].step/mv[i].elemSize();
                vector<int64_t> inputShape{1, mv[i].rows, cols, mv[i].channels()};
                opDesc.AddInputTensorDesc(dataType, inputShape.size(), inputShape.data(), ACL_FORMAT_NHWC);
            }

            int cols = dest.step/dest.elemSize();
            vector<int64_t> outputShape{1, dest.rows, cols, dest.channels()};
            opDesc.AddOutputTensorDesc(dataType, outputShape.size(), outputShape.data(), ACL_FORMAT_NHWC);

            ino64_t N = mv.size();
            aclopSetAttrInt(opDesc.opAttr, "N", N);

            aclSetTensorDescName(opDesc.inputDesc[0], "concat_dim");
            aclSetTensorDescName(opDesc.inputDesc[1], "x0");
            aclSetTensorDescName(opDesc.inputDesc[2], "x1");
            aclSetTensorDescName(opDesc.inputDesc[3], "x2");
            aclSetTensorDescName(opDesc.outputDesc[0], "y");

            void *dev;
            int64_t concat_dim = 3;
            size_t size = aclGetTensorDescSize(opDesc.inputDesc[0]);
            aclrtMalloc(&dev, size, ACL_MEM_MALLOC_HUGE_FIRST);
            aclrtMemcpy(dev, size, &concat_dim, size, ACL_MEMCPY_HOST_TO_DEVICE);
            inputBuffers_.emplace_back(aclCreateDataBuffer(dev, size));

            for (size_t i = 0; i < mv.size(); ++i)
                inputBuffers_.emplace_back(aclCreateDataBuffer(mv[i].data, mv[i].totalSize));

            outputBuffers_.emplace_back(aclCreateDataBuffer(dest.data, dest.totalSize));

            compileAndRunop(opDesc, inputBuffers_, outputBuffers_, dest.acl_context);

            for (size_t i = 0; i < inputBuffers_.size(); i++)
                AclSafeCall(aclDestroyDataBuffer(inputBuffers_[i]));
            for (size_t i = 0; i < outputBuffers_.size(); i++)
                AclSafeCall(aclDestroyDataBuffer(outputBuffers_[i]));

            aclrtFree(dev);
        }
 
        

/**
 * @brief : Dynamic shape reasoning, compiler problems
 * 
 */

        void transpose(const aclMat& src, aclMat& dest)
        {
            vector<aclDataBuffer *> inputBuffers_;
            vector<aclDataBuffer *> outputBuffers_;
            vector<aclDataBuffer *> inputBuffers_host;

            OperatorDesc opDesc("Transpose");
            aclDataType dataType = type_transition(src.depth());

            vector<int64_t> inputShape1{1, src.rows, src.cols, src.channels()};
            opDesc.AddInputTensorDesc(dataType, inputShape1.size(), inputShape1.data(), ACL_FORMAT_ND);

            vector<int64_t> inputShape2{4};
            opDesc.AddInputTensorDesc(ACL_INT32, inputShape2.size(), inputShape2.data(), ACL_FORMAT_ND);

            vector<int64_t> outputShape{-1, -1, -1, -1};
            opDesc.AddOutputTensorDesc(dataType, outputShape.size(), outputShape.data(), ACL_FORMAT_ND);

            inputBuffers_.emplace_back(aclCreateDataBuffer(src.data, src.totalSize));

            void *dev;
            void *perm;

            size_t size = aclGetTensorDescSize(opDesc.inputDesc[1]);
            aclrtMalloc(&dev, size, ACL_MEM_MALLOC_HUGE_FIRST);
            aclrtMallocHost(&perm, aclGetTensorDescSize(opDesc.inputDesc.data()[1]));
            ((int *)perm)[0] = 0;
            ((int *)perm)[1] = 2;
            ((int *)perm)[2] = 1;
            ((int *)perm)[3] = 3;
            aclrtMemcpy(dev, size, perm, size, ACL_MEMCPY_HOST_TO_DEVICE);
            inputBuffers_.emplace_back(aclCreateDataBuffer(dev, size));

            AclSafeCall(aclopCompile(opDesc.opType.c_str(),
                            opDesc.inputDesc.size(),
                            opDesc.inputDesc.data(),
                            opDesc.outputDesc.size(),
                            opDesc.outputDesc.data(),
                            opDesc.opAttr,
                            ACL_ENGINE_SYS,
                            ACL_COMPILE_SYS,
                            nullptr));

            void *host_data;
            size_t host_size = src.totalSize;
            aclrtMallocHost(&host_data, host_size);
            aclrtMemcpy(host_data, host_size, src.data, host_size, ACL_MEMCPY_DEVICE_TO_HOST);
            inputBuffers_host.emplace_back(aclCreateDataBuffer(host_data, host_size));
            inputBuffers_host.emplace_back(aclCreateDataBuffer(perm, size));

            AclSafeCall(aclopInferShape("Transpose", opDesc.inputDesc.size(), opDesc.inputDesc.data(), \
                    inputBuffers_host.data(), opDesc.outputDesc.size(), opDesc.outputDesc.data(), opDesc.opAttr));
            outputBuffers_.emplace_back(aclCreateDataBuffer(dest.data, dest.totalSize));
                
            AclSafeCall(aclopExecuteV2(opDesc.opType.c_str(),
                            inputBuffers_.size(),
                            opDesc.inputDesc.data(),
                            inputBuffers_.data(),
                            outputBuffers_.size(),
                            opDesc.outputDesc.data(),
                            outputBuffers_.data(),
                            opDesc.opAttr,
                            src.acl_context->get_stream(0)));

            AclSafeCall(aclrtSynchronizeStream(src.acl_context->get_stream(0)));
            
            AclSafeCall(aclDestroyDataBuffer(inputBuffers_[0]));
            AclSafeCall(aclDestroyDataBuffer(inputBuffers_[1]));
            AclSafeCall(aclDestroyDataBuffer(inputBuffers_host[0]));
            AclSafeCall(aclDestroyDataBuffer(inputBuffers_host[1]));
            AclSafeCall(aclDestroyDataBuffer(outputBuffers_[0]));
            aclrtFreeHost(perm);
            aclrtFreeHost(host_data);
        }



/*
        void transpose(const aclMat& src, aclMat& dest)
        {
            vector<aclDataBuffer *> inputBuffers_;
            vector<aclDataBuffer *> outputBuffers_;

            OperatorDesc opDesc("TransposeD");
            aclDataType dataType = type_transition(src.depth());

            vector<int64_t> inputShape1{1, src.rows, src.cols, src.channels()};
            opDesc.AddInputTensorDesc(dataType, inputShape1.size(), inputShape1.data(), ACL_FORMAT_NHWC);

            vector<int64_t> outputShape{1, src.cols, src.rows, src.channels()};
            opDesc.AddOutputTensorDesc(dataType, outputShape.size(), outputShape.data(), ACL_FORMAT_NHWC);

            vector<int64_t> permlist = {0, 2, 1, 3};
            aclopSetAttrListInt(opDesc.opAttr, "perm", permlist.size(), permlist.data());

            inputBuffers_.emplace_back(aclCreateDataBuffer(src.data, src.totalSize));
            outputBuffers_.emplace_back(aclCreateDataBuffer(dest.data, dest.totalSize));

            compileAndRunop(opDesc, inputBuffers_, outputBuffers_, src.acl_context);

            AclSafeCall(aclDestroyDataBuffer(inputBuffers_[0]));
            AclSafeCall(aclDestroyDataBuffer(outputBuffers_[0]));
        }
*/


        void split(const aclMat& src, vector<aclMat>& mv)
        {
            vector<aclDataBuffer *> inputBuffers_;
            vector<aclDataBuffer *> outputBuffers_;
            int split_dim = 3;
            int num_split = src.channels();

            OperatorDesc opDesc("SplitD");
            aclDataType dataType = type_transition(src.depth());

            int cols = src.step/src.elemSize();
            vector<int64_t> inputShape1{1, src.rows, cols, src.channels()};
            opDesc.AddInputTensorDesc(dataType, inputShape1.size(), inputShape1.data(), ACL_FORMAT_ND);

            for (int i = 0; i < num_split; ++i)
            {
                int cols = mv[i].step/mv[i].elemSize();
                vector<int64_t> outputShape{1, mv[i].rows, cols, mv[i].channels()};
                opDesc.AddOutputTensorDesc(dataType, outputShape.size(), outputShape.data(), ACL_FORMAT_ND);
            }
            
            auto opAttr = opDesc.opAttr;
            aclopSetAttrInt(opAttr, "split_dim", split_dim);
            aclopSetAttrInt(opAttr, "num_split", num_split);

            inputBuffers_.emplace_back(aclCreateDataBuffer(src.data, src.totalSize));

            for (int i = 0; i < num_split; ++i)
                outputBuffers_.emplace_back(aclCreateDataBuffer(mv[i].data, mv[i].totalSize));

            compileAndRunop(opDesc, inputBuffers_, outputBuffers_, src.acl_context);
                
            AclSafeCall(aclDestroyDataBuffer(inputBuffers_[0]));
            for (int i = 0; i < num_split; ++i)
                AclSafeCall(aclDestroyDataBuffer(outputBuffers_[i]));
        }


/*
        //disable

        void split(const aclMat& src, vector<aclMat>& mv)
        {
            vector<aclDataBuffer *> inputBuffers_;
            vector<aclDataBuffer *> inputBuffers_host;
            vector<aclDataBuffer *> outputBuffers_;
            int num_split = src.channels();

            OperatorDesc opDesc("Split");
            aclDataType dataType = type_transition(src.depth());

            vector<int64_t> inputShape{};
            opDesc.AddInputTensorDesc(ACL_INT32, inputShape.size(), inputShape.data(), ACL_FORMAT_ND);

            int cols = src.step/src.elemSize();
            vector<int64_t> inputShape1{1, src.rows, cols, src.channels()};
            opDesc.AddInputTensorDesc(dataType, inputShape1.size(), inputShape1.data(), ACL_FORMAT_ND);
            
            for (int i = 0; i < num_split; ++i)
            {
                vector<int64_t> outputShape{-1, -1, -1, -1};
                opDesc.AddOutputTensorDesc(dataType, outputShape.size(), outputShape.data(), ACL_FORMAT_ND);
            }
            
            aclSetTensorDescName(opDesc.inputDesc[0], "split_dim");
            aclSetTensorDescName(opDesc.inputDesc[1], "x");
            aclSetTensorDescName(opDesc.outputDesc[0], "y0");
            aclSetTensorDescName(opDesc.outputDesc[1], "y1");
            aclSetTensorDescName(opDesc.outputDesc[2], "y2");

            aclopSetAttrInt(opDesc.opAttr, "num_split", num_split);

            AclSafeCall(aclopCompile(opDesc.opType.c_str(),
                            opDesc.inputDesc.size(),
                            opDesc.inputDesc.data(),
                            opDesc.outputDesc.size(),
                            opDesc.outputDesc.data(),
                            opDesc.opAttr,
                            ACL_ENGINE_SYS,
                            ACL_COMPILE_SYS,
                            nullptr));

            void *dev;
            int split_dim = 3;
            size_t size = aclGetTensorDescSize(opDesc.inputDesc[0]);
            aclrtMalloc(&dev, size, ACL_MEM_MALLOC_HUGE_FIRST);
            aclrtMemcpy(dev, size, &split_dim, size, ACL_MEMCPY_HOST_TO_DEVICE);

            inputBuffers_host.emplace_back(aclCreateDataBuffer(&split_dim, size));

            void *host_data;
            size_t host_size = src.totalSize;
            aclrtMallocHost(&host_data, host_size);
            aclrtMemcpy(host_data, host_size, src.data, host_size, ACL_MEMCPY_DEVICE_TO_HOST);
            inputBuffers_host.emplace_back(aclCreateDataBuffer(host_data, host_size));

            AclSafeCall(aclopInferShape("Split", opDesc.inputDesc.size(), opDesc.inputDesc.data(), \
                    inputBuffers_host.data(), opDesc.outputDesc.size(), opDesc.outputDesc.data(), opDesc.opAttr));

            inputBuffers_.emplace_back(aclCreateDataBuffer(dev, size));
            inputBuffers_.emplace_back(aclCreateDataBuffer(src.data, src.totalSize));
                
            for (int i = 0; i < num_split; ++i)
                outputBuffers_.emplace_back(aclCreateDataBuffer(mv[i].data, mv[i].totalSize));

            AclSafeCall(aclopExecuteV2(opDesc.opType.c_str(),
                            inputBuffers_.size(),
                            opDesc.inputDesc.data(),
                            inputBuffers_.data(),
                            outputBuffers_.size(),
                            opDesc.outputDesc.data(),
                            outputBuffers_.data(),
                            opDesc.opAttr,
                            src.acl_context->get_stream(0)));

            AclSafeCall(aclrtSynchronizeStream(src.acl_context->get_stream(0)));

            AclSafeCall(aclDestroyDataBuffer(inputBuffers_[0]));
            AclSafeCall(aclDestroyDataBuffer(inputBuffers_[1]));
            AclSafeCall(aclDestroyDataBuffer(inputBuffers_host[0]));
            AclSafeCall(aclDestroyDataBuffer(inputBuffers_host[1]));
            for (int i = 0; i < num_split; ++i)
                AclSafeCall(aclDestroyDataBuffer(outputBuffers_[i]));
        }
*/

        static void flip_(const aclMat& src, aclMat& dest, int axis)
        {
            vector<aclDataBuffer *> inputBuffers_;
            vector<aclDataBuffer *> outputBuffers_;

            OperatorDesc opDesc("ReverseV2");
            aclDataType dataType = type_transition(src.depth());

            vector<int64_t> inputShape1{1, src.rows, src.cols, src.channels()};
            opDesc.AddInputTensorDesc(dataType, inputShape1.size(), inputShape1.data(), ACL_FORMAT_ND);

            vector<int64_t> inputShape2{1};
            opDesc.AddInputTensorDesc(ACL_INT32, inputShape2.size(), inputShape2.data(), ACL_FORMAT_ND);

            vector<int64_t> outputShape{1, dest.rows, dest.cols, dest.channels()};
            opDesc.AddOutputTensorDesc(dataType, outputShape.size(), outputShape.data(), ACL_FORMAT_ND);
            
            inputBuffers_.emplace_back(aclCreateDataBuffer(src.data, src.totalSize));

            void *dev;
            size_t size = aclGetTensorDescSize(opDesc.inputDesc[1]);
            aclrtMalloc(&dev, size, ACL_MEM_MALLOC_HUGE_FIRST);
            aclrtMemcpy(dev, size, &axis, size, ACL_MEMCPY_HOST_TO_DEVICE);
            inputBuffers_.emplace_back(aclCreateDataBuffer(dev, size));

            outputBuffers_.emplace_back(aclCreateDataBuffer(dest.data, dest.totalSize));

            compileAndRunop(opDesc, inputBuffers_, outputBuffers_, src.acl_context);
                
            AclSafeCall(aclDestroyDataBuffer(inputBuffers_[0]));
            AclSafeCall(aclDestroyDataBuffer(inputBuffers_[1]));
            AclSafeCall(aclDestroyDataBuffer(outputBuffers_[0]));
        }

        void flip(const aclMat& src, aclMat& dest, int filpCode)
        {
            if (filpCode == 0) {
                flip_(src, dest, 1);
            }
            else if (filpCode > 0) {
                flip_(src, dest, 2);
            }
            else {
                flip_(src, dest, 2);
                aclMat tmp(dest.rows, dest.cols, dest.type(), dest.acl_context);
                aclrtMemcpy(tmp.data, dest.totalSize, dest.data, dest.totalSize, ACL_MEMCPY_DEVICE_TO_DEVICE);
                flip_(tmp, dest, 1);
            }
        }
    } /* end of namespace acl */

} /* end of namespace cv */