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

namespace cv
{

    namespace acl
    {
        /////////////////////////////////////////////////aclEnv////////////////////////
        Mutex &getInitMutex();
        static Mutex *__initmutex = NULL;
        Mutex &getInitMutex()
        {
            if (__initmutex == NULL)
                __initmutex = new Mutex();
            return *__initmutex;
        }

        aclEnv *global_aclenv = nullptr;
        aclEnv* aclEnv::get_acl_env(const char* config_path)
        {
            if (nullptr == global_aclenv)
            {
                AutoLock lock(getInitMutex());
                if (nullptr == global_aclenv)
                    global_aclenv = new aclEnv(config_path);
            }
            return global_aclenv;
        }

        /////////////////////////////////////////////////aclCxt////////////////////////

        inline void aclCxt::create_stream(int count)
        {
            CV_Assert(count > 0);

            int i;
            for(i = 0; i <count; i++)
            {
                aclStream stream;
                AclSafeCall(aclrtCreateStream(&stream));

                _acl_streams.push_back(stream);
            }

            clog << "aclrtCreateStream() is success" << endl;
        }


        /////////////////////////////////////////////////create acl context////////////////////////
        aclCxt *set_device(const char* config_path, int device_id, int stream_count)
        {
            /* ??? */
            aclEnv *acl_env = aclEnv::get_acl_env(config_path);
            if (global_aclenv->refcount) {
                AutoLock lock(getInitMutex());
                CV_XADD(global_aclenv->refcount, 1);
            }
            int device_count = acl_env->get_device_count();
            CV_Assert(device_id < device_count);

            aclCxt *acl_context = new aclCxt(device_id);
            acl_context->set_current_context();
            acl_context->create_stream(stream_count);

            clog << "set_device() is success" << endl;
            return acl_context;
        }

        void release_device(aclCxt* context)
        {
            CV_Assert(context);
            delete context;
            if (global_aclenv->refcount)
            {
                AutoLock lock(getInitMutex());
                CV_XADD(global_aclenv->refcount, -1);
                if (*(global_aclenv->refcount) == 0)
                    delete global_aclenv;
            }
            clog << "release_device() is success" << endl;
        }

        // define operator
        OperatorDesc CreateOpDesc(const string opType, vector<aclMat>& input_Mat, vector<aclMat>& output_Mat, aclFormat format)
        {
            size_t i;

            aclDataType dataType = type_transition(input_Mat[0].type());
            OperatorDesc opDesc(opType);

            for (i = 0; i < input_Mat.size(); ++i) {
                vector<int64_t> shape{input_Mat[i].rows, input_Mat[i].cols * input_Mat[i].channels()};
                opDesc.AddInputTensorDesc(dataType, shape.size(), shape.data(), format);
            }

            for (i = 0; i < output_Mat.size(); ++i) {
                vector<int64_t> shape{output_Mat[i].rows, output_Mat[i].cols * output_Mat[i].channels()};
                opDesc.AddOutputTensorDesc(dataType, shape.size(), shape.data(), format);
            }

            return opDesc;
        }

        void compileAndRunop(OperatorDesc& opDesc, vector<aclDataBuffer *>& inputBuffers_, vector<aclDataBuffer *>& outputBuffers_, aclCxt *acl_context)
        {
/*
            AclSafeCall(aclopCompileAndExecute(opDesc.opType.c_str(),
                            opDesc.inputDesc.size(),
                            opDesc.inputDesc.data(),
                            inputBuffers_.data(),
                            opDesc.outputDesc.size(),
                            opDesc.outputDesc.data(),
                            outputBuffers_.data(),
                            opDesc.opAttr,
                            ACL_ENGINE_SYS,
                            ACL_COMPILE_SYS,
                            nullptr,
                            acl_context->get_stream(0)));
*/

            AclSafeCall(aclopCompile(opDesc.opType.c_str(),
                            opDesc.inputDesc.size(),
                            opDesc.inputDesc.data(),
                            opDesc.outputDesc.size(),
                            opDesc.outputDesc.data(),
                            opDesc.opAttr,
                            ACL_ENGINE_SYS,
                            ACL_COMPILE_SYS,
                            nullptr));

            AclSafeCall(aclopExecuteV2(opDesc.opType.c_str(),
                            inputBuffers_.size(),
                            opDesc.inputDesc.data(),
                            inputBuffers_.data(),
                            outputBuffers_.size(),
                            opDesc.outputDesc.data(),
                            outputBuffers_.data(),
                            opDesc.opAttr,
                            acl_context->get_stream(0)));

            AclSafeCall(aclrtSynchronizeStream(acl_context->get_stream(0)));


        }

        aclMat& Runop(vector<aclMat>& input, vector<aclMat>& output, const string opType) 
        {
            size_t i;

            vector<aclDataBuffer *> inputBuffers_;
            vector<aclDataBuffer *> outputBuffers_;

            for (i = 0; i < input.size(); ++i)
                inputBuffers_.emplace_back(aclCreateDataBuffer(input[i].data, input[i].totalSize));
            for (i = 0; i < output.size(); ++i)
                outputBuffers_.emplace_back(aclCreateDataBuffer(output[i].data, output[i].totalSize));
            OperatorDesc opDesc = CreateOpDesc(opType, input, output);

            compileAndRunop(opDesc, inputBuffers_, outputBuffers_, output[0].acl_context);
            output[0].data = aclGetDataBufferAddr(outputBuffers_[0]);

            for (i = 0; i < input.size(); ++i)
                AclSafeCall(aclDestroyDataBuffer(inputBuffers_[i]));
            for (i = 0; i < output.size(); ++i)
                AclSafeCall(aclDestroyDataBuffer(outputBuffers_[i]));
            
            return output[0];
        }

        void OneInAndOneOut(const aclMat& inputMat, aclMat& outputMat, const string opType)
        {
            vector<aclMat> input_Mat;
            vector<aclMat> output_Mat;

            input_Mat.emplace_back(inputMat);
            output_Mat.emplace_back(outputMat);

            outputMat = Runop(input_Mat, output_Mat, opType);
        }

        void TwoInAndOneOut(const aclMat& inputMat, const aclMat& inputMatOther, aclMat& outputMat, const string opType) 
        {
            vector<aclMat> input_Mat;
            vector<aclMat> output_Mat;

            input_Mat.emplace_back(inputMat);
            input_Mat.emplace_back(inputMatOther);
            output_Mat.emplace_back(outputMat);
           
            outputMat = Runop(input_Mat, output_Mat, opType);
        }
    }
}

