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



namespace cv
{
    namespace acl
    {
        ///////////////////////////aclEnv//////////////////////////////////
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


        /////////////////////////create acl context////////////////////////
        /**
         *  @brief: set device and stream 
         *  @param [in] config_path: ajson path
         *  @param [in] device_id: device id
         *  @param [in] stream_count: stream count
         */
        aclCxt *set_device(const char* config_path, int device_id, int stream_count)
        {
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
            context = nullptr;
            if (global_aclenv->refcount)
            {
                AutoLock lock(getInitMutex());
                CV_XADD(global_aclenv->refcount, -1);
                
                if (*(global_aclenv->refcount) == 0)
                {
                    delete global_aclenv;
                    global_aclenv = nullptr;
                }
            }
            clog << "release_device() is success" << endl;
        }

    } /* end of namespace acl */

} /* end of namespace cv */
