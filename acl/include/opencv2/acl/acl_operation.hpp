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

#ifndef __OPENCV_ACL_OPERATIONS_HPP__
#define __OPENCV_ACL_OPERATIONS_HPP__

#include "opencv2/acl/acl.hpp"
#include "acl/acl.h"
#include <iostream>

#define AclSafeCall(expr) __aclSafeCall(expr, __FILE__, __LINE__, __func__)
#define AclVerifyCall(expr) __aclSafeCall(res, __FILE__, __LINE__, __func__)

using namespace std;

namespace cv
{
    namespace acl
    {
        /////////////////////////////SafeCall/////////////////////////////

        static inline void __aclSafeCall(int err, const char* file, const int line, const char *func="")
        {
            if(0 !=  err)
            {
                const char* function = func ? func : "unknown function";
                std::cerr << "Acl Called Error: " << "file " << file  << ", func " << function << ", line " << line;
                std::cerr.flush();
            }
        }

        
        ///////////////////////////aclEnv//////////////////////////////////
        inline aclEnv::aclEnv() {}
        inline aclEnv::aclEnv(const char* config_path)
        {
            uint32_t device_count;

            AclSafeCall(aclInit(config_path));
            
            AclSafeCall(aclrtGetDeviceCount(&device_count));  

            _device_count = device_count;
            refcount = static_cast<int *>(fastMalloc(sizeof(*refcount)));
            *refcount = 0;
            
            clog << "aclInit() is success" << endl;
        }
        
        inline int aclEnv::get_device_count()
        {
            return _device_count;
        }
        inline aclEnv::~aclEnv()
        {
            AclSafeCall(aclFinalize());
            clog << "aclFinalize() is success" << endl;
        }

        /////////////////////////////////////////aclCxt////////////////////////////
        inline aclCxt::aclCxt() {};

        inline aclCxt::aclCxt(int device_id) : _device_id(device_id)
        {
            _context = static_cast<aclrtContext *>(fastMalloc(sizeof(*_context)));
            AclSafeCall(aclrtCreateContext(_context, _device_id));

            clog << "aclrtCreateContext() is success" << endl;
        }

        inline aclrtContext* aclCxt::get_context()
        {
            return _context;
        }
        
        inline aclrtStream aclCxt::get_stream(const size_t index)
        {
            CV_Assert(index < _acl_streams.size());
            return _acl_streams[index];
        }

        inline void aclCxt::set_current_context()
        {
            AclSafeCall(aclrtSetCurrentContext(*_context));
        }

        inline aclCxt::~aclCxt()
        {
            size_t i = 0;
            
            AclSafeCall(aclrtSetCurrentContext(*_context));
            for (i = 0; i < _acl_streams.size(); i++)
            {
                aclStream acl_stream = _acl_streams[i];
                AclSafeCall(aclrtDestroyStream(acl_stream));
            }

            clog << "aclrtDestroyStream() is success" << endl;

            std::vector<aclrtStream>().swap(_acl_streams);
            AclSafeCall(aclrtDestroyContext(*_context));

            clog << "aclrtDestroyContext() is success" << endl;
        }
    }
}

#endif