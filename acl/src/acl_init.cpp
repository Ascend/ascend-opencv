/*
 * Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "precomp.hpp"

using namespace std;
using namespace cv;
using namespace cv::acl;
namespace cv {
namespace acl {
///////////////////////////aclEnv//////////////////////////////////
static Mutex *__initmutex = NULL;
Mutex &getInitMutex() {
  if (__initmutex == NULL) __initmutex = new Mutex();
  return *__initmutex;
}

aclEnv *global_aclenv = nullptr;
aclEnv *aclEnv::get_acl_env(const char *config_path) {
  if (nullptr == global_aclenv) {
    AutoLock lock(getInitMutex());
    if (nullptr == global_aclenv) global_aclenv = new aclEnv(config_path);
  }
  return global_aclenv;
}

void wait_stream(aclCxt *acl_context, const int stream_id) {
  aclrtSynchronizeStream(acl_context->get_stream(stream_id));
}

/////////////////////////create acl context////////////////////////
/**
 *  @brief: set device and stream
 *  @param [in] config_path: ajson path
 *  @param [in] device_id: device id
 *  @param [in] stream_count: stream count
 */
aclCxt *set_device(const char *config_path, int device_id, int stream_count) {
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

void release_device(aclCxt *context) {
  CV_Assert(context);
  delete context;
  context = nullptr;
  if (global_aclenv->refcount) {
    AutoLock lock(getInitMutex());
    CV_XADD(global_aclenv->refcount, -1);

    if (*(global_aclenv->refcount) == 0) {
      delete global_aclenv;
      global_aclenv = nullptr;
    }
  }
  clog << "release_device() is success" << endl;
}
} /* end of namespace acl */
} /* end of namespace cv */
