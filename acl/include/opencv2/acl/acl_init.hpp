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
#ifndef OPENCV_ACL_INIT_HPP
#define OPENCV_ACL_INIT_HPP

#include <memory>
#include <vector>

#include "acl_type.hpp"
#include "opencv2/core.hpp"

namespace cv {
namespace acl {
CV_EXPORTS Mutex &getInitMutex();
//////////////////////////////// aclEnv ////////////////////////////////
class CV_EXPORTS aclEnv {
 public:
  aclEnv();
  aclEnv(const char *config_path);
  static aclEnv *get_acl_env(const char *config_path);
  int get_device_count();
  int *refcount;
  ~aclEnv();

 private:
  uint32_t _device_count;
};

//////////////////////////////// aclCxt ////////////////////////////////
class CV_EXPORTS aclCxt {
 public:
  aclCxt();
  aclCxt(int device_id);

  aclrtContext *get_context();
  void set_current_context();

  void create_stream(int count = 1);
  aclStream get_stream(const size_t index = 0);
  ~aclCxt();

 private:
  int32_t _device_id;
  aclrtContext *_context;
  std::vector<aclStream> _acl_streams;
};

CV_EXPORTS void wait_stream(aclCxt *context, const int stream_id = 0);
//////////////////////////////// device ////////////////////////////////
CV_EXPORTS aclCxt *set_device(const char *config_path, int device_id = 0,
                              int stream_count = 1);
CV_EXPORTS void release_device(aclCxt *context);
} /* end of namespace acl */
} /* end of namespace cv */

#endif