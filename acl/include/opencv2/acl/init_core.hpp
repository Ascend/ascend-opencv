#ifndef OPENCV_INIT_CORE_HPP
#define OPENCV_INIT_CORE_HPP

#include "acl_init.hpp"

namespace cv {
namespace acl {
///////////////////////////aclEnv//////////////////////////////////
/**
 * acl init
 */
inline aclEnv::aclEnv() {}

inline aclEnv::aclEnv(const char *config_path) {
  uint32_t device_count;

  AclSafeCall(aclInit(config_path));

  AclSafeCall(aclrtGetDeviceCount(&device_count));

  _device_count = device_count;
  // Reference Counting
  refcount = static_cast<int *>(fastMalloc(sizeof(*refcount)));
  *refcount = 0;

  std::clog << "aclInit() is success" << std::endl;
}

inline int aclEnv::get_device_count() { return _device_count; }

inline aclEnv::~aclEnv() {
  AclSafeCall(aclFinalize());
  std::clog << "aclFinalize() is success" << std::endl;
}

/////////////////////////////////////////aclCxt////////////////////////////
inline aclCxt::aclCxt(){};

inline aclCxt::aclCxt(int device_id) : _device_id(device_id) {
  _context = static_cast<aclrtContext *>(fastMalloc(sizeof(*_context)));
  AclSafeCall(aclrtCreateContext(_context, _device_id));

  std::clog << "aclrtCreateContext() is success" << std::endl;
}

inline aclrtContext *aclCxt::get_context() { return _context; }

/**
 *  set current context
 */
inline void aclCxt::set_current_context() {
  AclSafeCall(aclrtSetCurrentContext(*_context));
}

inline void aclCxt::create_stream(int count) {
  CV_Assert(count > 0);

  int i;
  for (i = 0; i < count; i++) {
    aclStream stream;
    AclSafeCall(aclrtCreateStream(&stream));

    _acl_streams.push_back(stream);
  }

  std::clog << "aclrtCreateStream() is success" << std::endl;
}

inline aclrtStream aclCxt::get_stream(const size_t index) {
  CV_Assert(index < _acl_streams.size());
  return _acl_streams[index];
}

/**
 * destroy stream and context
 */
inline aclCxt::~aclCxt() {
  size_t i = 0;

  AclSafeCall(aclrtSetCurrentContext(*_context));
  for (i = 0; i < _acl_streams.size(); i++) {
    aclStream acl_stream = _acl_streams[i];
    AclSafeCall(aclrtDestroyStream(acl_stream));
  }

  std::clog << "aclrtDestroyStream() is success" << std::endl;

  // empty vector
  std::vector<aclrtStream>().swap(_acl_streams);
  AclSafeCall(aclrtDestroyContext(*_context));

  std::clog << "aclrtDestroyContext() is success" << std::endl;
}
} /* end of namespace acl */
} /* end of namespace cv */

#endif