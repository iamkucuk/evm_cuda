// Minimal DLPack definitions, vendored.
//
// DLPack is the small, stable C structure that array libraries agree on so one
// of them can hand a device pointer to another without copying. PyTorch, CuPy
// and JAX all consume it. Only the handful of declarations below are needed to
// produce and consume a tensor, so they are written out here instead of taking
// on a header dependency for roughly sixty lines. The layout is ABI-stable: it
// is what every consumer already compiles against, and changing it would break
// them, so it does not drift.
//
// Reference: https://github.com/dmlc/dlpack (structures as of ABI version 0.8,
// which is what current PyTorch and CuPy releases accept).
#pragma once

#include <cstdint>

extern "C" {

// Which kind of memory the pointer refers to. Only the two values this project
// can produce are named; the enum itself is open.
typedef enum {
    kDLCPU = 1,
    kDLCUDA = 2,
} DLDeviceType;

typedef struct {
    int32_t device_type;
    int32_t device_id;
} DLDevice;

// How to read the bits at the pointer. `code` picks the family, `bits` the
// width, `lanes` is 1 for everything this project produces.
typedef enum {
    kDLInt = 0,
    kDLUInt = 1,
    kDLFloat = 2,
} DLDataTypeCode;

typedef struct {
    uint8_t code;
    uint8_t bits;
    uint16_t lanes;
} DLDataType;

typedef struct {
    void* data;
    DLDevice device;
    int32_t ndim;
    DLDataType dtype;
    int64_t* shape;
    // A null `strides` means "C-contiguous", which is the only layout this
    // project produces.
    int64_t* strides;
    uint64_t byte_offset;
} DLTensor;

// The owning wrapper. `manager_ctx` carries whatever the producer needs to keep
// the memory alive; `deleter` is called by the consumer when it is done, and is
// where that ownership is released.
typedef struct DLManagedTensor {
    DLTensor dl_tensor;
    void* manager_ctx;
    void (*deleter)(struct DLManagedTensor* self);
} DLManagedTensor;

}  // extern "C"
