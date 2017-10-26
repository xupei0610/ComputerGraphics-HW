#include "object/material/base_material.hpp"

using namespace px;

PX_CUDA_CALLABLE
BaseMaterial::BaseMaterial(const BumpMapping * const &bump_mapping)
        : _bump_mapping(bump_mapping)
{}


BumpMapping::BumpMapping()
    : _dev_ptr(nullptr), _need_update(true)
{}

BumpMapping::~BumpMapping()
{
#ifdef  USE_CUDA
    clearGpuData();
#endif
}

PX_CUDA_CALLABLE
Light BumpMapping::color(PREC const &u, PREC const &v) const
{
    return {0,0,0};
}

void BumpMapping::up2Gpu() {
#ifdef USE_CUDA
    if (_need_update)
    {
        if (_dev_ptr == nullptr)
            PX_CUDA_CHECK(cudaMalloc(&_dev_ptr, sizeof(BumpMapping)));

        PX_CUDA_CHECK(cudaMemcpy(_dev_ptr, this, sizeof(BumpMapping),
                                 cudaMemcpyHostToDevice));

        _need_update = false;
    }
#endif
}
void BumpMapping::clearGpuData() {
#ifdef USE_CUDA
    if (_dev_ptr == nullptr)
        return;

    PX_CUDA_CHECK(cudaFree(_dev_ptr));

    _dev_ptr = nullptr;
    _need_update = true;
#endif
}
