#include "object/material/base_material.hpp"

using namespace px;

MaterialObj::MaterialObj(void * obj,
            fnAmbient_t const &fn_ambient, fnDiffuse_t const &fn_diffuse,
            fnSpecular_t const &fn_specular, fnSpecularExp_t const &fn_specular_exp,
            fnTransmissive_t const &fn_transmissive, fnRefractiveIndex_t const &fn_refractive_index)
        : obj(obj),
          fn_ambient(fn_ambient), fn_diffuse(fn_diffuse),
          fn_specular(fn_specular), fn_specular_exp(fn_specular_exp),
          fn_transmissive(fn_transmissive), fn_refractive_index(fn_refractive_index)
{}

BaseMaterial::BaseMaterial()
    : dev_ptr(nullptr)
{}

void BaseMaterial::clearGpuData()
{
#ifdef USE_CUDA
    if (dev_ptr == nullptr)
        return;
    PX_CUDA_CHECK(cudaFree(dev_ptr));
    dev_ptr = nullptr;
#endif
}
