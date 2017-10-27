#ifndef PX_CG_UTIL_RANDOM_HPP
#define PX_CG_UTIL_RANDOM_HPP

#include <random>
#include "util/cuda.hpp"
#ifdef USE_CUDA
#  include <curand.h>
#  include <curand_kernel.h>
#endif

namespace px
{

namespace rnd
{
inline PREC rnd_cpu()
{
    std::uniform_real_distribution<PREC> static rd(-1, 1);
    std::mt19937 static sd(std::random_device{}());

    return rd(sd);
}


#ifdef USE_CUDA

    __device__
inline PREC rnd_gpu(curandState_t * const &state)
{
          // use thread-based random state for better performance
//        static curandState_t *state = nullptr;
//        if (state == nullptr)
//        {
//            state = new curandState_t;
//            curand_init(clock(), 0, 0, state);
//        }
        return curand_uniform(state)*2 - 1;
}

#else
template<typename T>
[[ noreturn ]]
inline PREC rnd_gpu(T)
{
    throw CUDAError("Failed to generate random number using GPU without the support of CUDA.");
}
#endif

}

}
#endif // PX_CG_UTIL_RANDOM_HPP