#include "CUDADataFormats/TrackingRecHit2DCUDA.h"
#include "CUDACore/copyAsync.h"
#include "CUDACore/cudaCheck.h"
#include "CUDACore/device_unique_ptr.h"
#include "CUDACore/host_unique_ptr.h"

template <>
cms::cuda::host::unique_ptr<float[]> TrackingRecHit2DCUDA::localCoordToHostAsync(hipStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<float[]>(4 * nHits(), stream);
  cms::cuda::copyAsync(ret, m_store32, 4 * nHits(), stream);
  return ret;
}

template <>
cms::cuda::host::unique_ptr<uint32_t[]> TrackingRecHit2DCUDA::hitsModuleStartToHostAsync(hipStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<uint32_t[]>(2001, stream);
  cudaCheck(hipMemcpyAsync(ret.get(), m_hitsModuleStart, 4 * 2001, hipMemcpyDefault, stream));
  return ret;
}

template <>
cms::cuda::host::unique_ptr<float[]> TrackingRecHit2DCUDA::globalCoordToHostAsync(hipStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<float[]>(4 * nHits(), stream);
  cudaCheck(hipMemcpyAsync(
      ret.get(), m_store32.get() + 4 * nHits(), 4 * nHits() * sizeof(float), hipMemcpyDefault, stream));
  return ret;
}

template <>
cms::cuda::host::unique_ptr<int32_t[]> TrackingRecHit2DCUDA::chargeToHostAsync(hipStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<int32_t[]>(nHits(), stream);
  cudaCheck(
      hipMemcpyAsync(ret.get(), m_store32.get() + 8 * nHits(), nHits() * sizeof(int32_t), hipMemcpyDefault, stream));
  return ret;
}

template <>
cms::cuda::host::unique_ptr<int16_t[]> TrackingRecHit2DCUDA::sizeToHostAsync(hipStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<int16_t[]>(2 * nHits(), stream);
  cudaCheck(hipMemcpyAsync(
      ret.get(), m_store16.get() + 2 * nHits(), 2 * nHits() * sizeof(int16_t), hipMemcpyDefault, stream));
  return ret;
}
