/*
 * Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "array/Array2D.h"
#include "utility/CudaImageTexture.h"

namespace visrtx {

Array2D::Array2D(DeviceGlobalState *state, const Array2DMemoryDescriptor &d)
    : Array(ANARI_ARRAY2D, state, d)
{
  if (d.byteStride1 != 0 || d.byteStride2 != 0)
    throw std::runtime_error("strided arrays not yet supported!");

  m_size[0] = d.numItems1;
  m_size[1] = d.numItems2;

  initManagedMemory();
}

size_t Array2D::totalSize() const
{
  return size(0) * size(1);
}

size_t Array2D::size(int dim) const
{
  return m_size[dim];
}

uvec2 Array2D::size() const
{
  return uvec2(uint32_t(size(0)), uint32_t(size(1)));
}

void Array2D::privatize()
{
  makePrivatizedCopy(size(0) * size(1));
}

cudaArray_t Array2D::acquireCUDAArrayFloat()
{
  if (!m_cuArrayFloat)
    makeCudaArrayFloat(m_cuArrayFloat, *this, size());
  m_arrayRefCountFloat++;
  return m_cuArrayFloat;
}

void Array2D::releaseCUDAArrayFloat()
{
  m_arrayRefCountFloat--;
  if (m_arrayRefCountFloat == 0) {
    cudaFreeArray(m_cuArrayFloat);
    m_cuArrayFloat = {};
  }
}

cudaArray_t Array2D::acquireCUDAArrayUint8()
{
  if (!m_cuArrayUint8)
    makeCudaArrayUint8(m_cuArrayUint8, *this, size());
  m_arrayRefCountUint8++;
  return m_cuArrayUint8;
}

void Array2D::releaseCUDAArrayUint8()
{
  m_arrayRefCountUint8--;
  if (m_arrayRefCountUint8 == 0) {
    cudaFreeArray(m_cuArrayUint8);
    m_cuArrayUint8 = {};
  }
}

void Array2D::uploadArrayData() const
{
  Array::uploadArrayData();
  if (m_cuArrayFloat)
    makeCudaArrayFloat(m_cuArrayFloat, *this, size());
  if (m_cuArrayUint8)
    makeCudaArrayUint8(m_cuArrayUint8, *this, size());
}

} // namespace visrtx

VISRTX_ANARI_TYPEFOR_DEFINITION(visrtx::Array2D *);
