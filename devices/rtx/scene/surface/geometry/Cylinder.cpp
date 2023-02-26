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

#include "Cylinder.h"
// std
#include <algorithm>
#include <numeric>

namespace visrtx {

Cylinder::Cylinder(DeviceGlobalState *d) : Geometry(d) {}

Cylinder::~Cylinder()
{
  cleanup();
}

void Cylinder::commit()
{
  Geometry::commit();

  cleanup();

  m_index = getParamObject<Array1D>("primitive.index");
  m_radius = getParamObject<Array1D>("primitive.radius");
  m_caps = getParamString("caps", "none") != "none";

  m_vertexPosition = getParamObject<Array1D>("vertex.position");
  m_vertexColor = getParamObject<Array1D>("vertex.color");
  m_vertexAttribute0 = getParamObject<Array1D>("vertex.attribute0");
  m_vertexAttribute1 = getParamObject<Array1D>("vertex.attribute1");
  m_vertexAttribute2 = getParamObject<Array1D>("vertex.attribute2");
  m_vertexAttribute3 = getParamObject<Array1D>("vertex.attribute3");

  if (!m_vertexPosition) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'vertex.position' on cylinders geometry");
    return;
  }

  reportMessage(ANARI_SEVERITY_DEBUG,
      "committing %s cylinder geometry",
      m_index ? "indexed" : "soup");

  if (m_index)
    m_index->addCommitObserver(this);
  m_vertexPosition->addCommitObserver(this);

  m_globalRadius = getParam<float>("radius", 1.f);

  computeCylinders();

  m_vertexBufferPtr = (CUdeviceptr)m_generatedVertices.dataDevice();
  m_radiusBufferPtr = (CUdeviceptr)m_generatedRadii.dataDevice();

  upload();
}

void Cylinder::populateBuildInput(OptixBuildInput &buildInput) const
{
  buildInput.type = OPTIX_BUILD_INPUT_TYPE_CURVES;

  auto &curveArray = buildInput.curveArray;
  curveArray.curveType = OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE;
  curveArray.flag = OPTIX_GEOMETRY_FLAG_NONE;
  curveArray.endcapFlags =
      m_caps ? OPTIX_CURVE_ENDCAP_ON : OPTIX_CURVE_ENDCAP_DEFAULT;

  curveArray.vertexStrideInBytes = sizeof(vec3);
  curveArray.numVertices = m_generatedVertices.size();
  curveArray.vertexBuffers = &m_vertexBufferPtr;

  curveArray.widthStrideInBytes = sizeof(float);
  curveArray.widthBuffers = &m_radiusBufferPtr;

  curveArray.indexStrideInBytes = sizeof(uint32_t);
  curveArray.numPrimitives = m_generatedIndices.size();
  curveArray.indexBuffer = (CUdeviceptr)m_generatedIndices.dataDevice();

  curveArray.normalBuffers = 0;
  curveArray.normalStrideInBytes = 0;
}

void Cylinder::computeCylinders()
{
  const auto numCylinders =
      m_index ? m_index->size() : m_vertexPosition->size() / 2;
  const auto numVertices = 3 * numCylinders;
  m_generatedVertices.resize(numVertices);
  m_generatedRadii.resize(numVertices);
  m_generatedIndices.resize(numCylinders);
  m_generatedAttributeIndices.resize(numCylinders);

  const auto *vIn = m_vertexPosition->beginAs<vec3>();
  const auto *crIn = m_radius ? m_radius->beginAs<float>() : nullptr;

  auto *vOut = m_generatedVertices.dataHost();
  auto *vrOut = m_generatedRadii.dataHost();

  if (m_index) {
    const auto *idxBegin = m_index->beginAs<uvec2>();
    const auto *idxEnd = m_index->endAs<uvec2>();
    auto *vaOut = m_generatedAttributeIndices.dataHost();
    uint32_t cID = 0;
    std::for_each(idxBegin, idxEnd, [&](const uvec2 &idx) {
      vaOut[cID] = idx;
      const auto v1 = vIn[idx.x];
      const auto v3 = vIn[idx.y];
      const auto v2 = (v1 + v3) / 2.f;
      vOut[cID + 0] = v1;
      vOut[cID + 1] = v2;
      vOut[cID + 2] = v3;
      vrOut[cID + 0] = crIn ? crIn[cID] : m_globalRadius;
      vrOut[cID + 1] = crIn ? crIn[cID] : m_globalRadius;
      vrOut[cID + 2] = crIn ? crIn[cID] : m_globalRadius;
      cID += 3;
    });
  } else {
    uint32_t cID = 0;
    for (uint32_t i = 0; i < numCylinders; i++) {
      const auto vi1 = i * 2 + 0;
      const auto vi2 = i * 2 + 1;

      const auto v1 = vIn[vi1];
      const auto v3 = vIn[vi2];
      const auto v2 = (v1 + v3) / 2.f;
      vOut[cID + 0] = v1;
      vOut[cID + 1] = v2;
      vOut[cID + 2] = v3;

      const auto cr = crIn ? crIn[i / 2] : m_globalRadius;
      vrOut[cID + 0] = cr;
      vrOut[cID + 1] = cr;
      vrOut[cID + 2] = cr;

      cID += 3;
    }

    auto *aIdx = (uint32_t *)m_generatedAttributeIndices.dataHost();
    std::iota(aIdx, aIdx + numCylinders * 2, 0);
  }

  auto *idx = m_generatedIndices.dataHost();
  std::iota(idx, idx + numCylinders, 0);
  std::transform(idx, idx + numCylinders, idx, [](auto &i) { return i * 3; });

  m_generatedVertices.upload();
  m_generatedIndices.upload();
  m_generatedAttributeIndices.upload();
  m_generatedRadii.upload();
}

GeometryGPUData Cylinder::gpuData() const
{
  auto retval = Geometry::gpuData();
  retval.type = GeometryType::CYLINDER;

  auto &cylinder = retval.cylinder;

  cylinder.vertices = m_generatedVertices.dataDevice();
  cylinder.indices = m_generatedIndices.dataDevice();
  cylinder.radii = m_generatedRadii.dataDevice();
  cylinder.attrIndices = m_generatedAttributeIndices.dataDevice();

  populateAttributePtr(m_vertexAttribute0, cylinder.vertexAttr[0]);
  populateAttributePtr(m_vertexAttribute1, cylinder.vertexAttr[1]);
  populateAttributePtr(m_vertexAttribute2, cylinder.vertexAttr[2]);
  populateAttributePtr(m_vertexAttribute3, cylinder.vertexAttr[3]);
  populateAttributePtr(m_vertexColor, cylinder.vertexAttr[4]);

  cylinder.indexed = m_index;

  return retval;
}

GeometryType Cylinder::geometryType() const
{
  return GeometryType::CYLINDER;
}

bool Cylinder::isValid() const
{
  return m_vertexPosition;
}

void Cylinder::cleanup()
{
  if (m_index)
    m_index->removeCommitObserver(this);
  if (m_vertexPosition)
    m_vertexPosition->removeCommitObserver(this);
}

} // namespace visrtx
