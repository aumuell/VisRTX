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

#pragma once

#include "gpu/gpu_util.h"

namespace visrtx {

struct LinearBSplineSegment
{
  RT_FUNCTION LinearBSplineSegment(const vec4 *q)
  {
    p[0] = q[0];
    p[1] = q[1] - q[0];
  }

  RT_FUNCTION float radius(const float &u) const
  {
    return p[0].w + p[1].w * u;
  }

  RT_FUNCTION vec3 position3(float u) const
  {
    return (vec3 &)p[0] + u * (vec3 &)p[1];
  }
  RT_FUNCTION vec4 position4(float u) const
  {
    return p[0] + u * p[1];
  }

  RT_FUNCTION float min_radius(float u1, float u2) const
  {
    return fminf(radius(u1), radius(u2));
  }

  RT_FUNCTION float max_radius(float u1, float u2) const
  {
    if (!p[1].w)
      return p[0].w;
    return fmaxf(radius(u1), radius(u2));
  }

  RT_FUNCTION vec3 velocity3(float u) const
  {
    return (vec3 &)p[1];
  }
  RT_FUNCTION vec4 velocity4(float u) const
  {
    return p[1];
  }

  RT_FUNCTION vec3 acceleration3(float u) const
  {
    return vec3(0.f);
  }

  vec4 p[2];
};

struct QuadraticBSplineSegment
{
  RT_FUNCTION QuadraticBSplineSegment(const vec4 *q)
  {
    initializeFromBSpline(q);
  }

  RT_FUNCTION void initializeFromBSpline(const vec4 *q)
  {
    p[0] = q[1] / 2.0f + q[0] / 2.0f;
    p[1] = q[1] - q[0];
    p[2] = q[0] / 2.0f - q[1] + q[2] / 2.0f;
  }

  RT_FUNCTION void export2BSpline(vec4 bs[3]) const
  {
    bs[0] = p[0] - p[1] / 2.f;
    bs[1] = p[0] + p[1] / 2.f;
    bs[2] = p[0] + 1.5f * p[1] + 2.f * p[2];
  }

  RT_FUNCTION vec3 position3(float u) const
  {
    return (vec3 &)p[0] + u * (vec3 &)p[1] + u * u * (vec3 &)p[2];
  }
  RT_FUNCTION vec4 position4(float u) const
  {
    return p[0] + u * p[1] + u * u * p[2];
  }

  RT_FUNCTION float radius(float u) const
  {
    return p[0].w + u * (p[1].w + u * p[2].w);
  }

  RT_FUNCTION float min_radius(float u1, float u2) const
  {
    float root1 = glm::clamp(-0.5f * p[1].w / p[2].w, u1, u2);
    return fminf(fminf(radius(u1), radius(u2)), radius(root1));
  }

  RT_FUNCTION float max_radius(float u1, float u2) const
  {
    if (!p[1].w && !p[2].w)
      return p[0].w; // a quick bypass for constant width
    float root1 = glm::clamp(-0.5f * p[1].w / p[2].w, u1, u2);
    return fmaxf(fmaxf(radius(u1), radius(u2)), radius(root1));
  }

  RT_FUNCTION vec3 velocity3(float u) const
  {
    return (vec3 &)p[1] + 2.f * u * (vec3 &)p[2];
  }
  RT_FUNCTION vec4 velocity4(float u) const
  {
    return p[1] + 2.f * u * p[2];
  }

  RT_FUNCTION vec3 acceleration3(float u) const
  {
    return 2.f * (vec3 &)p[2];
  }
  RT_FUNCTION vec4 acceleration4(float u) const
  {
    return 2.f * p[2];
  }

  RT_FUNCTION float derivative_of_radius(float u) const
  {
    return p[1].w + 2.f * u * p[2].w;
  }

  vec4 p[3];
};

template <typename CurveSegmentType>
RT_FUNCTION vec3 curveSurfaceNormal(
    const CurveSegmentType &bc, float u, const vec3 &ps)
{
  constexpr bool linear =
      std::is_same_v<CurveSegmentType, LinearBSplineSegment>;

  vec3 normal;
  if (u == 0.0f) {
    if constexpr (linear)
      normal = ps - (vec3 &)(bc.p[0]); // round endcaps
    else
      normal = -bc.velocity3(0); // flat endcaps
  } else if (u == 1.0f) {
    if constexpr (linear) {
      const vec3 p1 = (vec3 &)(bc.p[1]) + (vec3 &)(bc.p[0]);
      normal = ps - p1; // round endcaps
    } else
      normal = bc.velocity3(1); // flat endcaps
  } else {
    const vec4 p4 = bc.position4(u);
    const vec3 p(p4);
    const float r = p4.w;
    const vec4 d4 = bc.velocity4(u);
    const vec3 d = make_vec3(d4);

    float dd = dot(d, d);

    vec3 o1 = ps - p;
    o1 -= (dot(o1, d) / dd) * d;
    o1 *= r / length(o1);
    normal = normalize(o1);
  }

  return normal;
}

} // namespace visrtx