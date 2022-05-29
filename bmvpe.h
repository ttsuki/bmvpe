///
/// bmvpe - Typed DirectXMath Vector/Matrix Wrapper Library
///
/// Copyright (C) 2020-2022 ttsuki. All rights reserved.
/// Licensed under the MIT License.
///

#pragma once

#include <array>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <type_traits>

#ifndef DIRECTX_MATH_VERSION
#include <DirectXMath.h>
#include <DirectXPackedVector.h>
#endif

#define BMVPE_API inline __forceinline auto XM_CALLCONV
#define BMVPE_EXPECTED(...) ((void)(assert(__VA_ARGS__)), __VA_ARGS__)

namespace bmvpe
{
    inline namespace scalar
    {
        BMVPE_API cmp_eq(float a, float b) noexcept -> bool { return a == b; }
        BMVPE_API cmp_near_eq(float a, float b, float eps) noexcept -> bool { return DirectX::XMScalarNearEqual(a, b, eps); }
        BMVPE_API cmp_neq(float a, float b) noexcept -> bool { return a != b; }
        BMVPE_API cmp_gt(float a, float b) noexcept -> bool { return a > b; }
        BMVPE_API cmp_geq(float a, float b) noexcept -> bool { return a >= b; }
        BMVPE_API cmp_lt(float a, float b) noexcept -> bool { return a < b; }
        BMVPE_API cmp_leq(float a, float b) noexcept -> bool { return a <= b; }

        BMVPE_API in_bounds(float v, float bound) noexcept -> bool { return std::abs(v) <= bound; }
        BMVPE_API is_nan(float v) noexcept -> bool { return std::isnan(v); }
        BMVPE_API is_inf(float v) noexcept -> bool { return std::isinf(v); }

        BMVPE_API neg(float x) noexcept -> float { return -x; }
        BMVPE_API add(float x, float y) noexcept -> float { return x + y; }
        BMVPE_API sub(float x, float y) noexcept -> float { return x - y; }
        BMVPE_API mul(float x, float y) noexcept -> float { return x * y; }
        BMVPE_API div(float x, float y) noexcept -> float { return x / y; }
        BMVPE_API mul_add(float x, float y, float z) noexcept -> float { return x * y + z; }
        BMVPE_API neg_mul_sub(float x, float y, float z) noexcept -> float { return -x * y + z; }
        BMVPE_API mod(float x, float y) noexcept -> float { return std::fmod(x, y); }

        BMVPE_API add_angle(float x, float y) noexcept -> float { return DirectX::XMScalarModAngle(x + y); }
        BMVPE_API sub_angle(float x, float y) noexcept -> float { return DirectX::XMScalarModAngle(x - y); }
        BMVPE_API mod_angle(float v) noexcept -> float { return DirectX::XMScalarModAngle(v); }

        BMVPE_API min(float a, float b) noexcept -> float { return std::min(a, b); }
        BMVPE_API max(float a, float b) noexcept -> float { return std::max(a, b); }
        BMVPE_API clamp(float a, float min, float max) noexcept -> float { return std::clamp(a, min, max); }
        BMVPE_API round(float v) noexcept -> float { return std::round(v); }
        BMVPE_API truncate(float v) noexcept -> float { return std::trunc(v); }
        BMVPE_API floor(float v) noexcept -> float { return std::floor(v); }
        BMVPE_API ceil(float v) noexcept -> float { return std::ceil(v); }
        BMVPE_API saturate(float v) noexcept -> float { return std::clamp(v, 0.0f, 1.0f); }

        BMVPE_API inverse(float x) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVectorReciprocal(DirectX::XMVectorReplicate(x))); }
        BMVPE_API abs(float x) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVectorAbs(DirectX::XMVectorReplicate(x))); }
        BMVPE_API pow(float x, float y) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVectorPow(DirectX::XMVectorReplicate(x), DirectX::XMVectorReplicate(y))); }
        BMVPE_API exp_2(float x) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVectorExp2(DirectX::XMVectorReplicate(x))); }
        BMVPE_API exp_e(float x) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVectorExpE(DirectX::XMVectorReplicate(x))); }
        BMVPE_API log_2(float x) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVectorLog2(DirectX::XMVectorReplicate(x))); }
        BMVPE_API log_e(float x) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVectorLogE(DirectX::XMVectorReplicate(x))); }
        BMVPE_API sqrt(float x) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVectorSqrt(DirectX::XMVectorReplicate(x))); }
        BMVPE_API inverse_sqrt(float x) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVectorReciprocalSqrt(DirectX::XMVectorReplicate(x))); }

        BMVPE_API sin(float x) noexcept -> float { return DirectX::XMScalarSin(x); }
        BMVPE_API cos(float x) noexcept -> float { return DirectX::XMScalarCos(x); }
        BMVPE_API tan(float x) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVectorTanEst(DirectX::XMVectorReplicate(x))); }
        BMVPE_API asin(float x) noexcept -> float { return DirectX::XMScalarASin(x); }
        BMVPE_API acos(float x) noexcept -> float { return DirectX::XMScalarACos(x); }
        BMVPE_API atan(float x) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVectorATan(DirectX::XMVectorReplicate(x))); }
        BMVPE_API atan2(float y, float x) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVectorATan2(DirectX::XMVectorReplicate(y), DirectX::XMVectorReplicate(x))); }
        BMVPE_API sinh(float x) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVectorSinH(DirectX::XMVectorReplicate(x))); }
        BMVPE_API cosh(float x) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVectorCosH(DirectX::XMVectorReplicate(x))); }
        BMVPE_API tanh(float x) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVectorTanH(DirectX::XMVectorReplicate(x))); }

        BMVPE_API sin_cos(float x) noexcept -> std::array<float, 2>
        {
            float s, c;
            DirectX::XMScalarSinCos(&s, &c, x);
            return {s, c};
        }

        namespace est
        {
            BMVPE_API inverse(float x) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVectorReciprocalEst(DirectX::XMVectorReplicate(x))); }
            BMVPE_API sqrt(float x) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVectorSqrtEst(DirectX::XMVectorReplicate(x))); }
            BMVPE_API inverse_sqrt(float x) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVectorReciprocalSqrtEst(DirectX::XMVectorReplicate(x))); }

            BMVPE_API sin(float x) noexcept -> float { return DirectX::XMScalarSinEst(x); }
            BMVPE_API cos(float x) noexcept -> float { return DirectX::XMScalarCosEst(x); }
            BMVPE_API tan(float x) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVectorTanEst(DirectX::XMVectorReplicate(x))); }
            BMVPE_API asin(float x) noexcept -> float { return DirectX::XMScalarASinEst(x); }
            BMVPE_API acos(float x) noexcept -> float { return DirectX::XMScalarACosEst(x); }
            BMVPE_API atan(float x) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVectorATanEst(DirectX::XMVectorReplicate(x))); }
            BMVPE_API atan2(float y, float x) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVectorATan2Est(DirectX::XMVectorReplicate(y), DirectX::XMVectorReplicate(x))); }

            BMVPE_API sin_cos(float x) noexcept -> std::array<float, 2>
            {
                float s, c;
                DirectX::XMScalarSinCosEst(&s, &c, x);
                return {s, c};
            }
        }

        inline namespace interpolation
        {
            /// Linear interpolation:
            /// returns t * y + (1-t) * x = x + t(y-x)
            BMVPE_API lerp(float x, float y, float t) noexcept -> float { return x + t * (y - x); }

            /// AMD's SmoothStep interpolation:
            ///   t = clamp(t, 0.0f, 1.0f)
            ///   t = t * t * (3 - 2 * t)
            /// returns t * x + (1-t) * y
            BMVPE_API smooth_step(float x, float y, float t) noexcept -> float { return t = clamp(t, 0.0f, 1.0f), t = t * t * (3 - 2 * t), t * x + (1 - t) * y; }

            /// Hermite interpolation:
            /// returns (2 * t^3 - 3 * t^2 + 1) * v0 +
            ///         (t^3 - 2 * t^2 + t) * t0 +
            ///            (-2 * t^3 + 3 * t^2) * v1 +
            ///            (t^3 - t^2) * t1
            BMVPE_API hermite(float v0, float t0, float v1, float t1, float t) noexcept -> float
            {
                auto t2 = t * t;
                auto t3 = t2 * t;
                return (2 * t * 3 - 3 * t2 + 1) * v0 +
                    (t3 - 2 * t2 + t) * t0 +
                    (-2 * t3 + 3 * t2) * v1 +
                    (t3 - t2) * t1;
            }

            /// Catmull–Rom interpolation:
            /// returns ((-t^3 + 2 * t^2 - t) * v0 +
            ///         (3 * t^3 - 5 * t^2 + 2) * v1 +
            ///         (-3 * t^3 + 4 * t^2 + t) * v2 +
            ///         (t^3 - t^2) * v3) * 0.5
            BMVPE_API catmull_rom(float v0, float v1, float v2, float v3, float t) noexcept -> float
            {
                auto t2 = t * t;
                auto t3 = t2 * t;
                return ((-t3 + 2 * t2 - t) * v0 +
                    (3 * t3 - 5 * t2 + 2) * v1 +
                    (-3 * t3 + 4 * t2 + t) * v2 +
                    (t3 - t2) * v3) * 0.5f;
            }

            /// Barycentric interpolation
            /// returns v0 + t1 * (v1 - v0) + t2 * (v2 - v0)
            BMVPE_API barycentric(float v0, float v1, float v2, float t1, float t2) noexcept -> float
            {
                return v0 + t1 * (v1 - v0) + t2 * (v2 - v0);
            }
        }
    }

    inline namespace vector
    {
        using VectorSize = size_t;
        static constexpr inline VectorSize VectorSizeUnknown = 0;

        namespace enable
        {
            using vector_type_tag = DirectX::XMVECTOR;
            template <class T, class U, class D = T> using if_same_t = std::enable_if_t<std::is_same_v<T, U>, D>;
            template <class T, class U = T> using for_vector_t = enable::if_same_t<typename T::vector_type_tag, vector_type_tag, U>;;
            template <class T, class U = T> using for_float_vector_t = enable::if_same_t<typename T::Scalar, float, U>;
            template <class T, class U = T> using for_int_vector_t = enable::if_same_t<typename T::Scalar, uint32_t, U>;
        }

        // VectorType implementation helper
        namespace traits
        {
            template <class Scalar> BMVPE_API extract_x(DirectX::XMVECTOR v) -> Scalar = delete;
            template <class Scalar> BMVPE_API extract_y(DirectX::XMVECTOR v) -> Scalar = delete;
            template <class Scalar> BMVPE_API extract_z(DirectX::XMVECTOR v) -> Scalar = delete;
            template <class Scalar> BMVPE_API extract_w(DirectX::XMVECTOR v) -> Scalar = delete;
            template <class Scalar> BMVPE_API insert_x(DirectX::XMVECTOR v, std::decay_t<Scalar> e) -> DirectX::XMVECTOR = delete;
            template <class Scalar> BMVPE_API insert_y(DirectX::XMVECTOR v, std::decay_t<Scalar> e) -> DirectX::XMVECTOR = delete;
            template <class Scalar> BMVPE_API insert_z(DirectX::XMVECTOR v, std::decay_t<Scalar> e) -> DirectX::XMVECTOR = delete;
            template <class Scalar> BMVPE_API insert_w(DirectX::XMVECTOR v, std::decay_t<Scalar> e) -> DirectX::XMVECTOR = delete;
            template <class Scalar> BMVPE_API splat_x(DirectX::XMVECTOR v) -> DirectX::XMVECTOR { return DirectX::XMVectorSplatX(v); }
            template <class Scalar> BMVPE_API splat_y(DirectX::XMVECTOR v) -> DirectX::XMVECTOR { return DirectX::XMVectorSplatY(v); }
            template <class Scalar> BMVPE_API splat_z(DirectX::XMVECTOR v) -> DirectX::XMVECTOR { return DirectX::XMVectorSplatZ(v); }
            template <class Scalar> BMVPE_API splat_w(DirectX::XMVECTOR v) -> DirectX::XMVECTOR { return DirectX::XMVectorSplatW(v); }
            template <class Scalar> BMVPE_API zero() -> DirectX::XMVECTOR { return DirectX::XMVectorZero(); }
            template <class Scalar> BMVPE_API broadcast(std::decay_t<Scalar> x) -> DirectX::XMVECTOR = delete;
            template <class Scalar> BMVPE_API from_values(std::decay_t<Scalar> x, std::decay_t<Scalar> y, std::decay_t<Scalar> z, std::decay_t<Scalar> w) -> DirectX::XMVECTOR = delete;

            template <> BMVPE_API broadcast<float>(std::decay_t<float> x) -> DirectX::XMVECTOR { return DirectX::XMVectorReplicate(x); }
            template <> BMVPE_API from_values<float>(std::decay_t<float> x, std::decay_t<float> y, std::decay_t<float> z, std::decay_t<float> w) -> DirectX::XMVECTOR { return DirectX::XMVectorSet(x, y, z, w); }
            template <> BMVPE_API extract_x<float>(DirectX::XMVECTOR v) -> float { return DirectX::XMVectorGetX(v); }
            template <> BMVPE_API extract_y<float>(DirectX::XMVECTOR v) -> float { return DirectX::XMVectorGetY(v); }
            template <> BMVPE_API extract_z<float>(DirectX::XMVECTOR v) -> float { return DirectX::XMVectorGetZ(v); }
            template <> BMVPE_API extract_w<float>(DirectX::XMVECTOR v) -> float { return DirectX::XMVectorGetW(v); }
            template <> BMVPE_API insert_x<float>(DirectX::XMVECTOR v, std::decay_t<float> e) -> DirectX::XMVECTOR { return DirectX::XMVectorSetX(v, e); }
            template <> BMVPE_API insert_y<float>(DirectX::XMVECTOR v, std::decay_t<float> e) -> DirectX::XMVECTOR { return DirectX::XMVectorSetY(v, e); }
            template <> BMVPE_API insert_z<float>(DirectX::XMVECTOR v, std::decay_t<float> e) -> DirectX::XMVECTOR { return DirectX::XMVectorSetZ(v, e); }
            template <> BMVPE_API insert_w<float>(DirectX::XMVECTOR v, std::decay_t<float> e) -> DirectX::XMVECTOR { return DirectX::XMVectorSetW(v, e); }

            template <> BMVPE_API broadcast<uint32_t>(std::decay_t<uint32_t> x) -> DirectX::XMVECTOR { return DirectX::XMVectorReplicateInt(x); }
            template <> BMVPE_API from_values<uint32_t>(std::decay_t<uint32_t> x, std::decay_t<uint32_t> y, std::decay_t<uint32_t> z, std::decay_t<uint32_t> w) -> DirectX::XMVECTOR { return DirectX::XMVectorSetInt(x, y, z, w); }
            template <> BMVPE_API extract_x<uint32_t>(DirectX::XMVECTOR v) -> uint32_t { return DirectX::XMVectorGetIntX(v); }
            template <> BMVPE_API extract_y<uint32_t>(DirectX::XMVECTOR v) -> uint32_t { return DirectX::XMVectorGetIntY(v); }
            template <> BMVPE_API extract_z<uint32_t>(DirectX::XMVECTOR v) -> uint32_t { return DirectX::XMVectorGetIntZ(v); }
            template <> BMVPE_API extract_w<uint32_t>(DirectX::XMVECTOR v) -> uint32_t { return DirectX::XMVectorGetIntW(v); }
            template <> BMVPE_API insert_x<uint32_t>(DirectX::XMVECTOR v, std::decay_t<uint32_t> e) -> DirectX::XMVECTOR { return DirectX::XMVectorSetIntX(v, e); }
            template <> BMVPE_API insert_y<uint32_t>(DirectX::XMVECTOR v, std::decay_t<uint32_t> e) -> DirectX::XMVECTOR { return DirectX::XMVectorSetIntY(v, e); }
            template <> BMVPE_API insert_z<uint32_t>(DirectX::XMVECTOR v, std::decay_t<uint32_t> e) -> DirectX::XMVECTOR { return DirectX::XMVectorSetIntZ(v, e); }
            template <> BMVPE_API insert_w<uint32_t>(DirectX::XMVECTOR v, std::decay_t<uint32_t> e) -> DirectX::XMVECTOR { return DirectX::XMVectorSetIntW(v, e); }
        }

        /// Immutable Vector Type
        template <class TScalar, VectorSize TSize>
        class VectorType
        {
            DirectX::XMVECTOR v;
            VectorType(DirectX::XMVECTOR v) : v{v} { ; }
            template <class TVectorType> friend auto XM_CALLCONV reinterpret(DirectX::XMVECTOR) noexcept -> TVectorType;

        public:
            using vector_type_tag = enable::vector_type_tag;
            using Scalar = TScalar;
            static inline constexpr VectorSize Size = TSize;

            VectorType() = default;
            operator DirectX::XMVECTOR() const noexcept { return v; }

            [[nodiscard]] BMVPE_API extract_x() const noexcept -> TScalar { return BMVPE_EXPECTED(Size >= 1) ? traits::extract_x<TScalar>(v) : TScalar{}; }
            [[nodiscard]] BMVPE_API extract_y() const noexcept -> TScalar { return BMVPE_EXPECTED(Size >= 2) ? traits::extract_y<TScalar>(v) : TScalar{}; }
            [[nodiscard]] BMVPE_API extract_z() const noexcept -> TScalar { return BMVPE_EXPECTED(Size >= 3) ? traits::extract_z<TScalar>(v) : TScalar{}; }
            [[nodiscard]] BMVPE_API extract_w() const noexcept -> TScalar { return BMVPE_EXPECTED(Size >= 4) ? traits::extract_w<TScalar>(v) : TScalar{}; }
            [[nodiscard]] BMVPE_API with_x(float val) const noexcept -> VectorType { return BMVPE_EXPECTED(Size >= 1), VectorType{traits::insert_x<TScalar>(v, val)}; }
            [[nodiscard]] BMVPE_API with_y(float val) const noexcept -> VectorType { return BMVPE_EXPECTED(Size >= 2), VectorType{traits::insert_y<TScalar>(v, val)}; }
            [[nodiscard]] BMVPE_API with_z(float val) const noexcept -> VectorType { return BMVPE_EXPECTED(Size >= 3), VectorType{traits::insert_z<TScalar>(v, val)}; }
            [[nodiscard]] BMVPE_API with_w(float val) const noexcept -> VectorType { return BMVPE_EXPECTED(Size >= 4), VectorType{traits::insert_w<TScalar>(v, val)}; }
            [[nodiscard]] BMVPE_API splat_x() const noexcept -> VectorType { return VectorType{BMVPE_EXPECTED(Size >= 1) ? traits::splat_x<TScalar>(DirectX::XMVectorSplatX(v)) : traits::zero<TScalar>()}; }
            [[nodiscard]] BMVPE_API splat_y() const noexcept -> VectorType { return VectorType{BMVPE_EXPECTED(Size >= 2) ? traits::splat_y<TScalar>(DirectX::XMVectorSplatY(v)) : traits::zero<TScalar>()}; }
            [[nodiscard]] BMVPE_API splat_z() const noexcept -> VectorType { return VectorType{BMVPE_EXPECTED(Size >= 3) ? traits::splat_z<TScalar>(DirectX::XMVectorSplatZ(v)) : traits::zero<TScalar>()}; }
            [[nodiscard]] BMVPE_API splat_w() const noexcept -> VectorType { return VectorType{BMVPE_EXPECTED(Size >= 4) ? traits::splat_w<TScalar>(DirectX::XMVectorSplatW(v)) : traits::zero<TScalar>()}; }
        };

        template <class TVectorType> BMVPE_API reinterpret(DirectX::XMVECTOR v) noexcept -> TVectorType { return TVectorType{v}; }
        template <class TVectorType> BMVPE_API zero() noexcept -> TVectorType { return reinterpret<TVectorType>(traits::zero<typename TVectorType::Scalar>()); }
        template <class TVectorType> BMVPE_API broadcast(typename TVectorType::Scalar x) noexcept -> enable::for_vector_t<TVectorType> { return reinterpret<TVectorType>(traits::broadcast<typename TVectorType::Scalar>(x)); }

        template <class UVectorType, class TVectorType, enable::if_same_t<typename TVectorType::Scalar, typename UVectorType::Scalar> * = nullptr>
        BMVPE_API resize_cast(TVectorType v) noexcept -> UVectorType
        {
            constexpr VectorSize CommonSize = std::min(TVectorType::Size, UVectorType::Size);
            if constexpr (CommonSize == 1) return reinterpret<UVectorType>(DirectX::XMVectorAndInt(v, traits::from_values<uint32_t>(~0u, 0u, 0u, 0u)));
            else if constexpr (CommonSize == 2) return reinterpret<UVectorType>(DirectX::XMVectorAndInt(v, traits::from_values<uint32_t>(~0u, ~0u, 0u, 0u)));
            else if constexpr (CommonSize == 3) return reinterpret<UVectorType>(DirectX::XMVectorAndInt(v, traits::from_values<uint32_t>(~0u, ~0u, ~0u, 0u)));
            else if constexpr (CommonSize == 4) return reinterpret<UVectorType>(DirectX::XMVectorAndInt(v, traits::from_values<uint32_t>(~0u, ~0u, ~0u, ~0u)));
            else return __assume(0), reinterpret<UVectorType>(v);
        }

        template <VectorSize N> using VecNf = VectorType<float, N>;
        using Vec1f = VecNf<1>;
        using Vec2f = VecNf<2>;
        using Vec3f = VecNf<3>;
        using Vec4f = VecNf<4>;

        template <VectorSize N> using VecNi = VectorType<uint32_t, N>;
        using Vec1i = VecNi<1>;
        using Vec2i = VecNi<2>;
        using Vec3i = VecNi<3>;
        using Vec4i = VecNi<4>;

        BMVPE_API vec1f() noexcept -> Vec1f { return zero<Vec1f>(); }
        BMVPE_API vec2f() noexcept -> Vec2f { return zero<Vec2f>(); }
        BMVPE_API vec3f() noexcept -> Vec3f { return zero<Vec3f>(); }
        BMVPE_API vec4f() noexcept -> Vec4f { return zero<Vec4f>(); }
        BMVPE_API vec1i() noexcept -> Vec1i { return zero<Vec1i>(); }
        BMVPE_API vec2i() noexcept -> Vec2i { return zero<Vec2i>(); }
        BMVPE_API vec3i() noexcept -> Vec3i { return zero<Vec3i>(); }
        BMVPE_API vec4i() noexcept -> Vec4i { return zero<Vec4i>(); }

        BMVPE_API vec1f(float x) noexcept -> Vec1f { return broadcast<Vec1f>(x); }
        BMVPE_API vec2f(float x) noexcept -> Vec2f { return broadcast<Vec2f>(x); }
        BMVPE_API vec3f(float x) noexcept -> Vec3f { return broadcast<Vec3f>(x); }
        BMVPE_API vec4f(float x) noexcept -> Vec4f { return broadcast<Vec4f>(x); }
        BMVPE_API vec1i(uint32_t x) noexcept -> Vec1i { return broadcast<Vec1i>(x); }
        BMVPE_API vec2i(uint32_t x) noexcept -> Vec2i { return broadcast<Vec2i>(x); }
        BMVPE_API vec3i(uint32_t x) noexcept -> Vec3i { return broadcast<Vec3i>(x); }
        BMVPE_API vec4i(uint32_t x) noexcept -> Vec4i { return broadcast<Vec4i>(x); }

        BMVPE_API vec2f(float x, float y) noexcept -> Vec2f { return reinterpret<Vec2f>(DirectX::XMVectorSet(x, y, 0.0f, 1.0f)); }
        BMVPE_API vec3f(float x, float y, float z) noexcept -> Vec3f { return reinterpret<Vec3f>(DirectX::XMVectorSet(x, y, z, 1.0f)); }
        BMVPE_API vec4f(float x, float y, float z, float w) noexcept -> Vec4f { return reinterpret<Vec4f>(DirectX::XMVectorSet(x, y, z, w)); }
        BMVPE_API vec2i(uint32_t x, uint32_t y) noexcept -> Vec2i { return reinterpret<Vec2i>(DirectX::XMVectorSetInt(x, y, 0, 0)); }
        BMVPE_API vec3i(uint32_t x, uint32_t y, uint32_t z) noexcept -> Vec3i { return reinterpret<Vec3i>(DirectX::XMVectorSetInt(x, y, z, 0)); }
        BMVPE_API vec4i(uint32_t x, uint32_t y, uint32_t z, uint32_t w) noexcept -> Vec4i { return reinterpret<Vec4i>(DirectX::XMVectorSetInt(x, y, z, w)); }

        template Vec1f resize_cast<Vec1f>(Vec2f) noexcept; //test

        template <template <class Scalar , VectorSize Size> class VectorType, class Scalar, VectorSize Size>
        BMVPE_API blend(VectorType<uint32_t, Size> selector, VectorType<Scalar, Size> vTrue, VectorType<Scalar, Size> vFalse) noexcept
        -> enable::for_vector_t<VectorType<Scalar, Size>>
        {
            return reinterpret<VectorType<Scalar, Size>>(DirectX::XMVectorSelect(vFalse, vTrue, selector));
        }

        template Vec1f blend(Vec1i, Vec1f, Vec1f) noexcept; // test
        template Vec4f blend(Vec4i, Vec4f, Vec4f) noexcept; // test
        template Vec4i blend(Vec4i, Vec4i, Vec4i) noexcept; // test

        template <uint32_t x, uint32_t y = ~0u, uint32_t z = ~0u, uint32_t w = ~0u,
                  class Scalar,
                  VectorSize SrcSize,
                  VectorSize DstSize = w != ~0u ? 4 : z != ~0u ? 3 : y != ~0u ? 2 : x != ~0u ? 1 : 0,
                  std::enable_if_t< // condition check
                      DstSize == (w != ~0u ? 4 : z != ~0u ? 3 : y != ~0u ? 2 : x != ~0u ? 1 : 0)
                      && (x == ~0u || x < SrcSize)
                      && (y == ~0u || y < SrcSize)
                      && (z == ~0u || z < SrcSize)
                      && (w == ~0u || w < SrcSize)>* = nullptr>
        BMVPE_API shuffle(VectorType<Scalar, SrcSize> v) noexcept -> VectorType<Scalar, DstSize>
        {
            return reinterpret<VectorType<Scalar, DstSize>>(DirectX::XMVectorSwizzle<
                x == ~0u ? 0 : x,
                y == ~0u ? 0 : y,
                z == ~0u ? 0 : z,
                w == ~0u ? 0 : w>(v));
        }

        template Vec1f shuffle<0>(Vec1f) noexcept;    // test
        template Vec2f shuffle<0, 0>(Vec1f) noexcept; // test
        template Vec2f shuffle<0, 2>(Vec3f) noexcept; // test

        template <uint32_t x, uint32_t y = ~0, uint32_t z = ~0, uint32_t w = ~0,
                  class Scalar,
                  VectorSize SrcSize1,
                  VectorSize SrcSize2,
                  VectorSize DstSize = w != ~0u ? 4 : z != ~0u ? 3 : y != ~0u ? 2 : x != ~0u ? 1 : 0,
                  std::enable_if_t< // condition check
                      DstSize == (w != ~0u ? 4 : z != ~0u ? 3 : y != ~0u ? 2 : x != ~0u ? 1 : 0)
                      && (x == ~0u || (x & 4 ? (x & ~4) < SrcSize1 : (x & ~4) < SrcSize2))
                      && (y == ~0u || (y & 4 ? (y & ~4) < SrcSize1 : (y & ~4) < SrcSize2))
                      && (z == ~0u || (z & 4 ? (z & ~4) < SrcSize1 : (z & ~4) < SrcSize2))
                      && (w == ~0u || (w & 4 ? (w & ~4) < SrcSize1 : (w & ~4) < SrcSize2))>* = nullptr>
        BMVPE_API permute(VectorType<Scalar, SrcSize1> v1, VectorType<Scalar, SrcSize2> v2) noexcept -> VectorType<Scalar, DstSize>
        {
            return reinterpret<VectorType<Scalar, DstSize>>(DirectX::XMVectorPermute<
                x == ~0u ? 0 : x,
                y == ~0u ? 0 : y,
                z == ~0u ? 0 : z,
                w == ~0u ? 0 : w>(v1, v2));
        }

        template Vec1f permute<0>(Vec1f, Vec1f) noexcept;          //test
        template Vec1f permute<4>(Vec1f, Vec1f) noexcept;          //test
        template Vec4f permute<0, 1, 4, 5>(Vec2f, Vec2f) noexcept; //test

        template <VectorSize N> BMVPE_API vector_cmp_eq(VecNf<N> v1, VecNf<N> v2) noexcept -> VecNi<N> { return reinterpret<VecNi<N>>(DirectX::XMVectorEqual(v1, v2)); }
        template <VectorSize N> BMVPE_API vector_cmp_eq(VecNi<N> v1, VecNi<N> v2) noexcept -> VecNi<N> { return reinterpret<VecNi<N>>(DirectX::XMVectorEqualInt(v1, v2)); }
        template <VectorSize N> BMVPE_API vector_cmp_near_eq(VecNf<N> v1, VecNf<N> v2, VecNf<N> eps) noexcept -> VecNi<N> { return reinterpret<VecNi<N>>(DirectX::XMVectorNearEqual(v1, v2, eps)); }
        template <VectorSize N> BMVPE_API vector_cmp_neq(VecNf<N> v1, VecNf<N> v2) noexcept -> VecNi<N> { return reinterpret<VecNi<N>>(DirectX::XMVectorNotEqual(v1, v2)); }
        template <VectorSize N> BMVPE_API vector_cmp_neq(VecNi<N> v1, VecNi<N> v2) noexcept -> VecNi<N> { return reinterpret<VecNi<N>>(DirectX::XMVectorNotEqualInt(v1, v2)); }
        template <VectorSize N> BMVPE_API vector_cmp_gt(VecNf<N> v1, VecNf<N> v2) noexcept -> VecNi<N> { return reinterpret<VecNi<N>>(DirectX::XMVectorGreater(v1, v2)); }
        template <VectorSize N> BMVPE_API vector_cmp_geq(VecNf<N> v1, VecNf<N> v2) noexcept -> VecNi<N> { return reinterpret<VecNi<N>>(DirectX::XMVectorGreaterOrEqual(v1, v2)); }
        template <VectorSize N> BMVPE_API vector_cmp_lt(VecNf<N> v1, VecNf<N> v2) noexcept -> VecNi<N> { return reinterpret<VecNi<N>>(DirectX::XMVectorLess(v1, v2)); }
        template <VectorSize N> BMVPE_API vector_cmp_leq(VecNf<N> v1, VecNf<N> v2) noexcept -> VecNi<N> { return reinterpret<VecNi<N>>(DirectX::XMVectorLessOrEqual(v1, v2)); }
        template <VectorSize N> BMVPE_API vector_in_bounds(VecNf<N> v, VecNf<N> bounds) noexcept -> VecNi<N> { return reinterpret<VecNi<N>>(DirectX::XMVectorInBounds(v, bounds)); }
        template <VectorSize N> BMVPE_API vector_is_nan(VecNf<N> v) noexcept -> VecNi<N> { return reinterpret<VecNi<N>>(DirectX::XMVectorIsNaN(v)); }
        template <VectorSize N> BMVPE_API vector_is_inf(VecNf<N> v) noexcept -> VecNi<N> { return reinterpret<VecNi<N>>(DirectX::XMVectorIsInfinite(v)); }
        template <VectorSize N> BMVPE_API vector_bw_and(VecNi<N> v1, VecNi<N> v2) noexcept -> VecNi<N> { return reinterpret<VecNi<N>>(DirectX::XMVectorAndInt(v1, v2)); }
        template <VectorSize N> BMVPE_API vector_bw_andc(VecNi<N> v1, VecNi<N> v2) noexcept -> VecNi<N> { return reinterpret<VecNi<N>>(DirectX::XMVectorAndCInt(v1, v2)); }
        template <VectorSize N> BMVPE_API vector_bw_or(VecNi<N> v1, VecNi<N> v2) noexcept -> VecNi<N> { return reinterpret<VecNi<N>>(DirectX::XMVectorOrInt(v1, v2)); }
        template <VectorSize N> BMVPE_API vector_bw_nor(VecNi<N> v1, VecNi<N> v2) noexcept -> VecNi<N> { return reinterpret<VecNi<N>>(DirectX::XMVectorNorInt(v1, v2)); }
        template <VectorSize N> BMVPE_API vector_bw_xor(VecNi<N> v1, VecNi<N> v2) noexcept -> VecNi<N> { return reinterpret<VecNi<N>>(DirectX::XMVectorXorInt(v1, v2)); }

        BMVPE_API equal(Vec1f v1, Vec1f v2) noexcept -> bool { return DirectX::XMVectorGetIntX(vector_cmp_eq(v1, v2)); }
        BMVPE_API equal(Vec2f v1, Vec2f v2) noexcept -> bool { return DirectX::XMVector2Equal(v1, v2); }
        BMVPE_API equal(Vec3f v1, Vec3f v2) noexcept -> bool { return DirectX::XMVector3Equal(v1, v2); }
        BMVPE_API equal(Vec4f v1, Vec4f v2) noexcept -> bool { return DirectX::XMVector4Equal(v1, v2); }
        BMVPE_API equal(Vec1i v1, Vec1i v2) noexcept -> bool { return DirectX::XMVectorGetIntX(vector_cmp_eq(v1, v2)); }
        BMVPE_API equal(Vec2i v1, Vec2i v2) noexcept -> bool { return DirectX::XMVector2EqualInt(v1, v2); }
        BMVPE_API equal(Vec3i v1, Vec3i v2) noexcept -> bool { return DirectX::XMVector3EqualInt(v1, v2); }
        BMVPE_API equal(Vec4i v1, Vec4i v2) noexcept -> bool { return DirectX::XMVector4EqualInt(v1, v2); }

        BMVPE_API near_equal(Vec1f v1, Vec1f v2, Vec1f eps) noexcept -> bool { return DirectX::XMVectorGetIntX(vector_cmp_near_eq(v1, v2, eps)); }
        BMVPE_API near_equal(Vec2f v1, Vec2f v2, Vec2f eps) noexcept -> bool { return DirectX::XMVector2NearEqual(v1, v2, eps); }
        BMVPE_API near_equal(Vec3f v1, Vec3f v2, Vec3f eps) noexcept -> bool { return DirectX::XMVector3NearEqual(v1, v2, eps); }
        BMVPE_API near_equal(Vec4f v1, Vec4f v2, Vec4f eps) noexcept -> bool { return DirectX::XMVector4NearEqual(v1, v2, eps); }
        BMVPE_API near_equal(Vec1f v1, Vec1f v2, float eps) noexcept -> bool { return DirectX::XMVectorGetIntX(vector_cmp_near_eq(v1, v2, vec1f(eps))); }
        BMVPE_API near_equal(Vec2f v1, Vec2f v2, float eps) noexcept -> bool { return DirectX::XMVector2NearEqual(v1, v2, vec2f(eps)); }
        BMVPE_API near_equal(Vec3f v1, Vec3f v2, float eps) noexcept -> bool { return DirectX::XMVector3NearEqual(v1, v2, vec3f(eps)); }
        BMVPE_API near_equal(Vec4f v1, Vec4f v2, float eps) noexcept -> bool { return DirectX::XMVector4NearEqual(v1, v2, vec4f(eps)); }

        BMVPE_API not_equal(Vec1f v1, Vec1f v2) noexcept -> bool { return DirectX::XMVectorGetIntX(vector_cmp_neq(v1, v2)); }
        BMVPE_API not_equal(Vec2f v1, Vec2f v2) noexcept -> bool { return DirectX::XMVector2NotEqual(v1, v2); }
        BMVPE_API not_equal(Vec3f v1, Vec3f v2) noexcept -> bool { return DirectX::XMVector3NotEqual(v1, v2); }
        BMVPE_API not_equal(Vec4f v1, Vec4f v2) noexcept -> bool { return DirectX::XMVector4NotEqual(v1, v2); }
        BMVPE_API not_equal(Vec1i v1, Vec1i v2) noexcept -> bool { return DirectX::XMVectorGetIntX(vector_cmp_neq(v1, v2)); }
        BMVPE_API not_equal(Vec2i v1, Vec2i v2) noexcept -> bool { return DirectX::XMVector2NotEqualInt(v1, v2); }
        BMVPE_API not_equal(Vec3i v1, Vec3i v2) noexcept -> bool { return DirectX::XMVector3NotEqualInt(v1, v2); }
        BMVPE_API not_equal(Vec4i v1, Vec4i v2) noexcept -> bool { return DirectX::XMVector4NotEqualInt(v1, v2); }

        BMVPE_API in_bounds(Vec1f v, Vec1f bounds) noexcept -> bool { return DirectX::XMVectorGetIntX(vector_in_bounds(v, bounds)); }
        BMVPE_API in_bounds(Vec2f v, Vec2f bounds) noexcept -> bool { return DirectX::XMVector2InBounds(v, bounds); }
        BMVPE_API in_bounds(Vec3f v, Vec3f bounds) noexcept -> bool { return DirectX::XMVector3InBounds(v, bounds); }
        BMVPE_API in_bounds(Vec4f v, Vec4f bounds) noexcept -> bool { return DirectX::XMVector4InBounds(v, bounds); }

        BMVPE_API any_is_nan(Vec1f v) noexcept -> bool { return DirectX::XMVectorGetIntX(vector_is_nan(v)); }
        BMVPE_API any_is_nan(Vec2f v) noexcept -> bool { return DirectX::XMVector2IsNaN(v); }
        BMVPE_API any_is_nan(Vec3f v) noexcept -> bool { return DirectX::XMVector3IsNaN(v); }
        BMVPE_API any_is_nan(Vec4f v) noexcept -> bool { return DirectX::XMVector4IsNaN(v); }
        BMVPE_API any_is_inf(Vec1f v) noexcept -> bool { return DirectX::XMVectorGetIntX(vector_is_nan(v)); }
        BMVPE_API any_is_inf(Vec2f v) noexcept -> bool { return DirectX::XMVector2IsInfinite(v); }
        BMVPE_API any_is_inf(Vec3f v) noexcept -> bool { return DirectX::XMVector3IsInfinite(v); }
        BMVPE_API any_is_inf(Vec4f v) noexcept -> bool { return DirectX::XMVector4IsInfinite(v); }

        template <VectorSize N> BMVPE_API min(VecNf<N> v1, VecNf<N> v2) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorMin(v1, v2)); }
        template <VectorSize N> BMVPE_API min(VecNf<N> v1, float v2) noexcept -> VecNf<N> { return min(v1, broadcast<VecNf<N>>(v2)); }
        template <VectorSize N> BMVPE_API max(VecNf<N> v1, VecNf<N> v2) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorMax(v1, v2)); }
        template <VectorSize N> BMVPE_API max(VecNf<N> v1, float v2) noexcept -> VecNf<N> { return max(v1, broadcast<VecNf<N>>(v2)); }
        template <VectorSize N> BMVPE_API clamp(VecNf<N> v, VecNf<N> min, VecNf<N> max) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorClamp(v, min, max)); }
        template <VectorSize N> BMVPE_API clamp(VecNf<N> v, float min, float max) noexcept -> VecNf<N> { return clamp(v, broadcast<VecNf<N>>(min), broadcast<VecNf<N>>(max)); }
        template <VectorSize N> BMVPE_API round(VecNf<N> v) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorRound(v)); }
        template <VectorSize N> BMVPE_API truncate(VecNf<N> v) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorTruncate(v)); }
        template <VectorSize N> BMVPE_API floor(VecNf<N> v) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorFloor(v)); }
        template <VectorSize N> BMVPE_API ceiling(VecNf<N> v) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorCeiling(v)); }
        template <VectorSize N> BMVPE_API saturate(VecNf<N> v) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorSaturate(v)); }

        template <VectorSize N> BMVPE_API neg(VecNf<N> x) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorNegate(x)); }
        template <VectorSize N> BMVPE_API add(VecNf<N> x, VecNf<N> y) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorAdd(x, y)); }
        template <VectorSize N> BMVPE_API sub(VecNf<N> x, VecNf<N> y) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorSubtract(x, y)); }
        template <VectorSize N> BMVPE_API mul(VecNf<N> x, VecNf<N> y) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorMultiply(x, y)); }
        template <VectorSize N> BMVPE_API mul(VecNf<N> x, float y) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorScale(x, y)); }
        template <VectorSize N> BMVPE_API div(VecNf<N> x, VecNf<N> y) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorDivide(x, y)); }
        template <VectorSize N> BMVPE_API div(VecNf<N> x, float y) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorDivide(x, DirectX::XMVectorReplicate(y))); }
        template <VectorSize N> BMVPE_API mod(VecNf<N> x, VecNf<N> y) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorMod(x, y)); }
        template <VectorSize N> BMVPE_API mod(VecNf<N> x, float y) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorMod(x, DirectX::XMVectorReplicate(y))); }

        // x * y + z
        template <VectorSize N> BMVPE_API mul_add(VecNf<N> x, VecNf<N> y, VecNf<N> z) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorMultiplyAdd(x, y, z)); }

        // -x * y + z
        template <VectorSize N> BMVPE_API neg_mul_sub(VecNf<N> x, VecNf<N> y, VecNf<N> z) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorNegativeMultiplySubtract(x, y, z)); }

        // Mod in angle: Mod x into the range: for x ∈ [-Pi,Pi) and y ∈ [-2Pi,2Pi), result into [-Pi,Pi)
        template <VectorSize N> BMVPE_API add_angle(VecNf<N> x, VecNf<N> y) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorAddAngles(x, y)); }
        template <VectorSize N> BMVPE_API sub_angle(VecNf<N> x, VecNf<N> y) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorSubtractAngles(x, y)); }
        template <VectorSize N> BMVPE_API mod_angle(VecNf<N> x) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorModAngles(x)); }

        namespace est
        {
            // {1/x, 1/y, 1/z, 1/w}
            template <VectorSize N> BMVPE_API inverse(VecNf<N> x) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorReciprocalEst(x)); }

            // {sqrt(x), sqrt(y), sqrt(z), sqrt(w)}
            template <VectorSize N> BMVPE_API sqrt(VecNf<N> x) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorSqrtEst(x)); }

            // {1/sqrt(x), 1/sqrt(y), 1/sqrt(z), 1/sqrt(w)}
            template <VectorSize N> BMVPE_API inverse_sqrt(VecNf<N> x) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorReciprocalSqrtEst(x)); }
        }

        // {1/x,...}
        template <VectorSize N> BMVPE_API inverse(VecNf<N> x) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorReciprocal(x)); }
        // {sqrt(x),...}
        template <VectorSize N> BMVPE_API sqrt(VecNf<N> x) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorSqrt(x)); }
        // {1/sqrt(x),...}
        template <VectorSize N> BMVPE_API inverse_sqrt(VecNf<N> x) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorReciprocalSqrt(x)); }

        // {pow(2,x),...}
        template <VectorSize N> BMVPE_API exp_2(VecNf<N> x) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorExp2(x)); }
        // {pow(e,x),...}
        template <VectorSize N> BMVPE_API exp_e(VecNf<N> x) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorExpE(x)); }
        // {log_2(x),...}
        template <VectorSize N> BMVPE_API log_2(VecNf<N> x) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorLog2(x)); }
        // {ln(x),...}
        template <VectorSize N> BMVPE_API log_e(VecNf<N> x) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorLogE(x)); }
        // {pow(x,y),...}
        template <VectorSize N> BMVPE_API pow(VecNf<N> x, VecNf<N> y) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorPow(x, y)); }
        // {pow(x,y),...}
        template <VectorSize N> BMVPE_API pow(VecNf<N> x, float y) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorPow(x, DirectX::XMVectorReplicate(y))); }
        // {abs(x),...}
        template <VectorSize N> BMVPE_API abs(VecNf<N> x) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorAbs(x)); }

        template <VectorSize N> BMVPE_API sin(VecNf<N> x) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorSin(x)); }
        template <VectorSize N> BMVPE_API cos(VecNf<N> x) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorCos(x)); }
        template <VectorSize N> BMVPE_API tan(VecNf<N> x) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorTan(x)); }
        template <VectorSize N> BMVPE_API sinh(VecNf<N> x) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorSinH(x)); }
        template <VectorSize N> BMVPE_API cosh(VecNf<N> x) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorCosH(x)); }
        template <VectorSize N> BMVPE_API tanh(VecNf<N> x) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorTanH(x)); }
        template <VectorSize N> BMVPE_API asin(VecNf<N> x) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorASin(x)); }
        template <VectorSize N> BMVPE_API acos(VecNf<N> x) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorACos(x)); }
        template <VectorSize N> BMVPE_API atan(VecNf<N> x) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorATan(x)); }
        template <VectorSize N> BMVPE_API atan2(VecNf<N> y, VecNf<N> x) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorATan2(y, x)); }

        // returns [{sin(x),...}, {cos(x),...}]
        template <VectorSize N> BMVPE_API sin_cos(VecNf<N> x) noexcept -> std::array<VecNf<N>, 2>
        {
            DirectX::XMVECTOR s, c;
            DirectX::XMVectorSinCos(&s, &c, x);
            return {VecNf<N>(s), VecNf<N>(c)};
        }

        namespace est
        {
            template <VectorSize N> BMVPE_API sin(VecNf<N> x) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorSinEst(x)); }
            template <VectorSize N> BMVPE_API cos(VecNf<N> x) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorCosEst(x)); }
            template <VectorSize N> BMVPE_API tan(VecNf<N> x) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorTanEst(x)); }
            template <VectorSize N> BMVPE_API asin(VecNf<N> x) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorASinEst(x)); }
            template <VectorSize N> BMVPE_API acos(VecNf<N> x) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorACosEst(x)); }
            template <VectorSize N> BMVPE_API atan(VecNf<N> x) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorATanEst(x)); }
            template <VectorSize N> BMVPE_API atan2(VecNf<N> y, VecNf<N> x) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorATan2Est(y, x)); }

            // returns [{sin(x),...}, {cos(x),...}]
            template <VectorSize N> BMVPE_API sin_cos(VecNf<N> x) noexcept -> std::array<VecNf<N>, 2>
            {
                DirectX::XMVECTOR s, c;
                DirectX::XMVectorSinCosEst(&s, &c, x);
                return {reinterpret<VecNf<N>>(s), reinterpret<VecNf<N>>(c)};
            }
        }


        /// inner product (内積):
        BMVPE_API dot(Vec1f v1, Vec1f v2) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVectorAdd(v1, v2)); }
        BMVPE_API dot(Vec2f v1, Vec2f v2) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVector2Dot(v1, v2)); }
        BMVPE_API dot(Vec3f v1, Vec3f v2) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVector3Dot(v1, v2)); }
        BMVPE_API dot(Vec4f v1, Vec4f v2) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVector4Dot(v1, v2)); }

        /// outer product (外積):
        BMVPE_API cross(Vec1f v1, Vec1f v2) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVectorSubtract(v1, v2)); }
        BMVPE_API cross(Vec2f v1, Vec2f v2) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVector2Cross(v1, v2)); }
        BMVPE_API cross(Vec3f v1, Vec3f v2) noexcept -> Vec3f { return reinterpret<Vec3f>(DirectX::XMVector3Cross(v1, v2)); }
        BMVPE_API cross(Vec4f v1, Vec4f v2, Vec4f v3) noexcept -> Vec4f { return reinterpret<Vec4f>(DirectX::XMVector4Cross(v1, v2, v3)); }


        // squared length of vector (長さの二乗):
        //      length^2 = dot(v, v)
        BMVPE_API length_squared(Vec1f v) noexcept -> float { return DirectX::XMVectorGetX(v) * DirectX::XMVectorGetX(v); }
        BMVPE_API length_squared(Vec2f v) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVector2LengthSq(v)); }
        BMVPE_API length_squared(Vec3f v) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVector3LengthSq(v)); }
        BMVPE_API length_squared(Vec4f v) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVector4LengthSq(v)); }

        namespace est
        {
            // length of vector (長さ):
            //      length = sqrt(dot(v, v))
            BMVPE_API length(Vec1f v) noexcept -> float { return DirectX::XMVectorGetX(v); }
            BMVPE_API length(Vec2f v) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVector2LengthEst(v)); }
            BMVPE_API length(Vec3f v) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVector3LengthEst(v)); }
            BMVPE_API length(Vec4f v) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVector4LengthEst(v)); }
        }

        // length of vector (長さ):
        //      length = sqrt(dot(v, v))
        BMVPE_API length(Vec1f v) noexcept -> float { return DirectX::XMVectorGetX(v); }
        BMVPE_API length(Vec2f v) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVector2Length(v)); }
        BMVPE_API length(Vec3f v) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVector3Length(v)); }
        BMVPE_API length(Vec4f v) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVector4Length(v)); }

        namespace est
        {
            // inverse of length (長さの逆数):
            //      1.0 / length
            BMVPE_API inverse_length(Vec1f v) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVectorReciprocalEst(v)); }
            BMVPE_API inverse_length(Vec2f v) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVector2ReciprocalLengthEst(v)); }
            BMVPE_API inverse_length(Vec3f v) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVector3ReciprocalLengthEst(v)); }
            BMVPE_API inverse_length(Vec4f v) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVector4ReciprocalLengthEst(v)); }
        }

        // inverse of length (長さの逆数):
        //      1.0 / length
        BMVPE_API inverse_length(Vec1f v) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVectorReciprocal(v)); }
        BMVPE_API inverse_length(Vec2f v) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVector2ReciprocalLength(v)); }
        BMVPE_API inverse_length(Vec3f v) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVector3ReciprocalLength(v)); }
        BMVPE_API inverse_length(Vec4f v) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVector4ReciprocalLength(v)); }

        namespace est
        {
            // Normalized vector:
            //      direction(v) = v / length = 1
            BMVPE_API normalize(Vec1f v) noexcept -> Vec1f { return reinterpret<Vec1f>(DirectX::XMVectorMultiply(v, DirectX::XMVectorReciprocalEst(v))); }
            BMVPE_API normalize(Vec2f v) noexcept -> Vec2f { return reinterpret<Vec2f>(DirectX::XMVector2NormalizeEst(v)); }
            BMVPE_API normalize(Vec3f v) noexcept -> Vec3f { return reinterpret<Vec3f>(DirectX::XMVector3NormalizeEst(v)); }
            BMVPE_API normalize(Vec4f v) noexcept -> Vec4f { return reinterpret<Vec4f>(DirectX::XMVector4NormalizeEst(v)); }
        }

        // Normalized vector:
        //      direction(v) = v / length = 1
        BMVPE_API normalize(Vec1f v) noexcept -> Vec1f { return reinterpret<Vec1f>(DirectX::XMVectorScale(v, inverse_length(v))); }
        BMVPE_API normalize(Vec2f v) noexcept -> Vec2f { return reinterpret<Vec2f>(DirectX::XMVector2Normalize(v)); }
        BMVPE_API normalize(Vec3f v) noexcept -> Vec3f { return reinterpret<Vec3f>(DirectX::XMVector3Normalize(v)); }
        BMVPE_API normalize(Vec4f v) noexcept -> Vec4f { return reinterpret<Vec4f>(DirectX::XMVector4Normalize(v)); }

        // Clamp by length of vector
        BMVPE_API clamp_by_length(Vec1f v, float min, float max) noexcept -> Vec1f { return reinterpret<Vec1f>(DirectX::XMVectorClamp(v, DirectX::XMVectorReplicate(min), DirectX::XMVectorReplicate(max))); }
        BMVPE_API clamp_by_length(Vec2f v, float min, float max) noexcept -> Vec2f { return reinterpret<Vec2f>(DirectX::XMVector2ClampLength(v, min, max)); }
        BMVPE_API clamp_by_length(Vec3f v, float min, float max) noexcept -> Vec3f { return reinterpret<Vec3f>(DirectX::XMVector3ClampLength(v, min, max)); }
        BMVPE_API clamp_by_length(Vec4f v, float min, float max) noexcept -> Vec4f { return reinterpret<Vec4f>(DirectX::XMVector4ClampLength(v, min, max)); }

        inline namespace interpolation
        {
            // Linear interpolation:
            //      t * y + (1-t) * x = x + t(y-x)
            template <VectorSize N> BMVPE_API lerp(VecNf<N> x, VecNf<N> y, float t) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorLerp(x, y, t)); }
            template <VectorSize N> BMVPE_API lerp(VecNf<N> x, VecNf<N> y, VecNf<N> t) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorLerpV(x, y, t)); }
            template Vec1f lerp(Vec1f, Vec1f, float) noexcept; // test
            template Vec2f lerp(Vec2f, Vec2f, float) noexcept; // test
            template Vec3f lerp(Vec3f, Vec3f, float) noexcept; // test
            template Vec4f lerp(Vec4f, Vec4f, float) noexcept; // test
            template Vec1f lerp(Vec1f, Vec1f, Vec1f) noexcept; // test
            template Vec2f lerp(Vec2f, Vec2f, Vec2f) noexcept; // test
            template Vec3f lerp(Vec3f, Vec3f, Vec3f) noexcept; // test
            template Vec4f lerp(Vec4f, Vec4f, Vec4f) noexcept; // test

            // AMD's SmoothStep interpolation:
            //      t = clamp(t, 0.0f, 1.0f), t = t * t * (3 - 2 * t),  t * x + (1-t) * y
            template <VectorSize N> BMVPE_API smooth_step(VecNf<N> x, VecNf<N> y, float t) noexcept -> VecNf<N> { return t = scalar::clamp(t, 0.0f, 1.0f), t = scalar::mul(scalar::mul(t, t), scalar::sub(3.0f, scalar::mul(t, 2.0f))), vector::lerp(x, y, t); }
            template <VectorSize N> BMVPE_API smooth_step(VecNf<N> x, VecNf<N> y, VecNf<N> t) noexcept -> VecNf<N> { return t = vector::clamp(t, 0.0f, 1.0f), t = vector::mul(vector::mul(t, t), vector::sub(broadcast<VecNf<N>>(3.0f), vector::mul(t, 2.0f))), vector::lerp(x, y, t); }
            template Vec1f smooth_step(Vec1f, Vec1f, float) noexcept; // test
            template Vec2f smooth_step(Vec2f, Vec2f, float) noexcept; // test
            template Vec3f smooth_step(Vec3f, Vec3f, float) noexcept; // test
            template Vec4f smooth_step(Vec4f, Vec4f, float) noexcept; // test
            template Vec1f smooth_step(Vec1f, Vec1f, Vec1f) noexcept; // test
            template Vec2f smooth_step(Vec2f, Vec2f, Vec2f) noexcept; // test
            template Vec3f smooth_step(Vec3f, Vec3f, Vec3f) noexcept; // test
            template Vec4f smooth_step(Vec4f, Vec4f, Vec4f) noexcept; // test

            // Hermite interpolation:
            //      (2 * t^3 - 3 * t^2 + 1) * v0 +
            //      (t^3 - 2 * t^2 + t) * t0 +
            //      (-2 * t^3 + 3 * t^2) * v1 +
            //      (t^3 - t^2) * t1
            template <VectorSize N> BMVPE_API hermite(VecNf<N> v0, VecNf<N> t0, VecNf<N> v1, VecNf<N> t1, float t) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorHermite(v0, t0, v1, t1, t)); }
            template <VectorSize N> BMVPE_API hermite(VecNf<N> v0, VecNf<N> t0, VecNf<N> v1, VecNf<N> t1, VecNf<N> t) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorHermiteV(v0, t0, v1, t1, t)); }
            template Vec1f hermite(Vec1f, Vec1f, Vec1f, Vec1f, float) noexcept; // test
            template Vec2f hermite(Vec2f, Vec2f, Vec2f, Vec2f, float) noexcept; // test
            template Vec3f hermite(Vec3f, Vec3f, Vec3f, Vec3f, float) noexcept; // test
            template Vec4f hermite(Vec4f, Vec4f, Vec4f, Vec4f, float) noexcept; // test
            template Vec1f hermite(Vec1f, Vec1f, Vec1f, Vec1f, Vec1f) noexcept; // test
            template Vec2f hermite(Vec2f, Vec2f, Vec2f, Vec2f, Vec2f) noexcept; // test
            template Vec3f hermite(Vec3f, Vec3f, Vec3f, Vec3f, Vec3f) noexcept; // test
            template Vec4f hermite(Vec4f, Vec4f, Vec4f, Vec4f, Vec4f) noexcept; // test

            // Catmull–Rom interpolation
            //      ((-t^3 + 2 * t^2 - t) * Position0 +
            //      (3 * t^3 - 5 * t^2 + 2) * Position1 +
            //      (-3 * t^3 + 4 * t^2 + t) * Position2 +
            //      (t^3 - t^2) * Position3) * 0.5
            template <VectorSize N> BMVPE_API catmull_rom(VecNf<N> v0, VecNf<N> v1, VecNf<N> v2, VecNf<N> v3, float t) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorCatmullRom(v0, v1, v2, v3, t)); }
            template <VectorSize N> BMVPE_API catmull_rom(VecNf<N> v0, VecNf<N> v1, VecNf<N> v2, VecNf<N> v3, VecNf<N> t) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorCatmullRomV(v0, v1, v2, v3, t)); }
            template Vec1f catmull_rom(Vec1f, Vec1f, Vec1f, Vec1f, float) noexcept;
            template Vec2f catmull_rom(Vec2f, Vec2f, Vec2f, Vec2f, float) noexcept;
            template Vec3f catmull_rom(Vec3f, Vec3f, Vec3f, Vec3f, float) noexcept;
            template Vec4f catmull_rom(Vec4f, Vec4f, Vec4f, Vec4f, float) noexcept;
            template Vec1f catmull_rom(Vec1f, Vec1f, Vec1f, Vec1f, Vec1f) noexcept;
            template Vec2f catmull_rom(Vec2f, Vec2f, Vec2f, Vec2f, Vec2f) noexcept;
            template Vec3f catmull_rom(Vec3f, Vec3f, Vec3f, Vec3f, Vec3f) noexcept;
            template Vec4f catmull_rom(Vec4f, Vec4f, Vec4f, Vec4f, Vec4f) noexcept;

            /// Barycentric interpolation
            //      v0 + t1 * (v1 - v0) + t2 * (v2 - v0)
            template <VectorSize N> BMVPE_API barycentric(VecNf<N> v0, VecNf<N> v1, VecNf<N> v2, float t1, float t2) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorBaryCentric(v0, v1, v2, t1, t2)); }
            template <VectorSize N> BMVPE_API barycentric(VecNf<N> v0, VecNf<N> v1, VecNf<N> v2, VecNf<N> t1, VecNf<N> t2) noexcept -> VecNf<N> { return reinterpret<VecNf<N>>(DirectX::XMVectorBaryCentricV(v0, v1, v2, t1, t2)); }
            template Vec1f barycentric(Vec1f, Vec1f, Vec1f, float, float) noexcept;
            template Vec2f barycentric(Vec2f, Vec2f, Vec2f, float, float) noexcept;
            template Vec3f barycentric(Vec3f, Vec3f, Vec3f, float, float) noexcept;
            template Vec4f barycentric(Vec4f, Vec4f, Vec4f, float, float) noexcept;
            template Vec1f barycentric(Vec1f, Vec1f, Vec1f, Vec1f, Vec1f) noexcept;
            template Vec2f barycentric(Vec2f, Vec2f, Vec2f, Vec2f, Vec2f) noexcept;
            template Vec3f barycentric(Vec3f, Vec3f, Vec3f, Vec3f, Vec3f) noexcept;
            template Vec4f barycentric(Vec4f, Vec4f, Vec4f, Vec4f, Vec4f) noexcept;
        }


        // Orthogonal: 直交ベクトル
        BMVPE_API orthogonal(Vec2f v) noexcept -> Vec2f { return reinterpret<Vec2f>(DirectX::XMVector2Orthogonal(v)); }
        BMVPE_API orthogonal(Vec3f v) noexcept -> Vec3f { return reinterpret<Vec3f>(DirectX::XMVector3Orthogonal(v)); }
        BMVPE_API orthogonal(Vec4f v) noexcept -> Vec4f { return reinterpret<Vec4f>(DirectX::XMVector4Orthogonal(v)); }

        // AngleBetweenNormals(成す角)
        BMVPE_API angle_between_normals(Vec2f n1, Vec2f n2) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVector2AngleBetweenNormals(n1, n2)); }
        BMVPE_API angle_between_normals(Vec3f n1, Vec3f n2) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVector3AngleBetweenNormals(n1, n2)); }
        BMVPE_API angle_between_normals(Vec4f n1, Vec4f n2) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVector4AngleBetweenNormals(n1, n2)); }

        namespace est
        {
            // AngleBetweenNormals(成す角)
            BMVPE_API angle_between_normals(Vec2f n1, Vec2f n2) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVector2AngleBetweenNormalsEst(n1, n2)); }
            BMVPE_API angle_between_normals(Vec3f n1, Vec3f n2) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVector3AngleBetweenNormalsEst(n1, n2)); }
            BMVPE_API angle_between_normals(Vec4f n1, Vec4f n2) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVector4AngleBetweenNormalsEst(n1, n2)); }
        }

        // AngleBetweenVectors(成す角)
        BMVPE_API angle_between_vectors(Vec2f v1, Vec2f v2) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVector2AngleBetweenVectors(v1, v2)); }
        BMVPE_API angle_between_vectors(Vec3f v1, Vec3f v2) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVector3AngleBetweenVectors(v1, v2)); }
        BMVPE_API angle_between_vectors(Vec4f v1, Vec4f v2) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMVector4AngleBetweenVectors(v1, v2)); }

        inline namespace geometry
        {
            // Reflection vector (反射ベクトル):
            //      incident - (2 * dot(incident, normal)) * normal
            BMVPE_API reflect(Vec2f incident, Vec2f normal) noexcept -> Vec2f { return reinterpret<Vec2f>(DirectX::XMVector2Reflect(incident, normal)); }
            BMVPE_API reflect(Vec3f incident, Vec3f normal) noexcept -> Vec3f { return reinterpret<Vec3f>(DirectX::XMVector3Reflect(incident, normal)); }
            BMVPE_API reflect(Vec4f incident, Vec4f normal) noexcept -> Vec4f { return reinterpret<Vec4f>(DirectX::XMVector4Reflect(incident, normal)); }

            // Refract vector (屈折ベクトル):
            //        index * incident
            //        - Normal * (index * dot(incident, normal)
            //        + sqrt(1 - index * index * (1 - dot(incident, normal) * dot(incident, normal))))
            BMVPE_API refract(Vec2f incident, Vec2f normal, float index) noexcept -> Vec2f { return reinterpret<Vec2f>(DirectX::XMVector2Refract(incident, normal, index)); }
            BMVPE_API refract(Vec2f incident, Vec2f normal, Vec2f index) noexcept -> Vec2f { return reinterpret<Vec2f>(DirectX::XMVector2RefractV(incident, normal, index)); }
            BMVPE_API refract(Vec3f incident, Vec3f normal, float index) noexcept -> Vec3f { return reinterpret<Vec3f>(DirectX::XMVector3Refract(incident, normal, index)); }
            BMVPE_API refract(Vec3f incident, Vec3f normal, Vec3f index) noexcept -> Vec3f { return reinterpret<Vec3f>(DirectX::XMVector3RefractV(incident, normal, index)); }
            BMVPE_API refract(Vec4f incident, Vec4f normal, float index) noexcept -> Vec4f { return reinterpret<Vec4f>(DirectX::XMVector4Refract(incident, normal, index)); }
            BMVPE_API refract(Vec4f incident, Vec4f normal, Vec4f index) noexcept -> Vec4f { return reinterpret<Vec4f>(DirectX::XMVector4RefractV(incident, normal, index)); }

            // Distance of line and point.
            BMVPE_API line_point_distance(Vec2f linePoint1, Vec2f linePoint2, Vec2f point) noexcept -> float
            {
                return DirectX::XMVectorGetX(DirectX::XMVector2LinePointDistance(linePoint1, linePoint2, point));
            }

            // Distance of line and point.
            BMVPE_API line_point_distance(Vec3f linePoint1, Vec3f linePoint2, Vec3f point) noexcept -> float
            {
                return DirectX::XMVectorGetX(DirectX::XMVector3LinePointDistance(linePoint1, linePoint2, point));
            }

            /// Intersection point of 2 lines.
            BMVPE_API intersection_point(Vec2f line1Point1, Vec2f line1Point2, Vec2f line2Point1, Vec2f line2Point2) noexcept -> Vec2f
            {
                return reinterpret<Vec2f>(DirectX::XMVector2IntersectLine(line1Point1, line1Point2, line2Point1, line2Point2));
            }

            // ComponentsFromNormal.
            BMVPE_API components_from_normal(Vec3f v, Vec3f normal, _Out_ Vec3f* parallel, _Out_ Vec3f* perpendicular) noexcept -> void
            {
                DirectX::XMVECTOR para, per;
                DirectX::XMVector3ComponentsFromNormal(&para, &per, v, normal);
                *parallel = reinterpret<Vec3f>(para);
                *perpendicular = reinterpret<Vec3f>(per);
            }
        }
    }

    inline namespace matrix
    {
        /// Immutable matrix type
        template <class TScalar, VectorSize TRows, VectorSize TCols>
        class MatrixType
        {
            DirectX::XMMATRIX m;
            MatrixType(DirectX::XMMATRIX m) noexcept : m(m) { ; }
            template <class TVectorType> friend auto XM_CALLCONV reinterpret(DirectX::XMMATRIX) noexcept -> TVectorType;

        public:
            using Scalar = TScalar;
            static inline constexpr VectorSize Rows = TRows;
            static inline constexpr VectorSize Cols = TCols;

            MatrixType() = default;
            operator DirectX::FXMMATRIX() const noexcept { return m; }
            explicit operator DirectX::CXMMATRIX() const noexcept { return m; }
            Vec4f operator [](size_t index) const noexcept { return reinterpret<Vec4f>(m.r[index]); }
        };

        using vector::reinterpret;
        template <class TMatrixType> BMVPE_API reinterpret(DirectX::XMMATRIX m) noexcept -> TMatrixType { return TMatrixType{m}; }
        template <class TMatrixType> BMVPE_API zero() noexcept -> TMatrixType { return reinterpret<TMatrixType>(DirectX::XMMATRIX{vector::zero<Vec4f>(), vector::zero<Vec4f>(), vector::zero<Vec4f>(), vector::zero<Vec4f>()}); }

        using Mat4x4 = MatrixType<float, 4, 4>;

        BMVPE_API any_is_nan(const Mat4x4& m) noexcept -> bool { return DirectX::XMMatrixIsNaN(m); }
        BMVPE_API any_is_inf(const Mat4x4& m) noexcept -> bool { return DirectX::XMMatrixIsInfinite(m); }
        BMVPE_API is_identity(const Mat4x4& m) noexcept -> bool { return DirectX::XMMatrixIsIdentity(m); }

        BMVPE_API neg(const Mat4x4& m) noexcept -> Mat4x4 { return reinterpret<Mat4x4>(-static_cast<DirectX::CXMMATRIX>(m)); }
        BMVPE_API add(const Mat4x4& m, const Mat4x4& n) noexcept -> Mat4x4 { return reinterpret<Mat4x4>(static_cast<DirectX::CXMMATRIX>(m) + n); }
        BMVPE_API sub(const Mat4x4& m, const Mat4x4& n) noexcept -> Mat4x4 { return reinterpret<Mat4x4>(static_cast<DirectX::CXMMATRIX>(m) + n); }
        BMVPE_API mul(const Mat4x4& m, const Mat4x4& n) noexcept -> Mat4x4 { return reinterpret<Mat4x4>(static_cast<DirectX::CXMMATRIX>(m) * n); }
        BMVPE_API mul(const Mat4x4& m, float n) noexcept -> Mat4x4 { return reinterpret<Mat4x4>(static_cast<DirectX::CXMMATRIX>(m) * n); }
        BMVPE_API div(const Mat4x4& m, float n) noexcept -> Mat4x4 { return reinterpret<Mat4x4>(static_cast<DirectX::CXMMATRIX>(m) / n); }

        /// transpose(m) (転置行列)
        BMVPE_API transpose(const Mat4x4& m) noexcept -> Mat4x4 { return reinterpret<Mat4x4>(DirectX::XMMatrixTranspose(m)); }

        /// inverse:m^-1 (逆行列)
        BMVPE_API inverse(const Mat4x4& m) noexcept -> Mat4x4 { return reinterpret<Mat4x4>(DirectX::XMMatrixInverse(nullptr, m)); }

        /// determinant (行列式)
        BMVPE_API determinant(const Mat4x4& m) noexcept -> float { return DirectX::XMVectorGetX(DirectX::XMMatrixDeterminant(m)); }

        /// transpose(mul(m,n))
        BMVPE_API mul_transpose(const Mat4x4& m, const Mat4x4& n) noexcept -> Mat4x4 { return reinterpret<Mat4x4>(DirectX::XMMatrixMultiplyTranspose(m, static_cast<DirectX::CXMMATRIX>(n))); }

        /// 拡縮成分・回転成分・平行移動成分に分解。成功すれば true
        BMVPE_API decompose(
            const Mat4x4& m,
            _Out_ Vec3f* scaling,
            _Out_ Vec4f* rotationQuaternion,
            _Out_ Vec3f* translation) noexcept -> bool
        {
            DirectX::XMVECTOR s, r, t;

            if (bool ret = DirectX::XMMatrixDecompose(&s, &r, &t, m))
            {
                *scaling = reinterpret<Vec3f>(s);
                *rotationQuaternion = reinterpret<Vec4f>(s);
                *translation = reinterpret<Vec3f>(s);
                return ret;
            }
            return false;
        }

        BMVPE_API mat4x4(Vec4f row0, Vec4f row1, Vec4f row2, Vec4f row3) noexcept -> Mat4x4
        {
            return reinterpret<Mat4x4>(DirectX::XMMATRIX{row0, row1, row2, row3});
        }

        BMVPE_API mat4x4(
            float m00, float m01, float m02, float m03,
            float m10, float m11, float m12, float m13,
            float m20, float m21, float m22, float m23,
            float m30, float m31, float m32, float m33) noexcept -> Mat4x4
        {
            return reinterpret<Mat4x4>(
                DirectX::XMMatrixSet(m00, m01, m02, m03,
                                     m10, m11, m12, m13,
                                     m20, m21, m22, m23,
                                     m30, m31, m32, m33));
        }


        namespace mats
        {
            BMVPE_API zero() noexcept -> Mat4x4 { return matrix::zero<Mat4x4>(); }
            BMVPE_API identity() noexcept -> Mat4x4 { return reinterpret<Mat4x4>(DirectX::XMMatrixIdentity()); }

            BMVPE_API translation(float x, float y, float z) noexcept -> Mat4x4 { return reinterpret<Mat4x4>(DirectX::XMMatrixTranslation(x, y, z)); }
            BMVPE_API translation(Vec3f xyz) noexcept -> Mat4x4 { return reinterpret<Mat4x4>(DirectX::XMMatrixTranslationFromVector(xyz)); }
            BMVPE_API scaling(float x, float y, float z) noexcept -> Mat4x4 { return reinterpret<Mat4x4>(DirectX::XMMatrixScaling(x, y, z)); }
            BMVPE_API scaling(Vec3f xyz) noexcept -> Mat4x4 { return reinterpret<Mat4x4>(DirectX::XMMatrixScalingFromVector(xyz)); }
            BMVPE_API rotation_x_axis(float angle) noexcept -> Mat4x4 { return reinterpret<Mat4x4>(DirectX::XMMatrixRotationX(angle)); }
            BMVPE_API rotation_y_axis(float angle) noexcept -> Mat4x4 { return reinterpret<Mat4x4>(DirectX::XMMatrixRotationY(angle)); }
            BMVPE_API rotation_z_axis(float angle) noexcept -> Mat4x4 { return reinterpret<Mat4x4>(DirectX::XMMatrixRotationZ(angle)); }
            BMVPE_API rotation_roll_pitch_yaw(float pitch, float yaw, float roll) noexcept -> Mat4x4 { return reinterpret<Mat4x4>(DirectX::XMMatrixRotationRollPitchYaw(pitch, yaw, roll)); }
            BMVPE_API rotation_roll_pitch_yaw(Vec3f pitchYawRoll) noexcept -> Mat4x4 { return reinterpret<Mat4x4>(DirectX::XMMatrixRotationRollPitchYawFromVector(pitchYawRoll)); }
            BMVPE_API rotation_axis_normalized(Vec3f normalAxis, float angle) noexcept -> Mat4x4 { return reinterpret<Mat4x4>(DirectX::XMMatrixRotationNormal(normalAxis, angle)); }
            BMVPE_API rotation_axis(Vec3f axis, float angle) noexcept -> Mat4x4 { return reinterpret<Mat4x4>(DirectX::XMMatrixRotationAxis(axis, angle)); }
            BMVPE_API rotation_quaternion(Vec4f quaternion) noexcept -> Mat4x4 { return reinterpret<Mat4x4>(DirectX::XMMatrixRotationQuaternion(quaternion)); }

            /// 2D変換行列を作る
            /// M = Inverse(MScalingOrigin) * Transpose(MScalingOrientation) * MScaling * MScalingOrientation *
            ///         MScalingOrigin * Inverse(MRotationOrigin) * MRotation * MRotationOrigin * MTranslation;
            BMVPE_API transformation_2d(
                Vec3f scalingOrigin, float scalingOrientation, Vec3f scaling,
                Vec3f rotationOrigin, float rotation,
                Vec3f translation) noexcept -> Mat4x4
            {
                return reinterpret<Mat4x4>(
                    DirectX::XMMatrixTransformation2D(
                        scalingOrigin, scalingOrientation, scaling,
                        rotationOrigin, rotation,
                        translation));
            }

            /// 3D変換行列を作る
            /// M = Inverse(MScalingOrigin) * Transpose(MScalingOrientation) * MScaling * MScalingOrientation *
            ///         MScalingOrigin * Inverse(MRotationOrigin) * MRotation * MRotationOrigin * MTranslation;
            BMVPE_API transformation(
                Vec3f scalingOrigin, Vec4f scalingOrientationQuaternion, Vec3f scaling,
                Vec3f rotationOrigin, Vec4f rotationQuaternion, Vec3f translation) -> Mat4x4
            {
                return reinterpret<Mat4x4>(
                    DirectX::XMMatrixTransformation(
                        scalingOrigin, scalingOrientationQuaternion, scaling,
                        rotationOrigin, rotationQuaternion,
                        translation));
            }

            /// 2D変換行列を作る
            /// M = MScaling * Inverse(MRotationOrigin) * MRotation * MRotationOrigin * MTranslation;
            BMVPE_API affine_transformation_2d(
                Vec3f scaling,
                Vec3f rotationOrigin, float rotation,
                Vec3f translation) -> Mat4x4
            {
                return reinterpret<Mat4x4>(
                    DirectX::XMMatrixAffineTransformation2D(
                        scaling,
                        rotationOrigin, rotation,
                        translation));
            }

            /// 3D変換行列を作る
            /// M = MScaling * Inverse(MRotationOrigin) * MRotation * MRotationOrigin * MTranslation;
            BMVPE_API affine_transformation(
                Vec3f scaling,
                Vec3f rotationOrigin, Vec4f rotationQuaternion,
                Vec3f translation) noexcept -> Mat4x4
            {
                return reinterpret<Mat4x4>(
                    DirectX::XMMatrixAffineTransformation(
                        scaling,
                        rotationOrigin, rotationQuaternion,
                        translation));
            }

            /// 2Dの[拡縮→回転→平行移動]変換行列を作る。
            BMVPE_API simple_transformation_2d(Vec2f scale, float rollAngle, Vec2f translate) noexcept -> Mat4x4
            {
                auto [sin, cos] = sin_cos(rollAngle);
                DirectX::XMMATRIX m{};
                m.r[0] = DirectX::XMVectorMultiply(DirectX::XMVectorSet(cos, sin, 0.0f, 0.0f), DirectX::XMVectorSplatX(scale));
                m.r[1] = DirectX::XMVectorMultiply(DirectX::XMVectorSet(-sin, cos, 0.0f, 0.0f), DirectX::XMVectorSplatY(scale));
                m.r[2] = DirectX::g_XMIdentityR2;
                m.r[3] = DirectX::XMVectorSelect(DirectX::g_XMIdentityR3, translate, DirectX::g_XMSelect1100);
                return reinterpret<Mat4x4>(m);

                // _XM_NO_INTRINSICS_
                //Matrix m = Identity;
                //double cosRoll = cos(rollAngle), sinRoll = sin(rollAngle);
                //m.M11 = (float)(cosRoll * scaleX);
                //m.M12 = (float)(sinRoll * scaleX);
                //m.M21 = (float)(-sinRoll * scaleY);
                //m.M22 = (float)(cosRoll * scaleY);
                //m.M33 = 1.0f;
                //m.M41 = (float)transX;
                //m.M42 = (float)transY;
                //m.M44 = 1.0f;
            }

            /// 3Dの[拡縮→回転→平行移動]変換行列を作る。
            BMVPE_API simple_transformation(Vec3f scale, Vec3f pitchYawRollAngle, Vec3f translate) noexcept -> Mat4x4
            {
                auto m = DirectX::XMMatrixRotationRollPitchYawFromVector(pitchYawRollAngle);
                m.r[0] = DirectX::XMVectorMultiply(m.r[0], DirectX::XMVectorSplatX(scale));
                m.r[1] = DirectX::XMVectorMultiply(m.r[1], DirectX::XMVectorSplatY(scale));
                m.r[2] = DirectX::XMVectorMultiply(m.r[2], DirectX::XMVectorSplatZ(scale));
                m.r[3] = DirectX::XMVectorSelect(DirectX::g_XMIdentityR3, translate, DirectX::g_XMSelect1110);
                return reinterpret<Mat4x4>(m);

                // _XM_NO_INTRINSICS_
                //double cosYaw = cos(yawRotate), sinYaw = sin(yawRotate);
                //double cosPitch = cos(pitchRotate), sinPitch = sin(pitchRotate);
                //double cosRoll = cos(rollRotate), sinRoll = sin(rollRotate);
                //Matrix m;
                //m.M11 = (float)(xScale * cosRoll * cosYaw + xScale * sinRoll * sinPitch * sinYaw);
                //m.M12 = (float)(xScale * sinRoll * cosPitch);
                //m.M13 = (float)(xScale * cosRoll * -sinYaw + xScale * sinRoll * sinPitch * cosYaw);
                //m.M14 = 0.0f;
                //m.M21 = (float)(yScale * -sinRoll * cosYaw + yScale * cosRoll * sinPitch * sinYaw);
                //m.M22 = (float)(yScale * cosRoll * cosPitch);
                //m.M23 = (float)(yScale * -sinRoll * -sinYaw + yScale * cosRoll * sinPitch * cosYaw);
                //m.M24 = 0.0f;
                //m.M31 = (float)(zScale * cosPitch * sinYaw);
                //m.M32 = (float)(zScale * -sinPitch);
                //m.M33 = (float)(zScale * cosPitch * cosYaw);
                //m.M34 = 0.0f;
                //m.M41 = (float)xTranslate;
                //m.M42 = (float)yTranslate;
                //m.M43 = (float)zTranslate;
                //m.M44 = 1f;
            }

            /// Builds a transformation matrix designed to reflect vectors through a given plane.
            /// @param reflectionPlane: Plane to reflect through.
            /// @returns the transformation matrix.
            BMVPE_API reflection(Vec4f reflectionPlane) noexcept -> Mat4x4 { return reinterpret<Mat4x4>(DirectX::XMMatrixReflect(reflectionPlane)); }

            /// Builds a transformation matrix that flattens geometry into a plane.
            /// @param shadowPlane: Reference plane.
            /// @param lightPosition: 4D vector describing the light's position. If the light's w-component is 0.0f, the ray from the origin to the light represents a directional light. If it is 1.0f, the light is a point light.
            /// @returns the transformation matrix that flattens the geometry into the plane ShadowPlane.
            BMVPE_API shaddow(Vec4f shadowPlane, Vec3f lightPosition) noexcept -> Mat4x4 { return reinterpret<Mat4x4>(DirectX::XMMatrixShadow(shadowPlane, lightPosition)); }

            namespace left_hand
            {
                BMVPE_API look_at(Vec3f eyePosition, Vec3f focusPosition, Vec3f upDirection) noexcept -> Mat4x4 { return reinterpret<Mat4x4>(DirectX::XMMatrixLookAtLH(eyePosition, focusPosition, upDirection)); }
                BMVPE_API look_to(Vec3f eyePosition, Vec3f eyeDirection, Vec3f upDirection) noexcept -> Mat4x4 { return reinterpret<Mat4x4>(DirectX::XMMatrixLookToLH(eyePosition, eyeDirection, upDirection)); }
                BMVPE_API perspective(float viewWidth, float viewHeight, float nearZ, float farZ) noexcept -> Mat4x4 { return reinterpret<Mat4x4>(DirectX::XMMatrixPerspectiveLH(viewWidth, viewHeight, nearZ, farZ)); }
                BMVPE_API perspective_fov(float fovAngleY, float aspectRatio, float nearZ, float farZ) noexcept -> Mat4x4 { return reinterpret<Mat4x4>(DirectX::XMMatrixPerspectiveFovLH(fovAngleY, aspectRatio, nearZ, farZ)); }
                BMVPE_API perspective_off_center(float viewLeft, float viewRight, float viewBottom, float viewTop, float nearZ, float farZ) noexcept -> Mat4x4 { return reinterpret<Mat4x4>(DirectX::XMMatrixPerspectiveOffCenterLH(viewLeft, viewRight, viewBottom, viewTop, nearZ, farZ)); }
                BMVPE_API orthographic(float viewWidth, float viewHeight, float nearZ, float farZ) noexcept -> Mat4x4 { return reinterpret<Mat4x4>(DirectX::XMMatrixOrthographicLH(viewWidth, viewHeight, nearZ, farZ)); }
                BMVPE_API orthographic_off_center(float viewLeft, float viewRight, float viewBottom, float viewTop, float nearZ, float farZ) noexcept -> Mat4x4 { return reinterpret<Mat4x4>(DirectX::XMMatrixOrthographicOffCenterLH(viewLeft, viewRight, viewBottom, viewTop, nearZ, farZ)); }
            }

            namespace right_hand
            {
                BMVPE_API look_at(Vec3f eyePosition, Vec3f focusPosition, Vec3f upDirection) noexcept -> Mat4x4 { return reinterpret<Mat4x4>(DirectX::XMMatrixLookAtRH(eyePosition, focusPosition, upDirection)); }
                BMVPE_API look_to(Vec3f eyePosition, Vec3f eyeDirection, Vec3f upDirection) noexcept -> Mat4x4 { return reinterpret<Mat4x4>(DirectX::XMMatrixLookToRH(eyePosition, eyeDirection, upDirection)); }
                BMVPE_API perspective(float viewWidth, float viewHeight, float nearZ, float farZ) noexcept -> Mat4x4 { return reinterpret<Mat4x4>(DirectX::XMMatrixPerspectiveRH(viewWidth, viewHeight, nearZ, farZ)); }
                BMVPE_API perspective_fov(float fovAngleY, float aspectRatio, float nearZ, float farZ) noexcept -> Mat4x4 { return reinterpret<Mat4x4>(DirectX::XMMatrixPerspectiveFovRH(fovAngleY, aspectRatio, nearZ, farZ)); }
                BMVPE_API perspective_off_center(float viewLeft, float viewRight, float viewBottom, float viewTop, float nearZ, float farZ) noexcept -> Mat4x4 { return reinterpret<Mat4x4>(DirectX::XMMatrixPerspectiveOffCenterRH(viewLeft, viewRight, viewBottom, viewTop, nearZ, farZ)); }
                BMVPE_API orthographic(float viewWidth, float viewHeight, float nearZ, float farZ) noexcept -> Mat4x4 { return reinterpret<Mat4x4>(DirectX::XMMatrixOrthographicRH(viewWidth, viewHeight, nearZ, farZ)); }
                BMVPE_API orthographic_off_center(float viewLeft, float viewRight, float viewBottom, float viewTop, float nearZ, float farZ) noexcept -> Mat4x4 { return reinterpret<Mat4x4>(DirectX::XMMatrixOrthographicOffCenterRH(viewLeft, viewRight, viewBottom, viewTop, nearZ, farZ)); }
            }
        }

        /// Transforms a 1D vector by a given matrix.
        /// RESULT =  m * [x, 0, 0, 1]
        BMVPE_API transform(Vec1f v, Mat4x4 m) noexcept -> Vec4f { return reinterpret<Vec4f>(DirectX::XMVectorMultiplyAdd(DirectX::XMVectorSplatX(v), m[0], m[3])); }

        /// Transforms a 2D vector by a given matrix.
        /// RESULT =  m * [x, y, 0, 1]
        /// XMVector2Transform performs transformations by using the input matrix rows 0 and 1 for rotation and scaling,
        /// and row 3 for translation (effectively assuming row 2 is 0).
        /// The w component of the input vector is assumed to be 0.
        /// The z component of the output vector should be ignored and its w component may be non-homogeneous (!= 1.0).
        BMVPE_API transform(Vec2f v, Mat4x4 m) noexcept -> Vec4f { return reinterpret<Vec4f>(DirectX::XMVector2Transform(v, m)); }

        /// Transforms a 3D vector by a given matrix
        /// RESULT =  m * [x, y, z, 1]
        /// XMVector3Transform ignores the w component of the input vector, and uses a value of 1 instead.
        /// The w component of the returned vector may be non-homogeneous (!= 1.0).
        BMVPE_API transform(Vec3f v, Mat4x4 m) noexcept -> Vec4f { return reinterpret<Vec4f>(DirectX::XMVector3Transform(v, m)); }

        /// Transforms a 4D vector by a given matrix.
        /// RESULT =  m * [x, y, z, w]
        BMVPE_API transform(Vec4f v, Mat4x4 m) noexcept -> Vec4f { return reinterpret<Vec4f>(DirectX::XMVector4Transform(v, m)); }

        /// Transforms a 1D vector by a given matrix, projecting the result back into w = 1.
        /// 1D空間上の座標ベクトル変換を行う
        /// RESULT =  (m * [x, 0, 0, 1]) / w
        BMVPE_API transform_coordinate(Vec1f v, Mat4x4 m) noexcept -> Vec1f
        {
            Vec4f t = transform(v, m);
            return reinterpret<Vec1f>(DirectX::XMVectorDivide(t, DirectX::XMVectorSplatW(t)));
        }

        /// Transforms a 2D vector by a given matrix, projecting the result back into w = 1.
        /// 2D空間上の座標ベクトル変換を行う
        /// RESULT =  (m * [x, y, 0, 1]) / w
        /// XMVector2TransformCoordinate performs transformations by using the input matrix row 0 and row 1 for rotation and scaling,
        /// and row 3 for translation (effectively assuming row 2 is 0).
        /// The w component of the input vector is assumed to be 1.0.
        /// The z component of the returned vector should be ignored and its w component will have a value of 1.0.
        BMVPE_API transform_coordinate(Vec2f v, Mat4x4 m) noexcept -> Vec2f
        {
            return reinterpret<Vec2f>(DirectX::XMVector2TransformCoord(v, m));
        }

        /// Transforms a 3D vector by a given matrix, projecting the result back into w = 1.
        /// 3D空間上の座標ベクトル変換を行う
        /// RESULT =  (m * [x, y, z, 1]) / w
        /// XMVector3TransformCoordinate ignores the w component of the input vector, and uses a value of 1.0 instead.
        /// The w component of the returned vector will always be 1.0.
        BMVPE_API transform_coordinate(Vec3f v, Mat4x4 m) noexcept -> Vec3f
        {
            return reinterpret<Vec3f>(DirectX::XMVector3TransformCoord(v, m));
        }

        /// Transforms the 1D vector normal by the given matrix.
        /// 法線ベクトルの変換を行う
        /// RESULT =  (m * [x, 0, 0, 0])
        BMVPE_API transform_normal(Vec1f v, Mat4x4 m) noexcept -> Vec1f
        {
            return reinterpret<Vec1f>(DirectX::XMVectorMultiply(DirectX::XMVectorSplatX(v), m[0]));
        }

        /// Transforms the 2D vector normal by the given matrix.
        /// 法線ベクトルの変換を行う
        /// RESULT =  (m * [x, y, 0, 0])
        /// XMVector2TransformNormal uses row 0 and 1 of the input transformation matrix for rotation and scaling. Rows 2 and 3 are ignored.
        BMVPE_API transform_normal(Vec2f v, Mat4x4 m) noexcept -> Vec2f
        {
            return reinterpret<Vec2f>(DirectX::XMVector2TransformNormal(v, m));
        }

        /// Transforms the 3D vector normal by the given matrix.
        /// 法線ベクトルの変換を行う
        /// RESULT =  (m * [x, y, z, 0])
        /// XMVector3TransformNormal performs transformations using the input matrix rows 0, 1, and 2 for rotation and scaling, and ignores row 3.
        BMVPE_API transform_normal(Vec3f v, Mat4x4 m) noexcept -> Vec3f
        {
            return reinterpret<Vec3f>(DirectX::XMVector3TransformNormal(v, m));
        }

        /// Project a 3D vector from object space into screen space.
        /// ローカル座標をスクリーン座標に射影変換する
        /// The ViewportX, ViewportY, ViewportWidth, and ViewportHeight parameters describe the position and dimensions of the viewport on
        /// the render-target surface. Usually, applications render to the entire target surface; when rendering on a 640*480 surface, these
        /// parameters should be 0, 0, 640, and 480, respectively. The ViewportMinZ and ViewportMaxZ are typically set to 0.0f and 1.0f but can
        /// be set to other values to achieve specific effects.
        BMVPE_API project(Vec3f v, float viewportX, float viewportY, float viewportWidth, float viewportHeight, float viewportMinZ, float viewportMaxZ, Mat4x4 projection, const Mat4x4& view, const Mat4x4& world) noexcept -> Vec3f
        {
            return reinterpret<Vec3f>(DirectX::XMVector3Project(v, viewportX, viewportY, viewportWidth, viewportHeight, viewportMinZ, viewportMaxZ, projection, view, world));
        }

        /// Projects a 3D vector from screen space into object space.
        /// スクリーン座標からローカル座標に射影変換する
        /// The ViewportX, ViewportY, ViewportWidth, and ViewportHeight parameters describe the position and dimensions of the viewport on
        /// the render-target surface. Usually, applications render to the entire target surface; when rendering on a 640*480 surface, these
        /// parameters should be 0, 0, 640, and 480, respectively. The ViewportMinZ and ViewportMaxZ are typically set to 0.0f and 1.0f but can
        /// be set to other values to achieve specific effects.
        BMVPE_API unproject(Vec3f v, float viewportX, float viewportY, float viewportWidth, float viewportHeight, float viewportMinZ, float viewportMaxZ, Mat4x4 projection, const Mat4x4& view, const Mat4x4& world) noexcept -> Vec3f
        {
            return reinterpret<Vec3f>(DirectX::XMVector3Unproject(v, viewportX, viewportY, viewportWidth, viewportHeight, viewportMinZ, viewportMaxZ, projection, view, world));
        }
    }

    inline namespace memory
    {
        namespace auxiliary
        {
            /// Single
            template <class T>
            struct XMSCALAR
            {
                T x;

                XMSCALAR() = default;

                constexpr XMSCALAR(T x) noexcept
                    : x(x)
                {
                }

                explicit constexpr XMSCALAR(_In_reads_(1) const T pArray[1]) noexcept
                    : x(pArray[0])
                {
                }
            };

            template <class T>
            struct alignas(16) XMSCALARA : XMSCALAR<T>
            {
                using XMSCALAR<T>::XMSCALAR;
#               pragma warning(suppress: 4324) // structure was padded due to __declspec(align())
            };

            struct Float1 : XMSCALAR<float>
            {
                using VectorType = Vec1f;
                using StorageType = XMSCALAR<float>;
                using Scalar = VectorType::Scalar;
                static inline constexpr VectorSize Size = VectorType::Size;

                using StorageType::StorageType;
                Float1(Vec1f vec) : StorageType() { DirectX::XMStoreFloat(&x, vec); }
                Float1& operator =(Vec1f vec) { return DirectX::XMStoreFloat(&x, vec), *this; }
                operator Vec1f() const { return reinterpret<Vec1f>(DirectX::XMLoadFloat(&x)); }
                static Vec1f load(_In_reads_(1) const float src[1]) { return reinterpret<Vec1f>(DirectX::XMLoadFloat(src)); }
                static void store(_Out_writes_(1) float dst[1], _In_ Vec1f vec) { return DirectX::XMStoreFloat(dst, vec); }
            };

            struct Float2 : DirectX::XMFLOAT2
            {
                using VectorType = Vec2f;
                using StorageType = DirectX::XMFLOAT2;
                using Scalar = VectorType::Scalar;
                static inline constexpr VectorSize Size = VectorType::Size;

                using StorageType::StorageType;
                Float2(Vec2f vec) : StorageType() { DirectX::XMStoreFloat2(this, vec); }
                Float2& operator =(Vec2f vec) { return DirectX::XMStoreFloat2(this, vec), *this; }
                operator Vec2f() const { return reinterpret<Vec2f>(DirectX::XMLoadFloat2(this)); }
                static Vec2f load(_In_reads_(2) const float src[2]) { return reinterpret<Vec2f>(DirectX::XMLoadFloat2(reinterpret_cast<const StorageType*>(src))); }
                static void store(_Out_writes_(2)float dst[2], _In_ Vec2f vec) { return DirectX::XMStoreFloat2(reinterpret_cast<StorageType*>(dst), vec); }
            };

            struct Float3 : DirectX::XMFLOAT3
            {
                using VectorType = Vec3f;
                using StorageType = DirectX::XMFLOAT3;
                using Scalar = VectorType::Scalar;
                static inline constexpr VectorSize Size = VectorType::Size;

                using StorageType::StorageType;
                Float3(Vec3f vec) : StorageType() { DirectX::XMStoreFloat3(this, vec); }
                Float3& operator =(Vec3f vec) { return DirectX::XMStoreFloat3(this, vec), *this; }
                operator Vec3f() const { return reinterpret<Vec3f>(DirectX::XMLoadFloat3(this)); }
                static Vec3f load(_In_reads_(3) const float src[3]) { return reinterpret<Vec3f>(DirectX::XMLoadFloat3(reinterpret_cast<const StorageType*>(src))); }
                static void store(_Out_writes_(3)float dst[3], _In_ Vec3f vec) { return DirectX::XMStoreFloat3(reinterpret_cast<StorageType*>(dst), vec); }
            };

            struct Float4 : DirectX::XMFLOAT4
            {
                using VectorType = Vec4f;
                using StorageType = DirectX::XMFLOAT4;
                using Scalar = VectorType::Scalar;

                static inline constexpr VectorSize Size = VectorType::Size;
                using StorageType::StorageType;
                Float4(Vec4f vec) : StorageType() { DirectX::XMStoreFloat4(this, vec); }
                Float4& operator =(Vec4f vec) { return DirectX::XMStoreFloat4(this, vec), *this; }
                operator Vec4f() const { return reinterpret<Vec4f>(DirectX::XMLoadFloat4(this)); }
                static Vec4f load(_In_reads_(4) const float src[4]) { return reinterpret<Vec4f>(DirectX::XMLoadFloat4(reinterpret_cast<const StorageType*>(src))); }
                static void store(_Out_writes_(4) float dst[4], _In_ Vec4f vec) { return DirectX::XMStoreFloat4(reinterpret_cast<StorageType*>(dst), vec); }
            };

            struct Float1a : XMSCALARA<float>
            {
                using VectorType = Vec1f;
                using StorageType = XMSCALARA<float>;
                using Scalar = VectorType::Scalar;
                static inline constexpr VectorSize Size = VectorType::Size;

                using StorageType::StorageType;
                Float1a(Vec1f vec) : StorageType() { DirectX::XMStoreFloat(&x, vec); }
                Float1a& operator =(Vec1f vec) { return DirectX::XMStoreFloat(&x, vec), *this; }
                operator Vec1f() const { return reinterpret<Vec1f>(DirectX::XMLoadFloat(&x)); }
                static Vec1f load_a(_In_reads_(4) const float src[4]) { return reinterpret<Vec1f>(DirectX::XMLoadFloat(src)); }
                static void store_a(_Out_writes_(4) float dst[4], _In_ Vec1f vec) { return DirectX::XMStoreFloat(dst, vec); }
            };

            struct Float2a : DirectX::XMFLOAT2A
            {
                using VectorType = Vec2f;
                using StorageType = DirectX::XMFLOAT2A;
                using Scalar = VectorType::Scalar;
                static inline constexpr VectorSize Size = VectorType::Size;

                using StorageType::StorageType;
                Float2a(Vec2f vec) : StorageType() { DirectX::XMStoreFloat2A(this, vec); }
                Float2a& operator =(Vec2f vec) { return DirectX::XMStoreFloat2A(this, vec), *this; }
                operator Vec2f() const { return reinterpret<Vec2f>(DirectX::XMLoadFloat2A(this)); }
                static Vec2f load_a(_In_reads_(4) const float src[4]) { return reinterpret<Vec2f>(DirectX::XMLoadFloat2A(reinterpret_cast<const StorageType*>(src))); }
                static void store_a(_Out_writes_(4) float dst[4], _In_ Vec2f vec) { return DirectX::XMStoreFloat2A(reinterpret_cast<StorageType*>(dst), vec); }
            };

            struct Float3a : DirectX::XMFLOAT3A
            {
                using VectorType = Vec3f;
                using StorageType = DirectX::XMFLOAT3A;
                using Scalar = VectorType::Scalar;
                static inline constexpr VectorSize Size = VectorType::Size;

                using StorageType::StorageType;
                Float3a(Vec3f vec) : StorageType() { DirectX::XMStoreFloat3A(this, vec); }
                Float3a& operator =(Vec3f vec) { return DirectX::XMStoreFloat3A(this, vec), *this; }
                operator Vec3f() const { return reinterpret<Vec3f>(DirectX::XMLoadFloat3A(this)); }
                static Vec3f load_a(_In_reads_(4) const float src[4]) { return reinterpret<Vec3f>(DirectX::XMLoadFloat3A(reinterpret_cast<const StorageType*>(src))); }
                static void store_a(_Out_writes_(4) float dst[4], _In_ Vec3f vec) { return DirectX::XMStoreFloat3A(reinterpret_cast<StorageType*>(dst), vec); }
            };

            struct Float4a : DirectX::XMFLOAT4A
            {
                using VectorType = Vec4f;
                using StorageType = DirectX::XMFLOAT4A;
                using Scalar = VectorType::Scalar;
                static inline constexpr VectorSize Size = VectorType::Size;

                using StorageType::StorageType;
                Float4a(Vec4f vec) : StorageType() { DirectX::XMStoreFloat4A(this, vec); }
                Float4a& operator =(Vec4f vec) { return DirectX::XMStoreFloat4A(this, vec), *this; }
                operator Vec4f() const { return reinterpret<Vec4f>(DirectX::XMLoadFloat4A(this)); }
                static Vec4f load_a(_In_reads_(4) const float src[4]) { return reinterpret<Vec4f>(DirectX::XMLoadFloat4A(reinterpret_cast<const StorageType*>(src))); }
                static void store_a(_Out_writes_(4) float dst[4], _In_ Vec4f vec) { return DirectX::XMStoreFloat4A(reinterpret_cast<StorageType*>(dst), vec); }
            };

            struct Int1 : std::array<uint32_t, 1>
            {
                using VectorType = Vec1i;
                using StorageType = std::array<uint32_t, 1>;
                using Scalar = VectorType::Scalar;
                static inline constexpr VectorSize Size = VectorType::Size;

                using StorageType::StorageType;
                Int1(Vec1i vec) : StorageType() { DirectX::XMStoreInt(data(), vec); }
                Int1& operator =(Vec1i vec) { return DirectX::XMStoreInt(data(), vec), *this; }
                operator Vec1i() const { return reinterpret<Vec1i>(DirectX::XMLoadInt(data())); }
                static Vec1i load(_In_reads_(1) const uint32_t src[1]) { return reinterpret<Vec1i>(DirectX::XMLoadInt(src)); }
                static void store(_Out_writes_(1) uint32_t dst[1], _In_ Vec1i vec) { return DirectX::XMStoreInt(dst, vec); }
            };

            struct Int2 : std::array<uint32_t, 2>
            {
                using VectorType = Vec2i;
                using StorageType = std::array<uint32_t, 2>;
                using Scalar = VectorType::Scalar;
                static inline constexpr VectorSize Size = VectorType::Size;

                using StorageType::StorageType;
                Int2(Vec2i vec) : StorageType() { DirectX::XMStoreInt2(data(), vec); }
                Int2& operator =(Vec2i vec) { return DirectX::XMStoreInt2(data(), vec), *this; }
                operator Vec2i() const { return reinterpret<Vec2i>(DirectX::XMLoadInt2(data())); }
                static Vec2i load(_In_reads_(2) const uint32_t src[2]) { return reinterpret<Vec2i>(DirectX::XMLoadInt2(src)); }
                static void store(_Out_writes_(2) uint32_t dst[2], _In_ Vec2i vec) { return DirectX::XMStoreInt2(dst, vec); }
            };

            struct Int3 : std::array<uint32_t, 3>
            {
                using VectorType = Vec3i;
                using StorageType = std::array<uint32_t, 3>;
                using Scalar = VectorType::Scalar;
                static inline constexpr VectorSize Size = VectorType::Size;

                using StorageType::StorageType;
                Int3(Vec3i vec) : StorageType() { DirectX::XMStoreInt3(data(), vec); }
                Int3& operator =(Vec3i vec) { return DirectX::XMStoreInt3(data(), vec), *this; }
                operator Vec3i() const { return reinterpret<Vec3i>(DirectX::XMLoadInt3(data())); }
                static Vec3i load(_In_reads_(3) const uint32_t src[3]) { return reinterpret<Vec3i>(DirectX::XMLoadInt3(src)); }
                static void store(_Out_writes_(3) uint32_t dst[3], _In_ Vec3i vec) { return DirectX::XMStoreInt3(dst, vec); }
            };

            struct Int4 : std::array<uint32_t, 4>
            {
                using VectorType = Vec4i;
                using StorageType = std::array<uint32_t, 4>;
                using Scalar = VectorType::Scalar;
                static inline constexpr VectorSize Size = VectorType::Size;

                using StorageType::StorageType;
                Int4(Vec4i vec) : StorageType() { DirectX::XMStoreInt4(data(), vec); }
                Int4& operator =(Vec4i vec) { return DirectX::XMStoreInt4(data(), vec), *this; }
                operator Vec4i() const { return reinterpret<Vec4i>(DirectX::XMLoadInt4(data())); }
                static Vec4i load(_In_reads_(4) const uint32_t src[4]) { return reinterpret<Vec4i>(DirectX::XMLoadInt4(src)); }
                static void store(_Out_writes_(4) uint32_t dst[4], _In_ Vec4i vec) { return DirectX::XMStoreInt4(dst, vec); }
            };

            struct Float4x4 : DirectX::XMFLOAT4X4
            {
                using MatrixType = Mat4x4;
                using StorageType = DirectX::XMFLOAT4X4;
                using Scalar = MatrixType::Scalar;
                static inline constexpr VectorSize Rows = MatrixType::Rows;
                static inline constexpr VectorSize Cols = MatrixType::Cols;

                using StorageType::StorageType;
                Float4x4(Mat4x4 mtx) : StorageType() { DirectX::XMStoreFloat4x4(this, mtx); }
                Float4x4& operator =(Mat4x4 mtx) { return DirectX::XMStoreFloat4x4(this, mtx), *this; }
                operator Mat4x4() const { return reinterpret<Mat4x4>(DirectX::XMLoadFloat4x4(this)); }
            };

            struct Float4x4a : DirectX::XMFLOAT4X4A
            {
                using MatrixType = Mat4x4;
                using StorageType = DirectX::XMFLOAT4X4A;
                using Scalar = MatrixType::Scalar;
                static inline constexpr VectorSize Rows = MatrixType::Rows;
                static inline constexpr VectorSize Cols = MatrixType::Cols;

                using StorageType::StorageType;
                Float4x4a(Mat4x4 mtx) : StorageType() { DirectX::XMStoreFloat4x4A(this, mtx); }
                Float4x4a& operator =(Mat4x4 mtx) { return DirectX::XMStoreFloat4x4A(this, mtx), *this; }
                operator Mat4x4() const { return reinterpret<Mat4x4>(DirectX::XMLoadFloat4x4A(this)); }
            };

            static_assert(sizeof(Float1) == 4);
            static_assert(sizeof(Float2) == sizeof(DirectX::XMFLOAT2));
            static_assert(sizeof(Float3) == sizeof(DirectX::XMFLOAT3));
            static_assert(sizeof(Float4) == sizeof(DirectX::XMFLOAT4));
            static_assert(sizeof(Float1a) == 16);
            static_assert(sizeof(Float2a) == sizeof(DirectX::XMFLOAT2A));
            static_assert(sizeof(Float3a) == sizeof(DirectX::XMFLOAT3A));
            static_assert(sizeof(Float4a) == sizeof(DirectX::XMFLOAT4A));
            static_assert(sizeof(Int1) == 4);
            static_assert(sizeof(Int2) == sizeof(DirectX::XMINT2));
            static_assert(sizeof(Int3) == sizeof(DirectX::XMINT3));
            static_assert(sizeof(Int4) == sizeof(DirectX::XMINT4));
            static_assert(sizeof(Float4x4) == sizeof(DirectX::XMFLOAT4X4));
            static_assert(sizeof(Float4x4a) == sizeof(DirectX::XMFLOAT4X4A));
        }

        using auxiliary::Float1;
        using auxiliary::Float2;
        using auxiliary::Float3;
        using auxiliary::Float4;

        using auxiliary::Float1a;
        using auxiliary::Float2a;
        using auxiliary::Float3a;
        using auxiliary::Float4a;

        using auxiliary::Int1;
        using auxiliary::Int2;
        using auxiliary::Int3;
        using auxiliary::Int4;

        using auxiliary::Float4x4;
        using auxiliary::Float4x4a;
    }

    inline namespace streaming
    {
        /// Transforms 1D vectors by a given matrix.
        /// RESULT =  m * [x, 0, 0, 1]
        BMVPE_API transform_stream(
            Float1 const* input, size_t inputStride, size_t count, Mat4x4 m,
            Float4* output, size_t outputStride) noexcept -> void
        {
            for (size_t i = 0; i < count; i++)
            {
                DirectX::XMStoreFloat4(
                    reinterpret_cast<DirectX::XMFLOAT4*>(reinterpret_cast<std::byte*>(output) + outputStride * i),
                    DirectX::XMVectorMultiplyAdd(
                        DirectX::XMVectorReplicatePtr(reinterpret_cast<const float*>(reinterpret_cast<const std::byte*>(input) + inputStride * i)),
                        m[0],
                        m[3]));
            }
        }

        ///  Transforms 2D vectors by a given matrix.
        /// RESULT =  m * [x, y, 0, 1]
        BMVPE_API transform_stream(
            Float2 const* input, size_t inputStride, size_t count, Mat4x4 m,
            Float4* output, size_t outputStride) noexcept -> void
        {
            DirectX::XMVector2TransformStream(
                output, outputStride,
                input, inputStride, count, m);
        }

        /// Transforms 3D vectors by a given matrix.
        /// RESULT =  m * [x, y, z, 1]
        BMVPE_API transform_stream(
            Float3 const* input, size_t inputStride, size_t count, Mat4x4 m,
            Float4* output, size_t outputStride) noexcept -> void
        {
            DirectX::XMVector3TransformStream(
                output, outputStride,
                input, inputStride, count, m);
        }

        ///  Transforms 4D vectors by a given matrix.
        /// RESULT =  m * [x, y, z, w]
        BMVPE_API transform_stream(
            Float4 const* input, size_t inputStride, size_t count, Mat4x4 m,
            Float4* output, size_t outputStride) noexcept -> void
        {
            DirectX::XMVector4TransformStream(
                output, outputStride,
                input, inputStride, count, m);
        }

        ///  Transforms 1D vectors by a given matrix.
        /// RESULT =  m * [x, 0, 0, 1] / w
        BMVPE_API transform_coordinate_stream(
            Float1 const* input, size_t inputStride, size_t count, Mat4x4 m,
            Float1* output, size_t outputStride) noexcept -> void
        {
            for (size_t i = 0; i < count; i++)
            {
                Vec4f t = reinterpret<Vec4f>(DirectX::XMVectorReplicatePtr(reinterpret_cast<const float*>(reinterpret_cast<const std::byte*>(input) + inputStride * i)));
                t = reinterpret<Vec4f>(DirectX::XMVectorMultiplyAdd(t, m[0], m[3]));
                DirectX::XMStoreFloat(
                    reinterpret_cast<float*>(reinterpret_cast<std::byte*>(output) + outputStride * i),
                    DirectX::XMVectorDivide(t, DirectX::XMVectorSplatW(t)));
            }
        }

        /// Transforms 2D vectors by a given matrix.
        /// RESULT = m * [x, y, 0, 1] / w
        BMVPE_API transform_coordinate_stream(
            Float2 const* input, size_t inputStride, size_t count, Mat4x4 m,
            Float2* output, size_t outputStride)
        {
            DirectX::XMVector2TransformCoordStream(
                output, outputStride,
                input, inputStride, count, m);
        }

        /// Transforms 3D vectors by a given matrix.
        /// RESULT =  m * [x, y, z, 1] / w
        BMVPE_API transform_coordinate_stream(
            Float3 const* input, size_t inputStride, size_t count, Mat4x4 m,
            Float3* output, size_t outputStride)
        {
            DirectX::XMVector3TransformCoordStream(
                output, outputStride,
                input, inputStride, count, m);
        }

        ///  Transforms 1D vectors by a given matrix.
        /// RESULT =  m * [x, 0, 0, 0]
        BMVPE_API transform_normal_stream(
            Float1 const* input, size_t inputStride, size_t count, Mat4x4 m,
            Float1* output, size_t outputStride) noexcept
        {
            for (size_t i = 0; i < count; i++)
            {
                DirectX::XMStoreFloat(
                    reinterpret_cast<float*>(reinterpret_cast<std::byte*>(output) + outputStride * i),
                    DirectX::XMVectorMultiply(
                        DirectX::XMVectorReplicatePtr(reinterpret_cast<const float*>(reinterpret_cast<const std::byte*>(input) + inputStride * i)),
                        m[0])
                );
            }
        }

        /// Transforms 2D vectors by a given matrix.
        /// RESULT = m * [x, y, 0, 0]
        BMVPE_API transform_normal_stream(
            Float2 const* input, size_t inputStride, size_t count, Mat4x4 m,
            Float2* output, size_t outputStride)
        {
            DirectX::XMVector2TransformNormalStream(
                output, outputStride,
                input, inputStride, count, m);
        }

        /// Transforms 3D vectors by a given matrix.
        /// RESULT = m * [x, y, z, 0]
        BMVPE_API transform_normal_stream(
            Float3 const* input, size_t inputStride, size_t count, Mat4x4 m,
            Float3* output, size_t outputStride)
        {
            DirectX::XMVector3TransformNormalStream(
                output, outputStride,
                input, inputStride, count, m);
        }

        /// Projects 3D vectors from object space into screen space.
        BMVPE_API project_stream(
            Float3 const* input, size_t inputStride, size_t count,
            float ViewportX, float ViewportY, float ViewportWidth, float ViewportHeight, float ViewportMinZ, float ViewportMaxZ,
            Mat4x4 Projection, Mat4x4 View, Mat4x4 World,
            Float3* output, size_t outputStride)
        {
            DirectX::XMVector3ProjectStream(
                output, outputStride,
                input, inputStride, count,
                ViewportX, ViewportY, ViewportWidth, ViewportHeight, ViewportMinZ, ViewportMaxZ,
                Projection, View, World);
        }

        /// Projects a 3D vectors from screen space into object space.
        BMVPE_API unproject_stream(
            Float3 const* input, size_t inputStride, size_t count,
            float ViewportX, float ViewportY, float ViewportWidth, float ViewportHeight, float ViewportMinZ, float ViewportMaxZ,
            Mat4x4 Projection, Mat4x4 View, Mat4x4 World,
            Float3* output, size_t outputStride)
        {
            DirectX::XMVector3UnprojectStream(
                output, outputStride,
                input, inputStride, count,
                ViewportX, ViewportY, ViewportWidth, ViewportHeight, ViewportMinZ, ViewportMaxZ,
                Projection, View, World);
        }
    }

    inline namespace Operators
    {
        template <VectorSize N> BMVPE_API operator ==(VecNf<N> lhs, VecNf<N> rhs) noexcept -> bool { return equal(lhs, rhs); }
        template <VectorSize N> BMVPE_API operator !=(VecNf<N> lhs, VecNf<N> rhs) noexcept -> bool { return NotEqual(lhs, rhs); }
        template <VectorSize N> BMVPE_API operator +(VecNf<N> rhs) noexcept -> VecNf<N> { return rhs; }
        template <VectorSize N> BMVPE_API operator -(VecNf<N> rhs) noexcept -> VecNf<N> { return Negate(rhs); }
        template <VectorSize N> BMVPE_API operator +(VecNf<N> lhs, VecNf<N> rhs) noexcept -> VecNf<N> { return Add(lhs, rhs); }
        template <VectorSize N> BMVPE_API operator -(VecNf<N> lhs, VecNf<N> rhs) noexcept -> VecNf<N> { return Subtract(lhs, rhs); }
        template <VectorSize N> BMVPE_API operator *(VecNf<N> lhs, VecNf<N> rhs) noexcept -> VecNf<N> { return Multiply(lhs, rhs); }
        template <VectorSize N> BMVPE_API operator *(VecNf<N> lhs, float rhs) noexcept -> VecNf<N> { return Multiply(lhs, rhs); }
        template <VectorSize N> BMVPE_API operator /(VecNf<N> lhs, VecNf<N> rhs) noexcept -> VecNf<N> { return Divide(lhs, rhs); }
        template <VectorSize N> BMVPE_API operator /(VecNf<N> lhs, float rhs) noexcept -> VecNf<N> { return Divide(lhs, rhs); }
        template <VectorSize N> BMVPE_API operator %(VecNf<N> lhs, VecNf<N> rhs) noexcept -> VecNf<N> { return Mod(lhs, rhs); }
        template <VectorSize N> BMVPE_API operator %(VecNf<N> lhs, float rhs) noexcept -> VecNf<N> { return Mod(lhs, rhs); }
        template <VectorSize N> BMVPE_API operator +=(VecNf<N>& lhs, VecNf<N> rhs) noexcept -> VecNf<N>& { return lhs = lhs + rhs; }
        template <VectorSize N> BMVPE_API operator -=(VecNf<N>& lhs, VecNf<N> rhs) noexcept -> VecNf<N>& { return lhs = lhs - rhs; }
        template <VectorSize N> BMVPE_API operator *=(VecNf<N>& lhs, VecNf<N> rhs) noexcept -> VecNf<N>& { return lhs = lhs * rhs; }
        template <VectorSize N> BMVPE_API operator *=(VecNf<N>& lhs, float rhs) noexcept -> VecNf<N>& { return lhs = lhs * rhs; }
        template <VectorSize N> BMVPE_API operator /=(VecNf<N>& lhs, VecNf<N> rhs) noexcept -> VecNf<N>& { return lhs = lhs / rhs; }
        template <VectorSize N> BMVPE_API operator /=(VecNf<N>& lhs, float rhs) noexcept -> VecNf<N>& { return lhs = lhs / rhs; }
        template <VectorSize N> BMVPE_API operator %=(VecNf<N>& lhs, VecNf<N> rhs) noexcept -> VecNf<N>& { return lhs = lhs % rhs; }
        template <VectorSize N> BMVPE_API operator %=(VecNf<N>& lhs, float rhs) noexcept -> VecNf<N>& { return lhs = lhs % rhs; }

        template <VectorSize N> BMVPE_API operator ==(VecNi<N> lhs, VecNi<N> rhs) noexcept -> bool { return equal(lhs, rhs); }
        template <VectorSize N> BMVPE_API operator !=(VecNi<N> lhs, VecNi<N> rhs) noexcept -> bool { return not_equal(lhs, rhs); }
        template <VectorSize N> BMVPE_API operator +(VecNi<N> rhs) noexcept -> VecNi<N> { return rhs; }
        template <VectorSize N> BMVPE_API operator -(VecNi<N> rhs) noexcept -> VecNi<N> { return Negate(rhs); }
        template <VectorSize N> BMVPE_API operator &(VecNi<N> lhs, VecNi<N> rhs) noexcept -> VecNi<N> { return vector_bw_and(lhs, rhs); }
        template <VectorSize N> BMVPE_API operator |(VecNi<N> lhs, VecNi<N> rhs) noexcept -> VecNi<N> { return vector_bw_or(lhs, rhs); }
        template <VectorSize N> BMVPE_API operator ^(VecNi<N> lhs, VecNi<N> rhs) noexcept -> VecNi<N> { return vector_bw_xor(lhs, rhs); }
        template <VectorSize N> BMVPE_API operator &=(VecNi<N>& lhs, VecNi<N> rhs) noexcept -> VecNi<N>& { return lhs = lhs & rhs; }
        template <VectorSize N> BMVPE_API operator |=(VecNi<N>& lhs, VecNi<N> rhs) noexcept -> VecNi<N>& { return lhs = lhs | rhs; }
        template <VectorSize N> BMVPE_API operator ^=(VecNi<N>& lhs, VecNi<N> rhs) noexcept -> VecNi<N>& { return lhs = lhs ^ rhs; }

        BMVPE_API operator +(Mat4x4 rhs) noexcept -> Mat4x4 { return rhs; }
        BMVPE_API operator -(Mat4x4 rhs) noexcept -> Mat4x4 { return matrix::neg(rhs); }
        BMVPE_API operator +(Mat4x4 lhs, const Mat4x4& rhs) noexcept -> Mat4x4 { return matrix::add(lhs, rhs); }
        BMVPE_API operator -(Mat4x4 lhs, const Mat4x4& rhs) noexcept -> Mat4x4 { return matrix::sub(lhs, rhs); }
        BMVPE_API operator *(Mat4x4 lhs, const Mat4x4& rhs) noexcept -> Mat4x4 { return matrix::mul(lhs, rhs); }
        BMVPE_API operator *(Mat4x4 lhs, float rhs) noexcept -> Mat4x4 { return matrix::mul(lhs, rhs); }
        BMVPE_API operator *(float lhs, Mat4x4 rhs) noexcept -> Mat4x4 { return matrix::mul(rhs, lhs); }
        BMVPE_API operator *(Vec4f lhs, Mat4x4 rhs) noexcept -> Vec4f { return matrix::transform(lhs, rhs); }
        BMVPE_API operator /(Mat4x4 lhs, float rhs) noexcept -> Mat4x4 { return matrix::div(lhs, rhs); }

        BMVPE_API operator +=(Mat4x4& lhs, const Mat4x4& rhs) noexcept -> Mat4x4& { return lhs = lhs + rhs; }
        BMVPE_API operator -=(Mat4x4& lhs, const Mat4x4& rhs) noexcept -> Mat4x4& { return lhs = lhs - rhs; }
        BMVPE_API operator *=(Mat4x4& lhs, const Mat4x4& rhs) noexcept -> Mat4x4& { return lhs = lhs * rhs; }
        BMVPE_API operator *=(Mat4x4& lhs, float rhs) noexcept -> Mat4x4& { return lhs = lhs * rhs; }
        BMVPE_API operator *=(Vec4f& lhs, Mat4x4 rhs) noexcept -> Vec4f& { return lhs = lhs * rhs; }
        BMVPE_API operator /=(Mat4x4& lhs, float rhs) noexcept -> Mat4x4& { return lhs = lhs / rhs; }
    }

    inline namespace color
    {
        using Color4f = Vec4f;
        BMVPE_API color4f(float r, float g, float b, float a = 1.0f) noexcept -> Color4f { return vec4f(r, g, b, a); }

        namespace colors
        {
            BMVPE_API transparent() noexcept -> Color4f { return color4f(0.0f, 0.0f, 0.0f, 0.0f); }
            BMVPE_API transparent_white() noexcept -> Color4f { return color4f(1.0f, 1.0f, 1.0f, 0.0f); }
            BMVPE_API black() noexcept -> Color4f { return color4f(0.0f, 0.0f, 0.0f, 1.0f); }
            BMVPE_API white() noexcept -> Color4f { return color4f(1.0f, 1.0f, 1.0f, 1.0f); }
            BMVPE_API gray(float gray, float alpha = 1.0f) noexcept -> Color4f { return vec4f(gray, gray, gray, alpha); }
            BMVPE_API alpha(float alpha) noexcept -> Color4f { return vec4f(1.0f, 1.0f, 1.0f, alpha); }

            /// argb >> 0 & 0xFF = Blue
            /// argb >> 8 & 0xFF = Green
            /// argb >> 16 & 0xFF = GetRed
            /// argb >> 24 & 0xFF = Alpha
            BMVPE_API from_argb8888(uint32_t argb) -> Color4f
            {
                return reinterpret<Color4f>(DirectX::PackedVector::XMLoadColor(reinterpret_cast<const DirectX::PackedVector::XMCOLOR*>(&argb)));
            }

            /// argb >> 0 & 0xFF = Blue
            /// argb >> 8 & 0xFF = Green
            /// argb >> 16 & 0xFF = GetRed
            /// argb >> 24 & 0xFF = Alpha
            BMVPE_API to_argb8888(Color4f rgba) -> uint32_t
            {
                DirectX::PackedVector::XMCOLOR ret{};
                DirectX::PackedVector::XMStoreColor(&ret, rgba);
                return ret.c;
            }

            /// From [Hue, Sat, Value, Alpha]: Each component has range of 0.0..1.0.
            BMVPE_API from_hsv(Vec4f hsv) noexcept -> Color4f { return reinterpret<Color4f>(DirectX::XMColorHSVToRGB(hsv)); }

            /// To [Hue, Sat, Value, Alpha]: Each component has range of 0.0..1.0.
            BMVPE_API to_hsv(Color4f rgb) noexcept -> Vec4f { return reinterpret<Vec4f>(DirectX::XMColorRGBToHSV(rgb)); }

            /// From [Hue, Sat, Value, Alpha]: Each component has range of 0.0..1.0.
            BMVPE_API from_hsl(Vec4f hsl) noexcept -> Color4f { return reinterpret<Color4f>(DirectX::XMColorHSLToRGB(hsl)); }

            /// To [Hue, Sat, Luminance, Alpha]: Each component has range of 0.0..1.0.
            BMVPE_API to_hsl(Color4f rgb) noexcept -> Vec4f { return reinterpret<Vec4f>(DirectX::XMColorRGBToHSL(rgb)); }

            /// From [Y, U, V, Alpha]: Each component has range of 0.0..1.0.
            /// ITU-R BT.601/CCIR 601 W(r) = 0.299 W(b) = 0.114 U(max) = 0.436 V(max) = 0.615.
            BMVPE_API from_yuv_bt601(Vec4f hsl) noexcept -> Color4f { return reinterpret<Color4f>(DirectX::XMColorYUVToRGB(hsl)); }

            /// To [Y, U, V, Alpha]: Each component has range of 0.0..1.0.
            /// ITU-R BT.601/CCIR 601 W(r) = 0.299 W(b) = 0.114 U(max) = 0.436 V(max) = 0.615.
            BMVPE_API to_yuv_bt601(Color4f rgb) noexcept -> Vec4f { return reinterpret<Vec4f>(DirectX::XMColorRGBToYUV(rgb)); }

            /// From [Y, U, V, Alpha]: Each component has range of 0.0..1.0.
            /// ITU-R BT.709 W(r) = 0.2126 W(b) = 0.0722 U(max) = 0.436 V(max) = 0.615.
            BMVPE_API from_yuv_bt709(Vec4f yuv) noexcept -> Color4f { return reinterpret<Color4f>(DirectX::XMColorYUVToRGB_HD(yuv)); }

            /// To [Y, U, V, Alpha]: Each component has range of 0.0..1.0.
            /// ITU-R BT.709 W(r) = 0.2126 W(b) = 0.0722 U(max) = 0.436 V(max) = 0.615.
            BMVPE_API to_yuv_bt709(Color4f rgb) noexcept -> Vec4f { return reinterpret<Vec4f>(DirectX::XMColorRGBToYUV_HD(rgb)); }
        }

        /// w3c colors
        namespace colors
        {
            constexpr inline static Float4a from_rgb(uint32_t rgb)
            {
                return Float4a{
                    static_cast<float>(rgb >> 16 & 0xff) / 255.0f,
                    static_cast<float>(rgb >> 8 & 0xff) / 255.0f,
                    static_cast<float>(rgb >> 0 & 0xff) / 255.0f,
                    1.0f
                };
            }

            static constexpr inline Float4a Transparent = Float4a(0.0f, 0.0f, 0.0f, 0.0f);
            static constexpr inline Float4a TransparentBlack = Float4a(0.0f, 0.0f, 0.0f, 0.0f);
            static constexpr inline Float4a TransparentWhite = Float4a(1.0f, 1.0f, 1.0f, 0.0f);

            static constexpr inline Float4a AliceBlue = from_rgb(0xF0F8FF);
            static constexpr inline Float4a AntiqueWhite = from_rgb(0xFAEBD7);
            static constexpr inline Float4a Aqua = from_rgb(0x00FFFF);
            static constexpr inline Float4a Aquamarine = from_rgb(0x7FFFD4);
            static constexpr inline Float4a Azure = from_rgb(0xF0FFFF);
            static constexpr inline Float4a Beige = from_rgb(0xF5F5DC);
            static constexpr inline Float4a Bisque = from_rgb(0xFFE4C4);
            static constexpr inline Float4a Black = from_rgb(0x000000);
            static constexpr inline Float4a BlanchedAlmond = from_rgb(0xFFEBCD);
            static constexpr inline Float4a Blue = from_rgb(0x0000FF);
            static constexpr inline Float4a BlueViolet = from_rgb(0x8A2BE2);
            static constexpr inline Float4a Brown = from_rgb(0xA52A2A);
            static constexpr inline Float4a BurlyWood = from_rgb(0xDEB887);
            static constexpr inline Float4a CadetBlue = from_rgb(0x5F9EA0);
            static constexpr inline Float4a Chartreuse = from_rgb(0x7FFF00);
            static constexpr inline Float4a Chocolate = from_rgb(0xD2691E);
            static constexpr inline Float4a Coral = from_rgb(0xFF7F50);
            static constexpr inline Float4a CornflowerBlue = from_rgb(0x6495ED);
            static constexpr inline Float4a Cornsilk = from_rgb(0xFFF8DC);
            static constexpr inline Float4a Crimson = from_rgb(0xDC143C);
            static constexpr inline Float4a Cyan = from_rgb(0x00FFFF);
            static constexpr inline Float4a DarkBlue = from_rgb(0x00008B);
            static constexpr inline Float4a DarkCyan = from_rgb(0x008B8B);
            static constexpr inline Float4a DarkGoldenRod = from_rgb(0xB8860B);
            static constexpr inline Float4a DarkGray = from_rgb(0xA9A9A9);
            static constexpr inline Float4a DarkGreen = from_rgb(0x006400);
            static constexpr inline Float4a DarkKhaki = from_rgb(0xBDB76B);
            static constexpr inline Float4a DarkMagenta = from_rgb(0x8B008B);
            static constexpr inline Float4a DarkOliveGreen = from_rgb(0x556B2F);
            static constexpr inline Float4a DarkOrange = from_rgb(0xFF8C00);
            static constexpr inline Float4a DarkOrchid = from_rgb(0x9932CC);
            static constexpr inline Float4a DarkRed = from_rgb(0x8B0000);
            static constexpr inline Float4a DarkSalmon = from_rgb(0xE9967A);
            static constexpr inline Float4a DarkSeaGreen = from_rgb(0x8FBC8F);
            static constexpr inline Float4a DarkSlateBlue = from_rgb(0x483D8B);
            static constexpr inline Float4a DarkSlateGray = from_rgb(0x2F4F4F);
            static constexpr inline Float4a DarkTurquoise = from_rgb(0x00CED1);
            static constexpr inline Float4a DarkViolet = from_rgb(0x9400D3);
            static constexpr inline Float4a DeepPink = from_rgb(0xFF1493);
            static constexpr inline Float4a DeepSkyBlue = from_rgb(0x00BFFF);
            static constexpr inline Float4a DimGray = from_rgb(0x696969);
            static constexpr inline Float4a DodgerBlue = from_rgb(0x1E90FF);
            static constexpr inline Float4a FireBrick = from_rgb(0xB22222);
            static constexpr inline Float4a FloralWhite = from_rgb(0xFFFAF0);
            static constexpr inline Float4a ForestGreen = from_rgb(0x228B22);
            static constexpr inline Float4a Fuchsia = from_rgb(0xFF00FF);
            static constexpr inline Float4a Gainsboro = from_rgb(0xDCDCDC);
            static constexpr inline Float4a GhostWhite = from_rgb(0xF8F8FF);
            static constexpr inline Float4a Gold = from_rgb(0xFFD700);
            static constexpr inline Float4a GoldenRod = from_rgb(0xDAA520);
            static constexpr inline Float4a Gray = from_rgb(0x808080);
            static constexpr inline Float4a Green = from_rgb(0x008000);
            static constexpr inline Float4a GreenYellow = from_rgb(0xADFF2F);
            static constexpr inline Float4a HoneyDew = from_rgb(0xF0FFF0);
            static constexpr inline Float4a HotPink = from_rgb(0xFF69B4);
            static constexpr inline Float4a IndianRed = from_rgb(0xCD5C5C);
            static constexpr inline Float4a Indigo = from_rgb(0x4B0082);
            static constexpr inline Float4a Ivory = from_rgb(0xFFFFF0);
            static constexpr inline Float4a Khaki = from_rgb(0xF0E68C);
            static constexpr inline Float4a Lavender = from_rgb(0xE6E6FA);
            static constexpr inline Float4a LavenderBlush = from_rgb(0xFFF0F5);
            static constexpr inline Float4a LawnGreen = from_rgb(0x7CFC00);
            static constexpr inline Float4a LemonChiffon = from_rgb(0xFFFACD);
            static constexpr inline Float4a LightBlue = from_rgb(0xADD8E6);
            static constexpr inline Float4a LightCoral = from_rgb(0xF08080);
            static constexpr inline Float4a LightCyan = from_rgb(0xE0FFFF);
            static constexpr inline Float4a LightGoldenRodYellow = from_rgb(0xFAFAD2);
            static constexpr inline Float4a LightGray = from_rgb(0xD3D3D3);
            static constexpr inline Float4a LightGreen = from_rgb(0x90EE90);
            static constexpr inline Float4a LightPink = from_rgb(0xFFB6C1);
            static constexpr inline Float4a LightSalmon = from_rgb(0xFFA07A);
            static constexpr inline Float4a LightSeaGreen = from_rgb(0x20B2AA);
            static constexpr inline Float4a LightSkyBlue = from_rgb(0x87CEFA);
            static constexpr inline Float4a LightSlateGray = from_rgb(0x778899);
            static constexpr inline Float4a LightSteelBlue = from_rgb(0xB0C4DE);
            static constexpr inline Float4a LightYellow = from_rgb(0xFFFFE0);
            static constexpr inline Float4a Lime = from_rgb(0x00FF00);
            static constexpr inline Float4a LimeGreen = from_rgb(0x32CD32);
            static constexpr inline Float4a Linen = from_rgb(0xFAF0E6);
            static constexpr inline Float4a Magenta = from_rgb(0xFF00FF);
            static constexpr inline Float4a Maroon = from_rgb(0x800000);
            static constexpr inline Float4a MediumAquaMarine = from_rgb(0x66CDAA);
            static constexpr inline Float4a MediumBlue = from_rgb(0x0000CD);
            static constexpr inline Float4a MediumOrchid = from_rgb(0xBA55D3);
            static constexpr inline Float4a MediumPurple = from_rgb(0x9370DB);
            static constexpr inline Float4a MediumSeaGreen = from_rgb(0x3CB371);
            static constexpr inline Float4a MediumSlateBlue = from_rgb(0x7B68EE);
            static constexpr inline Float4a MediumSpringGreen = from_rgb(0x00FA9A);
            static constexpr inline Float4a MediumTurquoise = from_rgb(0x48D1CC);
            static constexpr inline Float4a MediumVioletRed = from_rgb(0xC71585);
            static constexpr inline Float4a MidnightBlue = from_rgb(0x191970);
            static constexpr inline Float4a MintCream = from_rgb(0xF5FFFA);
            static constexpr inline Float4a MistyRose = from_rgb(0xFFE4E1);
            static constexpr inline Float4a Moccasin = from_rgb(0xFFE4B5);
            static constexpr inline Float4a NavajoWhite = from_rgb(0xFFDEAD);
            static constexpr inline Float4a Navy = from_rgb(0x000080);
            static constexpr inline Float4a OldLace = from_rgb(0xFDF5E6);
            static constexpr inline Float4a Olive = from_rgb(0x808000);
            static constexpr inline Float4a OliveDrab = from_rgb(0x6B8E23);
            static constexpr inline Float4a Orange = from_rgb(0xFFA500);
            static constexpr inline Float4a OrangeRed = from_rgb(0xFF4500);
            static constexpr inline Float4a Orchid = from_rgb(0xDA70D6);
            static constexpr inline Float4a PaleGoldenRod = from_rgb(0xEEE8AA);
            static constexpr inline Float4a PaleGreen = from_rgb(0x98FB98);
            static constexpr inline Float4a PaleTurquoise = from_rgb(0xAFEEEE);
            static constexpr inline Float4a PaleVioletRed = from_rgb(0xDB7093);
            static constexpr inline Float4a PapayaWhip = from_rgb(0xFFEFD5);
            static constexpr inline Float4a PeachPuff = from_rgb(0xFFDAB9);
            static constexpr inline Float4a Peru = from_rgb(0xCD853F);
            static constexpr inline Float4a Pink = from_rgb(0xFFC0CB);
            static constexpr inline Float4a Plum = from_rgb(0xDDA0DD);
            static constexpr inline Float4a PowderBlue = from_rgb(0xB0E0E6);
            static constexpr inline Float4a Purple = from_rgb(0x800080);
            static constexpr inline Float4a RebeccaPurple = from_rgb(0x663399);
            static constexpr inline Float4a Red = from_rgb(0xFF0000);
            static constexpr inline Float4a RosyBrown = from_rgb(0xBC8F8F);
            static constexpr inline Float4a RoyalBlue = from_rgb(0x4169E1);
            static constexpr inline Float4a SaddleBrown = from_rgb(0x8B4513);
            static constexpr inline Float4a Salmon = from_rgb(0xFA8072);
            static constexpr inline Float4a SandyBrown = from_rgb(0xF4A460);
            static constexpr inline Float4a SeaGreen = from_rgb(0x2E8B57);
            static constexpr inline Float4a SeaShell = from_rgb(0xFFF5EE);
            static constexpr inline Float4a Sienna = from_rgb(0xA0522D);
            static constexpr inline Float4a Silver = from_rgb(0xC0C0C0);
            static constexpr inline Float4a SkyBlue = from_rgb(0x87CEEB);
            static constexpr inline Float4a SlateBlue = from_rgb(0x6A5ACD);
            static constexpr inline Float4a SlateGray = from_rgb(0x708090);
            static constexpr inline Float4a Snow = from_rgb(0xFFFAFA);
            static constexpr inline Float4a SpringGreen = from_rgb(0x00FF7F);
            static constexpr inline Float4a SteelBlue = from_rgb(0x4682B4);
            static constexpr inline Float4a Tan = from_rgb(0xD2B48C);
            static constexpr inline Float4a Teal = from_rgb(0x008080);
            static constexpr inline Float4a Thistle = from_rgb(0xD8BFD8);
            static constexpr inline Float4a Tomato = from_rgb(0xFF6347);
            static constexpr inline Float4a Turquoise = from_rgb(0x40E0D0);
            static constexpr inline Float4a Violet = from_rgb(0xEE82EE);
            static constexpr inline Float4a Wheat = from_rgb(0xF5DEB3);
            static constexpr inline Float4a White = from_rgb(0xFFFFFF);
            static constexpr inline Float4a WhiteSmoke = from_rgb(0xF5F5F5);
            static constexpr inline Float4a Yellow = from_rgb(0xFFFF00);
            static constexpr inline Float4a YellowGreen = from_rgb(0x9ACD32);
        };
    }
}
