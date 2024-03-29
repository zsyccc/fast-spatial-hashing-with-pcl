#ifndef FSH_UTIL_HPP
#define FSH_UTIL_HPP
#pragma once

#include <cmath>
#include "point.hpp"

namespace psh {
    namespace {
        // these functions convert between multidimensional (points) and linear
        // (index) coordinates

        // helper struct to allow partial specialization of function templates
        template<uint d, class Int>
        struct point_helpers {
            static constexpr Int point_to_index(const point <d, Int> &p, Int width, Int max) {
                Int index = p[0];
                for (uint i = 1; i < d; i++) {
                    index += width * p[i];
                    width *= width;
                }
                return index % max;
            }

            static constexpr point <d, Int> index_to_point(Int index, Int width, Int max) {
                point <d, Int> output;
                max /= width;
                for (uint i = 0; i < d; i++) {
                    output[i] = index / max;
                    index = index % max;
                    if (i + 1 < d) {
                        max /= width;
                    }
                }
                return output;
            }
        };

        template<class Int>
        struct point_helpers<2, Int> {
            static constexpr Int point_to_index(const point<2, Int> &p, Int width, Int max) {
                return (width * p[1] + p[0]) % max;
            }

            static constexpr point<2, Int> index_to_point(Int index, Int width, Int max) {
                return point < 2, Int > {index % width, index / width};
            }
        };

        template<class Int>
        struct point_helpers<3, Int> {
            static constexpr Int point_to_index(const point<3, Int> &p, Int width, Int max) {
                return (width * width * p[2] + width * p[1] + p[0]) % max;
            }

            static constexpr point<3, Int> index_to_point(Int index, Int width, Int max) {
                return point < 3, Int > {(index % (width * width)) % width, (index % (width * width)) / width,
                                         index / (width * width),};
            }
        };
    }  // namespace

    template<uint d, class Int>
    constexpr Int point_to_index(const point <d, Int> &p, Int width, Int max) {
//         static_assert(sizeof(IntS) <= sizeof(IntL),
//                       "IntS must be smaller or equal to IntL");
        return point_helpers<d, Int>::point_to_index(point<d, Int>(p), Int(width), max);
    }

    template<uint d, class IntS, class IntL>
    constexpr IntL point_to_index(const point <d, IntL> &p, IntS width, IntL max) {
        static_assert(sizeof(IntS) <= sizeof(IntL), "IntS must be smaller or equal to IntL");
        return point_helpers<d, IntL>::point_to_index(point<d, IntL>(p), IntL(width), max);
    }

    template<uint d, class IntS, class IntL>
    constexpr IntL point_to_index(const point <d, IntS> &p, IntS width, IntL max) {
        static_assert(sizeof(IntS) <= sizeof(IntL), "IntS must be smaller or equal to IntL");
        return point_helpers<d, IntL>::point_to_index(point<d, IntL>(p), IntL(width), max);
    }

    template<uint d, class IntS, class IntL>
    constexpr point <d, IntS> index_to_point(IntL index, IntS width, IntL max) {
        static_assert(sizeof(IntS) <= sizeof(IntL),
                      "IntS must be smaller or equal to IntL");
        return point<d, IntS>(point_helpers<d, IntL>::index_to_point(index, IntL(width), max));
    }
}  // namespace psh

#endif //FSH_UTIL_HPP
