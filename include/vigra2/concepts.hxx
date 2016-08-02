/************************************************************************/
/*                                                                      */
/*               Copyright 2014-2016 by Ullrich Koethe                  */
/*                                                                      */
/*    This file is part of the VIGRA2 computer vision library.          */
/*    The VIGRA2 Website is                                             */
/*        http://ukoethe.github.io/vigra2                               */
/*    Please direct questions, bug reports, and contributions to        */
/*        ullrich.koethe@iwr.uni-heidelberg.de    or                    */
/*        vigra@informatik.uni-hamburg.de                               */
/*                                                                      */
/*    Permission is hereby granted, free of charge, to any person       */
/*    obtaining a copy of this software and associated documentation    */
/*    files (the "Software"), to deal in the Software without           */
/*    restriction, including without limitation the rights to use,      */
/*    copy, modify, merge, publish, distribute, sublicense, and/or      */
/*    sell copies of the Software, and to permit persons to whom the    */
/*    Software is furnished to do so, subject to the following          */
/*    conditions:                                                       */
/*                                                                      */
/*    The above copyright notice and this permission notice shall be    */
/*    included in all copies or substantial portions of the             */
/*    Software.                                                         */
/*                                                                      */
/*    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND    */
/*    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES   */
/*    OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND          */
/*    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT       */
/*    HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,      */
/*    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING      */
/*    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR     */
/*    OTHER DEALINGS IN THE SOFTWARE.                                   */
/*                                                                      */
/************************************************************************/

#pragma once

#ifndef VIGRA2_CONCEPTS_HXX
#define VIGRA2_CONCEPTS_HXX

#include "config.hxx"
#include "tags.hxx"
#include <type_traits>

namespace vigra {

/**********************************************************/
/*                                                        */
/*                     concept checking                   */
/*                                                        */
/**********************************************************/

namespace concepts_detail {
    struct Unsupported {};
}

using std::enable_if;
using std::common_type;

template <bool Predicate, class T=void>
using enable_if_t = typename enable_if<Predicate, T>::type;

template <class U, class V>
using common_type_t = typename common_type<U, V>::type;

struct require_ok {};

template <bool CONCEPTS>
using ConceptCheck = typename enable_if<CONCEPTS, require_ok>::type;

#define VIGRA_REQUIRE typename Require = ConceptCheck

/**********************************************************/
/*                                                        */
/*            multi-dimensional array concept             */
/*                                                        */
/**********************************************************/

struct TinyArrayTag {};

    // TinyArrayConcept refers to TinyArrayBase and TinyArray.
    // By default, 'ARRAY' fulfills the TinyArrayConcept if it is derived
    // from TinyArrayTag.
    //
    // Alternatively, one can partially specialize TinyArrayConcept.
template <class ARRAY>
struct TinyArrayConcept
{
    static const bool value =std::is_base_of<TinyArrayTag, ARRAY>::value;
};

/**********************************************************/
/*                                                        */
/*            multi-dimensional array concept             */
/*                                                        */
/**********************************************************/

struct PointerNDTag {};

    // PointerNDConcept refers to the low-level multi-dimensional array API.
    // By default, 'ARRAY' conforms to the PointerNDConcept if it is derived
    // from PointerNDTag.
    //
    // Alternatively, one can partially specialize PointerNDConcept.
template <class ARRAY>
struct PointerNDConcept
{
    typedef typename std::remove_reference<ARRAY>::type PLAIN_ARRAY;
    static const bool value = std::is_base_of<PointerNDTag, PLAIN_ARRAY>::value;
};

struct ArrayNDTag {};

    // ArrayNDConcept refers to the high-level multi-dimensional array API.
    // By default, 'ARRAY' conforms to the ArrayNDConcept if it is derived
    // from 'ArrayNDTag'.
    //
    // Alternatively, one can partially specialize ArrayNDConcept.
template <class ARRAY>
struct ArrayNDConcept
{
    typedef typename std::remove_reference<ARRAY>::type PLAIN_ARRAY;
    static const bool value = std::is_base_of<ArrayNDTag, PLAIN_ARRAY>::value;
};

struct ArrayMathTag
: public PointerNDTag
{};

    // ArrayMathConcept refers to array expression templates.
    // By default, 'ARRAY' conforms to the ArrayMathConcept if it
    // is derived from 'ArrayMathTag'.
    //
    // Alternatively, one can partially specialize ArrayMathConcept.
template <class ARRAY>
struct ArrayMathConcept
{
    typedef typename std::remove_reference<ARRAY>::type PLAIN_ARRAY;
    static const bool value = std::is_base_of<ArrayMathTag, PLAIN_ARRAY>::value;
};

    // ArrayLikeConcept is fulfilled when 'ARRAY' is either an array or an
    // array expression template.
    //
    // Alternatively, one can partially specialize ArrayLikeConcept.
template <class ARRAY>
struct ArrayLikeConcept
{
    static const bool value = ArrayNDConcept<ARRAY>::value || ArrayMathConcept<ARRAY>::value;
};

/**********************************************************/
/*                                                        */
/*       dimensions of multi-dimensional arrays           */
/*                                                        */
/**********************************************************/

template <int N, int M>
struct CompatibleDimensions
{
    static const bool value = N == M || N == runtime_size || M == runtime_size;
};

template <class ARRAY>
struct NDimConcept
{
    typedef typename std::decay<ARRAY>::type T;

    static char test(...);

    template <class U>
    static int test(U*, int = U::dimension);

    static const bool value = std::is_same<decltype(test((T*)0)), int>::value;
};

template <class ARRAY, bool = NDimConcept<ARRAY>::value>
struct NDimTraits
{};

template <class ARRAY>
struct NDimTraits<ARRAY, true>
{
    typedef typename std::decay<ARRAY>::type T;

    static const int value = T::dimension;
};

/**********************************************************/
/*                                                        */
/*            check if a class is an iterator             */
/*                                                        */
/**********************************************************/

    // currently, we apply only the simple rule that class T
    // must be a pointer or array or has an embedded typedef
    // 'iterator_category'. More sophisticated checks should
    // be added when needed.
template <class T>
struct IteratorConcept
{
    typedef typename std::decay<T>::type V;

    static char test(...);

    template <class U>
    static int test(U*, typename U::iterator_category * = 0);

    static const bool value =
        std::is_array<T>::value ||
        std::is_pointer<T>::value ||
        std::is_same<decltype(test((V*)0)), int>::value;
};

} // namespace vigra

#endif // VIGRA2_CONCEPTS_HXX
