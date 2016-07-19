/************************************************************************/
/*                                                                      */
/*               Copyright 2014-2015 by Ullrich Koethe                  */
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

#ifndef VIGRA2_ALGORITHM_ND_HXX
#define VIGRA2_ALGORITHM_ND_HXX

#include "array_nd.hxx"

namespace vigra {

/**********************************************************/
/*                                                        */
/*                    foreachCoordinate                   */
/*                                                        */
/**********************************************************/

namespace array_detail {

template <int N, class FCT, int M>
inline void
foreachCoordinateImpl(Shape<N> & coord,
                      Shape<N> const & start, Shape<N> const & end,
                      FCT && f, Shape<M> const & order, int dim)
{
    int axis = order[dim];
    if (dim == 0)
    {
        for (coord[axis] = start[axis]; coord[axis] < end[axis]; ++coord[axis])
            f(coord);
    }
    else
    {
        for (coord[axis] = start[axis]; coord[axis] < end[axis]; ++coord[axis])
            foreachCoordinateImpl(coord, start, end, std::forward<FCT>(f), order, dim - 1);
    }
}

} // namespace array_detail

    // execute a nested loop whose loop body is specified by a functor
template <int N, class FCT>
inline void
foreachCoordinate(Shape<N> const & start, Shape<N> const & end,
                  FCT && f, MemoryOrder order = C_ORDER)
{
    const int size = start.size();
    vigra_precondition(size >= 1 && size == end.size() && allLessEqual(start, end),
        "foreachCoordinate(): invalid shapes.");

    Shape<N> coord(start),
             axis_order = (order == C_ORDER)
                              ? reversed(Shape<N>::range(size))
                              : Shape<N>::range(size);
    array_detail::foreachCoordinateImpl(coord, start, end, std::forward<FCT>(f), axis_order, size-1);
}

template <int N, class FCT>
inline void
foreachCoordinate(Shape<N> const & shape, FCT && f, MemoryOrder order = C_ORDER)
{
    foreachCoordinate(Shape<N>(tags::size=shape.size(), 0), shape, std::forward<FCT>(f), order);
}

template <int N, class FCT, int M>
inline void
foreachCoordinate(Shape<N> const & start, Shape<N> const & end,
                  FCT && f, Shape<M> const & order)
{
    const int size = start.size();
    vigra_precondition(size >= 1 && size == end.size() && size == order.size() && allLessEqual(start, end),
        "foreachCoordinate(): invalid shapes.");

    Shape<N> coord(start);
    array_detail::foreachCoordinateImpl(coord, start, end, std::forward<FCT>(f), order, size-1);
}

template <int N, class FCT, int M>
inline void
foreachCoordinate(Shape<N> const & shape, FCT && f, Shape<M> const & order)
{
    foreachCoordinate(Shape<N>(tags::size=shape.size(), 0), shape, std::forward<FCT>(f), order);
}

/**********************************************************/
/*                                                        */
/*                        foreachND                       */
/*                                                        */
/**********************************************************/

template <class ARRAY, class FCT,
          VIGRA_REQUIRE<ArrayNDConcept<ARRAY>::value> >
void
foreachND(ARRAY && array, FCT && f)
{
    array_detail::universalArrayNDFunction(std::forward<ARRAY>(array), std::forward<FCT>(f));
}

/**********************************************************/
/*                                                        */
/*                      transformND                       */
/*                                                        */
/**********************************************************/

template <class ARRAY1, class ARRAY2, class FCT,
          VIGRA_REQUIRE<ArrayNDConcept<ARRAY1>::value && ArrayNDConcept<ARRAY2>::value> >
void
transformND(ARRAY1 const & src, ARRAY2 & target, FCT && f)
{
    array_detail::universalArrayNDFunction(target, src,
        [f](typename ARRAY2::reference u, typename ARRAY1::const_reference v) {
            u = detail::RequiresExplicitCast<typename ARRAY2::value_type>::cast(f(v));
        }
    );
}

template <class ARRAY1, class ARRAY2, class ARRAY3, class FCT,
          VIGRA_REQUIRE<ArrayNDConcept<ARRAY1>::value &&
                        ArrayNDConcept<ARRAY2>::value &&
                        ArrayNDConcept<ARRAY3>::value> >
void
transformND(ARRAY1 const & src1, ARRAY2 const & src2, ARRAY3 & target, FCT && f)
{
    using namespace array_math;
    typedef ArrayMathExpression<ArrayMathCustomFunctor<ARRAY1, ARRAY2, FCT>> FCT_WRAPPER;
    array_detail::universalArrayMathFunction(target, FCT_WRAPPER(src1, src2, std::forward<FCT>(f)),
        [](typename ARRAY3::reference u, typename FCT_WRAPPER::result_type v) {
            u = detail::RequiresExplicitCast<typename ARRAY3::value_type>::cast(v);
        },
        "transformND()"
    );
}

} // namespace vigra

#endif // VIGRA2_ALGORITHM_ND_HXX
