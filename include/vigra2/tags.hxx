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

#ifndef VIGRA2_TAGS_HXX
#define VIGRA2_TAGS_HXX

namespace vigra {

    /// use biggest signed type for array indexing
using ArrayIndex = std::ptrdiff_t;

    /// constants to specialize templates whose size/ndim is only known at runtime
static const int runtime_size  = -1;
static const int runtime_ndim  = -1;
static const int runtime_order = -1;

enum SkipInitialization { DontInit };
enum ReverseCopyTag { ReverseCopy };
enum MemoryOrder { C_ORDER = 1, F_ORDER = 2, RowMajor = C_ORDER, ColumnMajor = F_ORDER };

template <class VALUETYPE, int M=runtime_size, int ... N>
class TinyArray;

template <class VALUETYPE, int M=runtime_size, int ... N>
class TinyArrayView;

/********************************************************/
/*                                                      */
/*                 neighborhood types                   */
/*                                                      */
/********************************************************/

    /** \brief Choose the neighborhood system in a dimension-independent way.

        DirectNeighborhood corresponds to 4-neighborhood in 2D and 6-neighborhood in 3D, whereas
        IndirectNeighborhood means 8-neighborhood in 2D and 26-neighborhood in 3D. The general
        formula for N dimensions are 2*N for direct neighborhood and 3^N-1 for indirect neighborhood.
    */
enum NeighborhoodType {
        DirectNeighborhood=0,   ///< use only direct neighbors
        IndirectNeighborhood=1  ///< use direct and indirect neighbors
};

/********************************************************/
/*                                                      */
/*                    lemon::Invalid                    */
/*                                                      */
/********************************************************/

namespace lemon {

#if defined(VIGRA_WITH_LEMON)

using Invalid = ::lemon::Invalid;

#else

struct Invalid
{
  public:
    bool operator==(Invalid) { return true;  }
    bool operator!=(Invalid) { return false; }
    bool operator< (Invalid) { return false; }
};

#endif

const Invalid INVALID = Invalid();

} // namespace lemon

namespace tags {

/********************************************************/
/*                                                      */
/*                         AxisTag                      */
/*                                                      */
/********************************************************/

    // Tags to assign semantic meaning to axes.
    // (arranged in sorting order)
enum AxisTag  { no_channel_axis = -1,
                axis_unknown = 0,
                axis_c,  // channel axis
                axis_n,  // node map for a graph
                axis_x,  // spatial x-axis
                axis_y,  // spatial y-axis
                axis_z,  // spatial z-axis
                axis_t,  // time axis
                axis_fx, // Fourier transform of x-axis
                axis_fy, // Fourier transform of y-axis
                axis_fz, // Fourier transform of z-axis
                axis_ft, // Fourier transform of t-axis
                axis_e,  // edge map for a graph
                axis_end // marker for the end of the list
              };

/********************************************************/
/*                                                      */
/*                       tags::axis                     */
/*                                                      */
/********************************************************/

    // Support for tags::axis keyword argument to select
    // the axis an algorithm is supposed to operator on
struct AxisSelectionProxy
{
    int value;
};

struct AxisSelectionTag
{
    AxisSelectionProxy operator=(int i) const
    {
        return {i};
    }

    AxisSelectionProxy operator()(int i) const
    {
        return {i};
    }
};

namespace {

AxisSelectionTag axis;

}

/********************************************************/
/*                                                      */
/*                  tags::byte_strides                  */
/*                                                      */
/********************************************************/

    // Support for tags::byte_strides keyword argument
    // to pass strides in units of bytes rather than `sizeof(T)`.
template <int N>
struct ByteStridesProxy
{
    TinyArray<ArrayIndex, N> value;
};

struct ByteStridesTag
{
    template <int N>
    ByteStridesProxy<N> operator=(TinyArray<ArrayIndex, N> const & s) const
    {
        return {s};
    }

    template <int N>
    ByteStridesProxy<N> operator()(TinyArray<ArrayIndex, N> const & s) const
    {
        return {s};
    }
};

namespace {

ByteStridesTag byte_strides;

}

/********************************************************/
/*                                                      */
/*                       tags::size                     */
/*                                                      */
/********************************************************/

    // Support for tags::size keyword argument
    // to disambiguate array sizes from initial values.
struct SizeProxy
{
    ArrayIndex value;
};

struct SizeTag
{
    SizeProxy operator=(ArrayIndex s) const
    {
        return {s};
    }

    SizeProxy operator()(ArrayIndex s) const
    {
        return {s};
    }
};

namespace {

SizeTag size;

}

}} // namespace vigra::tags

#endif // VIGRA2_TAGS_HXX
