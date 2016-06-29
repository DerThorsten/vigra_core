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

#ifndef VIGRA2_ITERATOR_ND_HXX
#define VIGRA2_ITERATOR_ND_HXX

#include <vector>
#include <utility>
#include "config.hxx"
#include "concepts.hxx"
#include "tinyarray.hxx"
#include "shape.hxx"
#include "handle_nd.hxx"

// Bounds checking Macro used if VIGRA_CHECK_BOUNDS is defined.
#ifdef VIGRA_CHECK_BOUNDS
#define VIGRA_ASSERT_INSIDE(diff) \
  vigra_precondition(this->isInside(diff), "Index out of bounds")
#else
#define VIGRA_ASSERT_INSIDE(diff)
#endif

namespace vigra {

/** \addtogroup MultiIteratorGroup
*/
//@{

namespace array_detail {

/********************************************************/
/*                                                      */
/*                  IteratorNDAxisInfo                  */
/*                                                      */
/********************************************************/

template <int N, int ORDER>
struct IteratorNDAxisInfo;

template <int N>
struct IteratorNDAxisInfo<N, C_ORDER>
{
    IteratorNDAxisInfo(int)
    {}

    static const int minor_ = 0;
    static const int major_ = N-1;
};

template <>
struct IteratorNDAxisInfo<runtime_size, C_ORDER>
{
    IteratorNDAxisInfo(int major)
    : major_(major)
    {}

    static const int minor_ = 0;
    int major_;
};

template <int N>
struct IteratorNDAxisInfo<N, F_ORDER>
{
    IteratorNDAxisInfo(int)
    {}

    static const int minor_ = N-1;
};

template <>
struct IteratorNDAxisInfo<runtime_size, F_ORDER>
{
    IteratorNDAxisInfo(int minor)
    : minor_(minor)
    {}

    int minor_;
};

template <int N>
struct IteratorNDAxisInfo<N, runtime_order>
{
    IteratorNDAxisInfo(int minor)
    : minor_(minor)
    {}

    int minor_;
};

template <>
struct IteratorNDAxisInfo<runtime_size, runtime_order>
{
    IteratorNDAxisInfo(int minor)
    : minor_(minor)
    {}

    int minor_;
};

/********************************************************/
/*                                                      */
/*          IteratorNDBase<..., F_ORDER>                */
/*                                                      */
/********************************************************/

template <class HANDLES, int ORDER>
class IteratorNDBase;

template <class HANDLES>
struct IteratorNDBase<HANDLES, F_ORDER>
: protected IteratorNDAxisInfo<HANDLES::dimension, F_ORDER>
{
  protected:

    static const int N = HANDLES::dimension;

    HANDLES handles_;

    IteratorNDBase(HANDLES const & h)
    : IteratorNDAxisInfo<N, F_ORDER>(h.ndim()-1)
    , handles_(h)
    {}

    void operator++()
    {
        handles_.inc(0);
        if(handles_.coord(0) == handles_.shape(0))
        {
            for(int k=1; k<=minor_; ++k)
            {
                handles_.move(k-1, -handles_.shape(k-1));
                handles_.inc(k);
                if(handles_.coord(k) < handles_.shape(k))
                    break;
            }
        }
    }

    void operator--()
    {
        handles_.dec(0);
        if(handles_.coord(0) < 0)
        {
            for(int k=1; k<=minor_; ++k)
            {
                handles_.move(k-1, handles_.shape(k-1));
                handles_.dec(k);
                if(handles_.coord(k) >= 0)
                    break;
            }
        }
    }

    bool operator!=(HANDLES const & other) const
    {
        for(int k = minor_; k >= 0; --k)
            if(handles_.coord(k) != other.coord(k))
                return true;
        return false;
    }

    Shape<N> scanOrderToCoordinate(ArrayIndex d) const
    {
        Shape<N> res(handles_.ndim(), DontInit);
        for(int k=0; k<=minor_; ++k)
        {
            res[k] = d % handles_.shape(k);
            d /= handles_.shape(k);
        }
        return res;
    }

    template <class SHAPE>
    ArrayIndex scanOrderIndex(SHAPE const & s) const
    {
        ArrayIndex stride = 1, res = 0;
        for(int k=0; k<=minor_; ++k)
        {
            res += stride*s[k];
            stride *= handles_.shape(k);
        }
        return res;

    }

  public:

    // FIXME: should the scan-order index be stored in IteratorND?
    ArrayIndex scanOrderIndex() const
    {
        return scanOrderIndex(handles_.coord());
    }
};

/********************************************************/
/*                                                      */
/*          IteratorNDBase<..., C_ORDER>                */
/*                                                      */
/********************************************************/

template <class HANDLES>
struct IteratorNDBase<HANDLES, C_ORDER>
: protected IteratorNDAxisInfo<HANDLES::dimension, C_ORDER>
{
  protected:

    static const int N = HANDLES::dimension;

    HANDLES handles_;

    IteratorNDBase(HANDLES const & h)
    : IteratorNDAxisInfo<N, C_ORDER>(h.ndim()-1)
    , handles_(h)
    {}

    void operator++()
    {
        handles_.inc(major_);
        if(handles_.coord(major_) == handles_.shape(major_))
        {
            for(int k=major_-1; k>=0; --k)
            {
                handles_.move(k+1, -handles_.shape(k+1));
                handles_.inc(k);
                if(handles_.coord(k) < handles_.shape(k))
                    break;
            }
        }
    }

    void operator--()
    {
        handles_.dec(major_);
        if(handles_.coord(major_) < 0)
        {
            for(int k=major_-1; k>=0; --k)
            {
                handles_.move(k+1, handles_.shape(k+1));
                handles_.dec(k);
                if(handles_.coord(k) >= 0)
                    break;
            }
        }
    }

    bool operator!=(HANDLES const & other) const
    {
        for(int k = 0; k <= major_; ++k)
            if(handles_.coord(k) != other.coord(k))
                return true;
        return false;
    }

    Shape<N> scanOrderToCoordinate(ArrayIndex d) const
    {
        Shape<N> res(handles_.ndim(), DontInit);
        for(int k=major_; k>=0; --k)
        {
            res[k] = d % handles_.shape(k);
            d /= handles_.shape(k);
        }
        return res;
    }

    template <class SHAPE>
    ArrayIndex scanOrderIndex(SHAPE const & s) const
    {
        ArrayIndex stride = 1, res = 0;
        for(int k=major_; k>=0; --k)
        {
            res += stride*s[k];
            stride *= handles_.shape(k);
        }
        return res;
    }

  public:

    // FIXME: should the scan-order index be stored in IteratorND?
    ArrayIndex scanOrderIndex() const
    {
        return scanOrderIndex(handles_.coord());
    }
};

/********************************************************/
/*                                                      */
/*          IteratorNDBase<..., runtime_order>          */
/*                                                      */
/********************************************************/

template <class HANDLES>
struct IteratorNDBase<HANDLES, runtime_order>
: protected IteratorNDAxisInfo<HANDLES::dimension, runtime_order>
{
  protected:

    static const int N = HANDLES::dimension;

    HANDLES handles_;
    Shape<N> order_;

    IteratorNDBase(HANDLES const & h,
                   Shape<N> const & order)
    : IteratorNDAxisInfo<N, runtime_order>(order.back())
    , handles_(h)
    , order_(order)
    {}

    void operator++()
    {
        handles_.inc(order_[0]);
        if(handles_.coord(order_[0]) == handles_.shape(order_[0]))
        {
            for(int k=1; k<order_.size(); ++k)
            {
                handles_.move(order_[k-1], -handles_.shape(order_[k-1]));
                handles_.inc(order_[k]);
                if(handles_.coord(order_[k]) < handles_.shape(order_[k]))
                    break;
            }
        }
    }

    void operator--()
    {
        handles_.dec(order_[0]);
        if(handles_.coord(order_[0]) < 0)
        {
            for(int k=1; k<order_.size(); ++k)
            {
                handles_.move(order_[k-1], handles_.shape(order_[k-1]));
                handles_.dec(order_[k]);
                if(handles_.coord(order_[k]) >= 0)
                    break;
            }
        }
    }

    bool operator!=(HANDLES const & other) const
    {
        for(int k = order_.size()-1; k >= 0; --k)
            if(handles_.coord(order_[k]) != other.coord(order_[k]))
                return true;
        return false;
    }

    Shape<N> scanOrderToCoordinate(ArrayIndex d) const
    {
        Shape<N> res(handles_.ndim(), DontInit);
        for(int k=0; k<order_.size(); ++k)
        {
            res[order_[k]] = d % handles_.shape(order_[k]);
            d /= handles_.shape(order_[k]);
        }
        return res;
    }

    template <class SHAPE>
    ArrayIndex scanOrderIndex(SHAPE const & s) const
    {
        ArrayIndex stride = 1, res = 0;
        for(int k=0; k<order_.size(); ++k)
        {
            res += stride*s[order_[k]];
            stride *= handles_.shape(order_[k]);
        }
        return res;
    }

  public:

    // FIXME: should the scan-order index be stored in IteratorND?
    ArrayIndex scanOrderIndex() const
    {
        return scanOrderIndex(handles_.coord());
    }
};

} // namespace array_detail

/********************************************************/
/*                                                      */
/*                  IteratorND                          */
/*                                                      */
/********************************************************/

// FIXME: benchmark if hard-coding F_ORDER and C_ORDER is beneficial in IteratorND

/** \brief Iterate over multiple images simultaneously in scan order.

    The value type of this iterator is an instance of the handle class CoupledHandle. This allows to iterate over multiple arrays simultaneously. The coordinates can be accessed as a special band (index 0) in the handle. The scan-order is defined such that dimensions are iterated from front to back (first to last).

    Instances of this class are usually constructed by calling createCoupledIterator() .

    To get the type of a IteratorND for arrays of a certain dimension and element types use CoupledIteratorType::type.

    The iterator supports all functions listed in the STL documentation for
        <a href="http://www.sgi.com/tech/stl/RandomAccessIterator.html">Random Access Iterators</a>.

    Example of use:
    \code
    using namespace vigra;
    MultiArray<2, double> image1(Shape2(5, 5));
    MultiArray<2, double> image2(Shape2(5, 5));
    // fill image with data ...

    typedef CoupledIteratorType<2, double, double>::type Iterator; // the type of the IteratorND

    Iterator start = createCoupledIterator(image1, image2); // create coupled iterator for simultaneous iteration over image1, image2 and their coordinates
    Iterator end = start.getEndIterator();

    for (Iterator it = start; it < end; ++it) {
      std::cout << "coordinates: " << it.get<0>() << std::endl;
      std::cout << "image1: " << it.get<1>() << std::endl;
      std::cout << "image2: " << it.get<2>() << std::endl;
    }

    //random access:
    Iterator::value_type handle = start[15];
    std::cout << "image1: " << get<1>(handle) << std::endl;
    \endcode

    <b>\#include</b> \<vigra/multi_iterator_coupled.hxx\> <br/>
    Namespace: vigra
*/
template <class HANDLES, int ORDER = runtime_order>
class IteratorND
: public array_detail::IteratorNDBase<HANDLES, ORDER>
{
    static_assert(ORDER == runtime_order || ORDER == C_ORDER || ORDER == F_ORDER,
        "IteratorND<N, HANDLES, ORDER>: Order must be one of runtime_order, C_ORDER, F_ORDER.");
  public:

    static const int N = HANDLES::dimension;
    typedef array_detail::IteratorNDBase<HANDLES, ORDER>  base_type;
    typedef IteratorND                               self_type;
    typedef HANDLES                                  value_type;
    typedef ArrayIndex                               difference_type;
    typedef value_type &                             reference;
    typedef value_type const &                       const_reference;
    typedef value_type *                             pointer;
    typedef Shape<N>                                 shape_type;
    typedef std::random_access_iterator_tag          iterator_category;

    IteratorND() = default;

    template <int O = ORDER>
    explicit
    IteratorND(value_type const & handles,
               enable_if_t<O != runtime_order, bool> = true)
    : base_type(handles)
    {}

    template <int O = ORDER>
    explicit
    IteratorND(value_type const & handles, MemoryOrder order = C_ORDER,
               enable_if_t<O == runtime_order, bool> = true)
    : base_type(handles, order == C_ORDER
                             ? reversed(shape_type::range(handles.ndim()))
                             : shape_type::range(handles.ndim()))
    {}

    template <class SHAPE, int O = ORDER>
    explicit
    IteratorND(value_type const & handles, SHAPE const & order,
               enable_if_t<O == runtime_order, bool> = true)
    : base_type(handles, order)
    {}

    void inc()
    {
        base_type::operator++();
    }

    void dec()
    {
        base_type::operator--();
    }

    void inc(int dim)
    {
        handles_.inc(dim);
    }

    void dec(int dim)
    {
        handles_.dec(dim);
    }

    void move(int dim, ArrayIndex d)
    {
        handles_.move(dim, d);
    }

    void move(shape_type const & diff)
    {
        handles_.move(diff);
    }

    IteratorND & operator++()
    {
        base_type::operator++();
        return *this;
    }

    IteratorND operator++(int)
    {
        IteratorND res(*this);
        ++*this;
        return res;
    }

    IteratorND & operator+=(ArrayIndex i)
    {
        handles_.move(scanOrderToCoordinate(i+scanOrderIndex())-coord());
        return *this;
    }

    IteratorND & operator+=(shape_type const & coordOffset)
    {
        handles_.move(coordOffset);
        return *this;
    }

    IteratorND & operator--()
    {
        base_type::operator--();
        return *this;
    }

    IteratorND operator--(int)
    {
        IteratorND res(*this);
        --*this;
        return res;
    }

    IteratorND & operator-=(ArrayIndex i)
    {
        return operator+=(-i);
    }

    IteratorND & operator-=(shape_type const & coordOffset)
    {
        return operator+=(-coordOffset);
    }

    value_type operator[](ArrayIndex i) const
    {
        return *(IteratorND(*this) += i);
    }

    value_type operator[](shape_type const & coordOffset) const
    {
        return *(IteratorND(*this) += coordOffset);
    }

    IteratorND
    operator+(ArrayIndex d) const
    {
        return IteratorND(*this) += d;
    }

    IteratorND
    operator-(ArrayIndex d) const
    {
        return IteratorND(*this) -= d;
    }

    IteratorND operator+(const shape_type &coordOffset) const
    {
        return IteratorND(*this) += coordOffset;
    }

    IteratorND operator-(const shape_type &coordOffset) const
    {
        return IteratorND(*this) -= coordOffset;
    }

    ArrayIndex
    operator-(IteratorND const & r) const
    {
        return scanOrderIndex(coord() - r.coord());
    }

    bool operator==(IteratorND const & r) const
    {
        return !base_type::operator!=(r.handles_);
    }

    bool operator!=(IteratorND const & r) const
    {
        return base_type::operator!=(r.handles_);
    }

    bool operator<(IteratorND const & r) const
    {
        return (*this) - r < 0;
    }

    bool operator<=(IteratorND const & r) const
    {
        return (*this) - r <= 0;
    }

    bool operator>(IteratorND const & r) const
    {
        return (*this) - r > 0;
    }

    bool operator>=(IteratorND const & r) const
    {
        return (*this) - r >= 0;
    }

    bool isValid() const
    {
        return coord(minor_) < shape(minor_) && coord(minor_) >= 0;
    }

    bool atEnd() const
    {
        return coord(minor_) >= shape(minor_) || coord(minor_) < 0;
    }

    shape_type const & coord() const
    {
        return handles_.coord();
    }

    ArrayIndex coord(int dim) const
    {
        return handles_.coord(dim);
    }

    shape_type const & shape() const
    {
        return handles_.shape();
    }

    ArrayIndex shape(int dim) const
    {
        return handles_.shape(dim);
    }

    reference operator*()
    {
        return handles_;
    }

    const_reference operator*() const
    {
        return handles_;
    }

    template <int M = N>
    int ndim(enable_if_t<M == runtime_size, bool> = true) const
    {
        return handles_.ndim();
    }

    template <int M = N>
    constexpr
    int ndim(enable_if_t<M != runtime_size, bool> = true) const
    {
        return N;
    }

    // IteratorND &
    // restrictToSubarray(shape_type const & start, shape_type const & end)
    // {
        // operator+=(-coord());
        // handles_.restrictToSubarray(start, end);
        // strides_ = detail::defaultStride(shape());
        // return *this;
    // }

    IteratorND begin() const
    {
        IteratorND res(*this);
        res.move(-coord());
        return res;
    }

    IteratorND rbegin() const
    {
        IteratorND res(end());
        --res;
        return res;
    }

    IteratorND end() const
    {
        IteratorND res(*this);
        auto diff = -coord();
        diff[minor_] += shape(minor_);
        res.move(diff);
        return res;
    }

    IteratorND rend() const
    {
        IteratorND res(*this);
        res.move(-coord());
        --res;
        return res;
    }

    // bool atBorder() const
    // {
        // return (handles_.borderType() != 0);
    // }

    // unsigned int borderType() const
    // {
        // return handles_.borderType();
    // }

    reference handles()
    {
        return handles_;
    }

    const_reference handles() const
    {
        return handles_;
    }
};

/********************************************************/
/*                                                      */
/*                  CoordinateIterator                  */
/*                                                      */
/********************************************************/

    /** \brief Iterate over a virtual array where each element contains its coordinate.

        CoordinateIterator behaves like a read-only random access iterator.
        It moves accross the given region of interest in scan-order (with the first
        index changing most rapidly), and dereferencing the iterator returns the
        coordinate (i.e. multi-dimensional index) of the current array element.
        The functionality is thus similar to a meshgrid in Matlab or numpy.

        Internally, it is just a wrapper of a \ref IteratorND that
        has been created without any array and whose reference type is not a
        \ref CoupledHandle, but the coordinate itself.

        The iterator supports all functions listed in the STL documentation for
        <a href="http://www.sgi.com/tech/stl/RandomAccessIterator.html">Random Access Iterators</a>.

        <b>Usage:</b>

        <b>\#include</b> \<vigra/multi_iterator.hxx\><br/>
        Namespace: vigra

        \code
        CoordinateIterator<3> i(Shape3(3,2,1)), end = i.getEndIterator();

        for(; i != end; ++i)
            std::cout << *i << "\n";

        // Output:
        // (0, 0, 0)
        // (1, 0, 0)
        // (2, 0, 0)
        // (0, 1, 0)
        // (1, 1, 0)
        // (2, 1, 0)
        \endcode
    */
template <int N, int ORDER = runtime_order>
class CoordinateIterator
: public IteratorND<HandleNDChain<Shape<N>>, ORDER>
{
    static_assert(ORDER == runtime_order || ORDER == C_ORDER || ORDER == F_ORDER,
        "CoordinateIterator<N, ORDER>: Order must be one of runtime_order, C_ORDER, F_ORDER.");
  protected:

    typedef IteratorND<HandleNDChain<Shape<N>>, ORDER> base_type;
    typedef HandleNDChain<Shape<N>>                handle_type;

  public:
    typedef typename handle_type::value_type       value_type;
    typedef typename handle_type::reference        reference;
    typedef typename handle_type::const_reference  const_reference;
    typedef typename handle_type::pointer          pointer;
    typedef typename handle_type::const_pointer    const_pointer;
    typedef typename handle_type::shape_type       shape_type;
    typedef Shape<N>                               difference_type;
    typedef std::random_access_iterator_tag        iterator_category;

    CoordinateIterator() = default;

    template <int O = ORDER,
              VIGRA_REQUIRE<O == runtime_order> >
    explicit
    CoordinateIterator(shape_type const & shape,
                       MemoryOrder order = C_ORDER)
    : base_type(handle_type(shape), order)
    {}

    template <int O = ORDER,
              VIGRA_REQUIRE<O == runtime_order> >
    CoordinateIterator(shape_type const & shape,
                       shape_type const & order)
    : base_type(handle_type(shape), order)
    {}

    template <int O = ORDER,
              VIGRA_REQUIRE<O != runtime_order> >
    CoordinateIterator(shape_type const & shape)
    : base_type(handle_type(shape))
    {}

    // explicit CoordinateIterator(shape_type const & start, shape_type const & end)
    // : base_type(handle_type(end))
    // {
    //     this->restrictToSubarray(start, end);
    // }

    // template<class DirectedTag>
    // explicit CoordinateIterator(GridGraph<N, DirectedTag> const & g)
    // : base_type(handle_type(g.shape()))
    // {}

    // template<class DirectedTag>
    // explicit CoordinateIterator(GridGraph<N, DirectedTag> const & g,
    //                             typename GridGraph<N, DirectedTag>::Node const & node)
    // : base_type(handle_type(g.shape()))
    // {
    //     if( isInside(g,node))
    //         (*this)+=node;
    //     else
    //         *this=this->getEndIterator();
    // }


        // dereferencing the iterator yields the coordinate object
        // (used as vertex_descriptor)
    const_reference operator*() const
    {
        return this->coord();
    }

    operator value_type() const
    {
        return this->coord();
    }

    const_pointer operator->() const
    {
        return this->handles_.operator->();
    }

    value_type operator[](ArrayIndex i) const
    {
        return *(CoordinateIterator(*this) += i);
    }

    value_type operator[](shape_type const & coordOffset) const
    {
        return *(CoordinateIterator(*this) += coordOffset);
    }

    CoordinateIterator & operator++()
    {
        base_type::operator++();
        return *this;
    }

    CoordinateIterator operator++(int)
    {
        CoordinateIterator res(*this);
        ++*this;
        return res;
    }

    CoordinateIterator & operator+=(ArrayIndex i)
    {
        base_type::operator+=(i);
        return *this;
    }

    CoordinateIterator & operator+=(shape_type const & coordOffset)
    {
        base_type::operator+=(coordOffset);
        return *this;
    }

    CoordinateIterator & operator--()
    {
        base_type::operator--();
        return *this;
    }

    CoordinateIterator operator--(int)
    {
        CoordinateIterator res(*this);
        --*this;
        return res;
    }

    CoordinateIterator & operator-=(ArrayIndex i)
    {
        return operator+=(-i);
    }

    CoordinateIterator & operator-=(shape_type const & coordOffset)
    {
        return operator+=(-coordOffset);
    }

    CoordinateIterator begin() const
    {
       return CoordinateIterator(base_type::begin());
    }

    CoordinateIterator rbegin() const
    {
       return CoordinateIterator(base_type::rbegin());
    }

    CoordinateIterator end() const
    {
       return CoordinateIterator(base_type::end());
    }

    CoordinateIterator rend() const
    {
       return CoordinateIterator(base_type::rend());
    }

    CoordinateIterator operator+(ArrayIndex d) const
    {
        return CoordinateIterator(*this) += d;
    }

    CoordinateIterator operator-(ArrayIndex d) const
    {
        return CoordinateIterator(*this) -= d;
    }

    CoordinateIterator operator+(shape_type const & coordOffset) const
    {
        return CoordinateIterator(*this) += coordOffset;
    }

    CoordinateIterator operator-(shape_type const & coordOffset) const
    {
        return CoordinateIterator(*this) -= coordOffset;
    }

    ArrayIndex operator-(CoordinateIterator const & other) const
    {
        return base_type::operator-(other);
    }

  protected:
    CoordinateIterator(base_type const & base)
    : base_type(base)
    {}
};

/********************************************************/
/*                                                      */
/*                    ArrayNDIterator                   */
/*                                                      */
/********************************************************/

template <int N, class T, int ORDER = runtime_order>
class ArrayNDIterator
: public IteratorND<HandleNDChain<T, HandleNDChain<Shape<N>>>, ORDER>
{
    static_assert(ORDER == runtime_order || ORDER == C_ORDER || ORDER == F_ORDER,
        "CoordinateIterator<N, ORDER>: Order must be one of runtime_order, C_ORDER, F_ORDER.");
  protected:

    typedef HandleNDChain<T, ShapeHandle<N>>       handle_type;
    typedef HandleND<N, T>                         array_handle_type;
    typedef IteratorND<handle_type, ORDER>         base_type;

  public:
    typedef typename handle_type::value_type       value_type;
    typedef typename handle_type::reference        reference;
    typedef typename handle_type::const_reference  const_reference;
    typedef typename handle_type::pointer          pointer;
    typedef typename handle_type::const_pointer    const_pointer;
    typedef typename handle_type::shape_type       shape_type;
    typedef Shape<N>                               difference_type;
    typedef std::random_access_iterator_tag        iterator_category;

    ArrayNDIterator() = default;

    explicit
    ArrayNDIterator(ArrayViewND<N, T> const & array,
                    MemoryOrder order = C_ORDER)
    : base_type(handle_type(array.handle(), ShapeHandle<N>(array.shape())), order)
    {}

    ArrayNDIterator(ArrayViewND<N, T> const & array,
                    shape_type const & order)
    : base_type(handle_type(array.handle(), ShapeHandle<N>(array.shape())), order)
    {}

    reference operator*()
    {
        return *handles_;
    }

    const_reference operator*() const
    {
        return *handles_;
    }

    pointer operator->()
    {
        return this->handles_.operator->();
    }

    const_pointer operator->() const
    {
        return this->handles_.operator->();
    }

    value_type operator[](ArrayIndex i) const
    {
        return *(ArrayNDIterator(*this) += i);
    }

    value_type operator[](shape_type const & coordOffset) const
    {
        return *(ArrayNDIterator(*this) += coordOffset);
    }

    ArrayNDIterator & operator++()
    {
        base_type::operator++();
        return *this;
    }

    ArrayNDIterator operator++(int)
    {
        ArrayNDIterator res(*this);
        ++*this;
        return res;
    }

    ArrayNDIterator & operator+=(ArrayIndex i)
    {
        base_type::operator+=(i);
        return *this;
    }

    ArrayNDIterator & operator+=(shape_type const & coordOffset)
    {
        base_type::operator+=(coordOffset);
        return *this;
    }

    ArrayNDIterator & operator--()
    {
        base_type::operator--();
        return *this;
    }

    ArrayNDIterator operator--(int)
    {
        ArrayNDIterator res(*this);
        --*this;
        return res;
    }

    ArrayNDIterator & operator-=(ArrayIndex i)
    {
        return operator+=(-i);
    }

    ArrayNDIterator & operator-=(shape_type const & coordOffset)
    {
        return operator+=(-coordOffset);
    }

    ArrayNDIterator begin() const
    {
       return ArrayNDIterator(base_type::begin());
    }

    ArrayNDIterator rbegin() const
    {
       return ArrayNDIterator(base_type::rbegin());
    }

    ArrayNDIterator end() const
    {
       return ArrayNDIterator(base_type::end());
    }

    ArrayNDIterator rend() const
    {
       return ArrayNDIterator(base_type::rend());
    }

    ArrayNDIterator operator+(ArrayIndex d) const
    {
        return ArrayNDIterator(*this) += d;
    }

    ArrayNDIterator operator-(ArrayIndex d) const
    {
        return ArrayNDIterator(*this) -= d;
    }

    ArrayNDIterator operator+(shape_type const & coordOffset) const
    {
        return ArrayNDIterator(*this) += coordOffset;
    }

    ArrayNDIterator operator-(shape_type const & coordOffset) const
    {
        return ArrayNDIterator(*this) -= coordOffset;
    }

    ArrayIndex operator-(ArrayNDIterator const & other) const
    {
        return base_type::operator-(other);
    }

  protected:
    ArrayNDIterator(base_type const & base)
    : base_type(base)
    {}
};

/********************************************************/
/*                                                      */
/*                get<INDEX>(IteratorND)                */
/*                                                      */
/********************************************************/

template <int INDEX, class HANDLES, int ORDER>
auto
get(IteratorND<HANDLES, ORDER> const & i)
-> decltype(get<INDEX>(i.handles()))
{
    return get<INDEX>(i.handles());
}

template <int INDEX, class HANDLES, int ORDER>
auto
get(IteratorND<HANDLES, ORDER> & i)
-> decltype(get<INDEX>(i.handles()))
{
    return get<INDEX>(i.handles());
}

} // namespace vigra

#endif // VIGRA2_ITERATOR_ND_HXX
