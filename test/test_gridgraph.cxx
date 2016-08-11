/************************************************************************/
/*                                                                      */
/*              Copyright 2012-2013 by Ullrich Koethe                   */
/*                                                                      */
/*    This file is part of the VIGRA computer vision library.           */
/*    The VIGRA Website is                                              */
/*        http://hci.iwr.uni-heidelberg.de/vigra/                       */
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

#define VIGRA_CHECK_BOUNDS
#include "vigra2/unittest.hxx"
#include <vigra2/shape.hxx>
#include <vigra2/iterator_nd.hxx>
#include <vigra2/array_nd.hxx>
#include <vigra2/timing.hxx>
//#include <vigra2/gridgraph.hxx>
//#include <vigra2/multi_localminmax.hxx>
//#include <vigra2/algorithm.hxx>

#include <vector>
#include <numeric>

//#ifdef WITH_BOOST_GRAPH
//#  include <boost/graph/graph_concepts.hpp>
//#endif


using namespace vigra;

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#endif

template <int N>
struct NeighborhoodTests;

namespace vigra {

inline ArrayIndex gridGraphMaxDegree(unsigned int N, NeighborhoodType t)
{
    return t == DirectNeighborhood
        ? 2 * N
        : pow(3, (int)N) - 1;
}

template<int N>
class GridGraphArcDescriptor
: public Shape<(N == runtime_size) ? N : N + 1>
{
public:
    static const int dimension = (N == runtime_size) ? N : N + 1;
    typedef Shape<dimension>                                      base_type;
    typedef typename base_type::value_type                        value_type;
    typedef base_type                                             edge_coord_type;
    typedef ArrayIndex                                            index_type;
    typedef Shape<N>                                              shape_type;
    typedef decltype(((base_type*)0)->template subarray<0, 1>())  vertex_descriptor_view;

    GridGraphArcDescriptor()
        : is_reversed_(false)
    {}

    GridGraphArcDescriptor(lemon::Invalid)
    : base_type(tags::size = dimension, -1)
    , is_reversed_(false)
    {}

    GridGraphArcDescriptor(base_type const & b, bool reversed)
    : base_type(b)
    , is_reversed_(reversed)
    {}

    GridGraphArcDescriptor(shape_type const &vertex,
                           index_type edge_index,
                           bool reversed = false)
    : base_type(tags::size = vertex.size() + 1, DontInit)
    , is_reversed_(reversed)
    {
        for (int k = 0; k < vertex.size(); ++k)
            (*this)[k] = vertex[k];
        this->back() = edge_index;
    }

    void set(shape_type const & vertex, index_type edge_index, bool reversed)
    {
        GridGraphArcDescriptor(vertex, edge_index, reversed).swap(*this);
    }

    void increment(GridGraphArcDescriptor const & diff, bool opposite = false)
    {
        if (diff.is_reversed_)
        {
            is_reversed_ = !opposite;
            for (int k = 0; k < this->size() - 1; ++k)
                (*this)[k] += diff[k];
        }
        else
        {
            is_reversed_ = opposite;
        }
        this->back() = diff.back();
    }

    bool isReversed() const
    {
        return is_reversed_;
    }

    template <int M = N>
    enable_if_t<M != runtime_size, vertex_descriptor_view>
    vertexDescriptor() const
    {
        return this->template subarray<0, N>();
    }

    template <int M = N>
    enable_if_t<M == runtime_size, vertex_descriptor_view>
    vertexDescriptor() const
    {
        return this->subarray(0, this->size() - 1);
    }

    value_type edgeIndex() const
    {
        return this->back();
    }

protected:
    bool is_reversed_;
};

class Adjacency
{
  public:
    typedef ArrayIndex     node_descriptor_type;
    typedef ArrayIndex     edge_descriptor_type;
    typedef ArrayIndex     value_type;
    typedef ArrayIndex     index_type;

    Adjacency()
    : source_(0)
    , target_(0)
    , edge_(0)
    {}

    Adjacency(lemon::Invalid)
    : source_(0)
    , target_(0)
    , edge_(0)
    {}

    Adjacency(node_descriptor_type const & source, node_descriptor_type const & target = 0, edge_descriptor_type const & edge = 0)
    : source_(source)
    , target_(target)
    , edge_(edge)
    {}

    //CoordAdjacency(shape_type const &vertex,
    //    index_type edge_index,
    //    bool reversed = false)
    //    : base_type(tags::size = vertex.size() + 1, DontInit)
    //    , is_reversed_(reversed)
    //{
    //    for (int k = 0; k < vertex.size(); ++k)
    //        (*this)[k] = vertex[k];
    //    this->back() = edge_index;
    //}

    void set(node_descriptor_type const & source, node_descriptor_type const & target = 0, edge_descriptor_type const & edge = 0)
    {
        source_ = source;
        target_ = target;
        edge_ = edge;
    }

    void move(node_descriptor_type const & target, edge_descriptor_type const & edge, bool directed)
    {
        target_ = source_ + target;
        edge_ += edge;
    }

    node_descriptor_type source() const
    {
        return source_;
    }

    node_descriptor_type target() const
    {
        return target_;
    }


    node_descriptor_type node() const
    {
        return target();
    }

    edge_descriptor_type edge() const
    {
        return edge_;
    }

protected:
    node_descriptor_type source_, target_;
    edge_descriptor_type edge_;
};

template<int N>
class CoordAdjacency
{
public:
    static const int dimension = (N == runtime_size) ? N : N + 1;
    typedef Shape<N>                      node_descriptor_type;
    typedef TinyArrayView<ArrayIndex, N>  node_descriptor_view;
    typedef Shape<dimension>              edge_descriptor_type;
    typedef ArrayIndex                    value_type;
    typedef ArrayIndex                    index_type;

    CoordAdjacency()
        : arc_index_(0)
    {}

    CoordAdjacency(lemon::Invalid)
        : arc_(lemon::INVALID)
        , reverse_arc_(lemon::INVALID)
        , arc_index_(0)
    {}

    CoordAdjacency(node_descriptor_type const & source)
        : arc_(tags::size = source.size() + 1)
        , reverse_arc_(tags::size = source.size() + 1)
        , arc_index_(0)
    {
        arc_.subarray(0, source.size()) = source;
        reverse_arc_.subarray(0, source.size()) = source;
    }

    CoordAdjacency(edge_descriptor_type const & arc, edge_descriptor_type const & reverse, bool directed)
        : arc_(arc)
        , reverse_arc_(reverse)
        , arc_index_((directed || arc.back() < reverse.back()) ? 0 : 1)
    {}

    //CoordAdjacency(shape_type const &vertex,
    //    index_type edge_index,
    //    bool reversed = false)
    //    : base_type(tags::size = vertex.size() + 1, DontInit)
    //    , is_reversed_(reversed)
    //{
    //    for (int k = 0; k < vertex.size(); ++k)
    //        (*this)[k] = vertex[k];
    //    this->back() = edge_index;
    //}

    void set(edge_descriptor_type const & arc, edge_descriptor_type const & reverse, bool directed)
    {
        arc_ = arc;
        reverse_arc_ = reverse;
        arc_index_ = (directed || arc.back() < reverse.back()) ? 0 : 1;
    }

    void move(ArrayIndex edge_index, edge_descriptor_type const & diff, bool directed)
    {
        arc_.back() = edge_index;
        reverse_arc_ += diff;
        arc_index_ = (directed || arc_.back() < reverse_arc_.back()) ? 0 : 1;
    }

    template <int M = N>
    int ndim(enable_if_t<M == runtime_size, bool> = true) const
    {
        return arc_.size() - 1;
    }

    template <int M = N>
    constexpr int ndim(enable_if_t<(M > runtime_size), bool> = true) const
    {
        return N;
    }

    template <int M = N>
    enable_if_t<M != runtime_size, node_descriptor_view>
        source() const
    {
        return arc_.template subarray<0, N>();
    }

    template <int M = N>
    enable_if_t<M == runtime_size, node_descriptor_view>
        source() const
    {
        return arc_.subarray(0, ndim());
    }

    template <int M = N>
    enable_if_t<M != runtime_size, node_descriptor_view>
        target() const
    {
        return reverse_arc_.template subarray<0, N>();
    }

    template <int M = N>
    enable_if_t<M == runtime_size, node_descriptor_view>
        target() const
    {
        return reverse_arc_.subarray(0, ndim());
    }

    node_descriptor_view node() const
    {
        return target();
    }

    edge_descriptor_type const & arc() const
    {
        return arc_;
    }

    edge_descriptor_type const & reverseArc() const
    {
        return reverse_arc_;
    }

    edge_descriptor_type const & edge() const
    {
        return (&arc_)[arc_index_];
    }

    value_type edgeIndex() const
    {
        return arc_.back();
    }

  //protected:
    edge_descriptor_type arc_, reverse_arc_;
    int arc_index_;
};

template <class Shape>
ArrayIndex
gridGraphEdgeCount(Shape const & shape, NeighborhoodType t, bool directed)
{
    int res = 0;
    if (t == DirectNeighborhood)
    {
        for (unsigned int k = 0; k<shape.size(); ++k)
            res += 2 * prod(shape - Shape::unitVector(tags::size=shape.size(), k));
    }
    else
    {
        res = prod(3 * shape - 2) - prod(shape);
    }
    return directed
        ? res
        : res / 2;
}

template <class NODE, int N>
inline ArrayIndex
nodeIdFromCoord(NODE const & node, Shape<N> const & strides)
{
    return dot(node, strides);
}

template <int N>
inline Shape<N> 
nodeCoordFromId(ArrayIndex node, Shape<N> const & shape, MemoryOrder order)
{
    Shape<N> res(shape.size(), DontInit);
    if (order == F_ORDER)
    {
         for(int k=0; k<shape.size(); ++k)
         {
            res[k] = (node % shape[k]);
            node /= shape[k];
         }
    }
    else
    {
        for (int k=shape.size()-1; k >= 0; --k)
        {
            res[k] = (node % shape[k]);
            node /= shape[k];
        }
    }
    return res;
}

//template <int M, int N>
//inline ArrayIndex
//edgeIdFromCoord(Shape<M + 1> const & edge, Shape<M> const & shape, 
//    std::vector<Shape<M>> const & offsets, Shape<N> const & edge_offsets, MemoryOrder order)
//{
//    auto p = edge.template subarray<0, M>() + min(offsets[edge.back()], 0);
//    auto s = shapeToStrides(shape - abs(offsets[edge.back()]), order);
//    //std::cerr << p << " " << s << " " << edge << " " << edge_offsets << "\n";
//    return dot(p, s) + edge_offsets[edge.back()];
//}

template <int M, int N>
inline ArrayIndex
edgeIdFromCoord(Shape<M + 1> const & edge, Shape<M> const & strides, Shape<N> const & edge_offsets)
{
    return dot(edge.template subarray<0, M>(), strides) + edge_offsets[edge.back()];
}

template <int N>
inline ArrayIndex 
edgeIdFromCoord(Shape<runtime_size> const & edge, Shape<runtime_size> const & strides, Shape<N> const & edge_offsets)
{
    return dot(edge.subarray(0, strides.size()), strides) + edge_offsets[edge.back()];
}

template <int M, int N>
inline Shape<M + 1>
edgeCoordFromId(ArrayIndex edge, std::vector<Shape<M>> const & shapes, MemoryOrder order,
    Shape<N> const & edge_offsets, Shape<N> const & edge_offsets2)
{
    Shape<M + 1> res(DontInit);
    for (int k = 0; k < edge_offsets.size(); ++k)
    {
        if (edge < edge_offsets[k])
            break;
        res.back() = k;
    }

    res.template subarray<0, M>() = nodeCoordFromId(edge - edge_offsets[res.back()], shapes[res.back()], order);
    return res;
}

template <int M, int N>
inline Shape<M + 1>
edgeCoordFromId2(ArrayIndex edge, std::vector<Shape<M>> const & shapes, MemoryOrder order,
    Shape<N> const & edge_offsets, std::vector<Shape<M>> const & pivots)
{
    Shape<M + 1> res(DontInit);
    for (int k = 0; k < edge_offsets.size(); ++k)
    {
        if (edge < edge_offsets[k])
            break;
        res.back() = k;
    }

    res.template subarray<0, M>() = nodeCoordFromId(edge - edge_offsets[res.back()], shapes[res.back()], order) - 
        pivots[res.back()];
    return res;
}

template <int N>
inline Shape<runtime_size>
edgeCoordFromId(ArrayIndex edge, std::vector<Shape<runtime_size>> const & shapes, MemoryOrder order,
    Shape<N> const & edge_offsets, Shape<N> const & edge_offsets2)
{
    int size = shapes[0].size();
    Shape<runtime_size> res(size + 1, DontInit);
    for (int k = 0; k < edge_offsets.size(); ++k)
    {
        if (edge < edge_offsets[k])
            break;
        res.back() = k;
    }

    res.subarray(0, size) = nodeCoordFromId(edge - edge_offsets[res.back()], shapes[res.back()], order);
    return res;
}

template <int N>
inline Shape<runtime_size>
edgeCoordFromId2(ArrayIndex edge, std::vector<Shape<runtime_size>> const & shapes, MemoryOrder order,
    Shape<N> const & edge_offsets, std::vector<Shape<runtime_size>> const & pivots)
{
    int size = shapes[0].size();
    Shape<runtime_size> res(size + 1, DontInit);
    for (int k = 0; k < edge_offsets.size(); ++k)
    {
        if (edge < edge_offsets[k])
            break;
        res.back() = k;
    }

    res.subarray(0, size) = nodeCoordFromId(edge - edge_offsets[res.back()], shapes[res.back()], order) -
        pivots[res.back()];
    return res;
}

template <int N>
inline Shape<runtime_size>
edgeOffsets(Shape<N> shape, std::vector<Shape<N>> const & neighborOffsets, bool directed)
{
    shape = max(shape, Shape<N>(tags::size=shape.size(), 1));
    int degree = directed 
                     ? neighborOffsets.size() 
                     : neighborOffsets.size() / 2;
    Shape<runtime_size> offsets(degree, DontInit);
    offsets[0] = 0;
    for (int k = 0; k <degree-1; ++k)
        offsets[k + 1] = offsets[k] + prod(shape - abs(neighborOffsets[k]));
    return offsets;
}


namespace detail {

// Create the list of neighbor offsets for the given neighborhood type
// and dimension (the dimension is implicitly defined by the Shape type)
// an return it in 'neighborOffsets'. Moreover, create a list of flags
// for each BorderType that is 'true' when the corresponding neighbor exists
// in this border situation and return the result in 'neighborExists'.
template <class SHAPE>
void
makeArrayNeighborhoodOld(
    unsigned int ndim, NeighborhoodType neighborhoodType, MemoryOrder order,
    std::vector<SHAPE> & neighborOffsets,
    std::vector<std::vector<bool>> & neighborExists,
    std::vector<std::vector<ArrayIndex>> & validNeighborIndices)
{
    neighborOffsets.clear();
    if (neighborhoodType == DirectNeighborhood)
    {
        SHAPE axes = (order == F_ORDER)
            ? reversed(SHAPE::range(ndim))
            : SHAPE::range(ndim);
        for (int k = 0; k < (int)ndim; ++k)
            neighborOffsets.push_back(-SHAPE::unitVector(tags::size = ndim, axes[k]));
        for (int k = (int)ndim - 1; k >= 0; --k)
            neighborOffsets.push_back(SHAPE::unitVector(tags::size = ndim, axes[k]));
    }
    else
    {
        CoordinateIterator<SHAPE::static_size> c(SHAPE(tags::size = ndim, 3), order);
        for (; c.isValid(); ++c)
        {
            if (*c == 1)
                continue;
            neighborOffsets.push_back(*c - 1);
        }
    }

    unsigned int borderTypeCount = 1 << 2 * ndim,
                 degree = gridGraphMaxDegree(ndim, neighborhoodType);
    neighborExists.clear();
    validNeighborIndices.resize(borderTypeCount);
    for (unsigned int bt = 0; bt < borderTypeCount; ++bt)
    {
        std::vector<bool> exists(degree);
        validNeighborIndices[bt].clear();
        for (unsigned int o = 0; o < degree; ++o)
        {
            bool inside = true;
            for (unsigned int k = 0; k < ndim; ++k)
            {
                if ((neighborOffsets[o][k] < 0 && (bt & (1 << 2 * k)) != 0) ||
                    (neighborOffsets[o][k] > 0 && (bt & (2 << 2 * k)) != 0))
                {
                    inside = false;
                    break;
                }
            }
            exists[o] = inside;
            if (inside)
                validNeighborIndices[bt].push_back(o);
        }
        neighborExists.emplace_back(std::move(exists));
    }
}

template <class SHAPE>
void
makeNeighborhoodND(
    unsigned int ndim, NeighborhoodType neighborhoodType, MemoryOrder order,
    std::vector<SHAPE> & neighborOffsets)
{
    neighborOffsets.clear();
    if (neighborhoodType == DirectNeighborhood)
    {
        SHAPE axes = (order == F_ORDER)
            ? reversed(SHAPE::range(ndim))
            : SHAPE::range(ndim);
        for (int k = 0; k < (int)ndim; ++k)
            neighborOffsets.push_back(-SHAPE::unitVector(tags::size = ndim, axes[k]));
        for (int k = (int)ndim - 1; k >= 0; --k)
            neighborOffsets.push_back(SHAPE::unitVector(tags::size = ndim, axes[k]));
    }
    else
    {
        CoordinateIterator<SHAPE::static_size> c(SHAPE(tags::size = ndim, 3), order);
        for (; c.isValid(); ++c)
        {
            if (*c == 1)
                continue;
            neighborOffsets.push_back(*c - 1);
        }
    }
}

template <class SHAPE>
void
makeBorderNeighborhoodND(
    std::vector<SHAPE> const & neighborOffsets,
    std::vector<std::vector<ArrayIndex>> & validNeighborIndices,
    std::vector<std::vector<ArrayIndex>> & backwardNeighborIndices)
{
    unsigned int ndim = neighborOffsets[0].size(),
                 borderTypeCount = 1 << 2 * ndim,
                 degree = neighborOffsets.size();
    validNeighborIndices.resize(borderTypeCount);
    backwardNeighborIndices.resize(borderTypeCount);
    for (unsigned int bt = 0; bt < borderTypeCount; ++bt)
    {
        validNeighborIndices[bt].clear();
        backwardNeighborIndices[bt].clear();
        for (unsigned int o = 0; o < degree; ++o)
        {
            bool inside = true;
            for (unsigned int k = 0; k < ndim; ++k)
            {
                if ((neighborOffsets[o][k] < 0 && (bt & (1 << 2 * k)) != 0) ||
                    (neighborOffsets[o][k] > 0 && (bt & (2 << 2 * k)) != 0))
                {
                    inside = false;
                    break;
                }
            }
            if (inside)
            {
                validNeighborIndices[bt].push_back(o);
                if (o < degree / 2)
                    backwardNeighborIndices[bt].push_back(o);
            }
        }
    }
}

template <class SHAPE, class ARC_DESCRIPTOR>
void
computeNeighborIncrements(
    std::vector<SHAPE> const & neighborOffsets,
    std::vector<std::vector<ArrayIndex> > const & indices,
    std::vector<std::vector<SHAPE> > & adjacentNodeIncrements,
    std::vector<std::vector<GridGraphArcDescriptor<SHAPE::static_size>>> & adjacentArcIncrements,
    std::vector<std::vector<ARC_DESCRIPTOR>> & adjacencyIncrements,
    bool directed)
{
    typedef GridGraphArcDescriptor<SHAPE::static_size> ArcDescriptor;

    int borderTypeCount = indices.size(),
        maxDegree = neighborOffsets.size(),
        ndim = neighborOffsets[0].size();
    SHAPE zero(tags::size = ndim);
    adjacentNodeIncrements.resize(borderTypeCount);
    adjacentArcIncrements.resize(borderTypeCount);
    adjacencyIncrements.resize(borderTypeCount);

    for (int bt = 0; bt<borderTypeCount; ++bt)
    {
        adjacentNodeIncrements[bt].clear();
        adjacentArcIncrements[bt].clear();
        adjacencyIncrements[bt].clear();

        int degree = (int)indices[bt].size();

        for (int k = 0; k < degree; ++k)
        {
            ArrayIndex j = indices[bt][k];
            if (k == 0)
            {
                adjacentNodeIncrements[bt].push_back(neighborOffsets[j]);
                ARC_DESCRIPTOR arc(ndim + 1, DontInit);
                arc.subarray(0, ndim) = neighborOffsets[j];
                arc.back() = maxDegree - j - 1;
                adjacencyIncrements[bt].push_back(arc);
            }
            else
            {
                adjacentNodeIncrements[bt].push_back(neighborOffsets[j] - neighborOffsets[indices[bt][k - 1]]);
                ARC_DESCRIPTOR diff(ndim + 1, DontInit);
                diff.subarray(0, ndim) = adjacentNodeIncrements[bt].back();
                diff.back() = indices[bt][k - 1] - j;
                adjacencyIncrements[bt].push_back(diff);
            }

            if (directed || j < maxDegree / 2) // directed or backward edge
            {
                adjacentArcIncrements[bt].push_back(ArcDescriptor(zero, j));
            }
            else if (k == 0 || !adjacentArcIncrements[bt].back().isReversed()) // the first forward edge
            {
                adjacentArcIncrements[bt].push_back(ArcDescriptor(neighborOffsets[j], maxDegree - j - 1, true));
            }
            else // second or higher forward edge
            {
                adjacentArcIncrements[bt].push_back(ArcDescriptor(neighborOffsets[j] - neighborOffsets[indices[bt][k - 1]],
                    maxDegree - j - 1, true));
            }
        }
    }
}

} // namespace detail

template<int N>
class GridGraphNeighborIterator
{
public:
    typedef Shape<N>                                   shape_type;
    typedef CoordinateIterator<N>                      vertex_iterator;
    typedef typename vertex_iterator::value_type       vertex_descriptor;
    typedef vertex_descriptor                          value_type;
    typedef typename vertex_iterator::pointer          pointer;
    typedef typename vertex_iterator::const_pointer    const_pointer;
    typedef typename vertex_iterator::reference        reference;
    typedef typename vertex_iterator::const_reference  const_reference;
    typedef ArrayIndex                                 difference_type;
    typedef ArrayIndex                                 index_type;
    typedef std::forward_iterator_tag                  iterator_category;

    friend struct NeighborhoodTests<N>;

    GridGraphNeighborIterator()
    : neighborOffsets_(0)
    , neighborIndices_(0)
    , index_(0)
    {}

    //template <class DirectedTag>
    //GridGraphNeighborIterator(GridGraph<N, DirectedTag> const & g, typename GridGraph<N, DirectedTag>::Node const & v)
    //    : neighborOffsets_(0),
    //    neighborIndices_(0),
    //    target_(v),
    //    index_(0)
    //{
    //    unsigned int nbtype = g.get_border_type(v);
    //    neighborOffsets_ = &(*g.neighborIncrementArray())[nbtype];
    //    neighborIndices_ = &(*g.neighborIndexArray(BackEdgesOnly))[nbtype];
    //    updateTarget();
    //}

    //template <class DirectedTag>
    //GridGraphNeighborIterator(GridGraph<N, DirectedTag> const & g, typename GridGraph<N, DirectedTag>::NodeIt const & v)
    //    : neighborOffsets_(0),
    //    neighborIndices_(0),
    //    target_(v),
    //    index_(0)
    //{
    //    unsigned int nbtype = g.get_border_type(v);
    //    neighborOffsets_ = &(*g.neighborIncrementArray())[nbtype];
    //    neighborIndices_ = &(*g.neighborIndexArray(BackEdgesOnly))[nbtype];
    //    updateTarget();
    //}

    // TODO: implement a "goto-neighbor" operation
    // yielding a vertex_iterator! -> useful for
    // watershed algo.

    GridGraphNeighborIterator & operator++()
    {
        ++index_;
        updateTarget();
        return *this;
    }

    GridGraphNeighborIterator operator++(int)
    {
        GridGraphNeighborIterator ret(*this);
        ++*this;
        return ret;
    }

    const_reference operator*() const
    {
        return target_;
    }

    const_pointer operator->() const
    {
        return &target_;
    }

    operator const_reference() const
    {
        return target_;
    }

    const_reference target() const
    {
        return target_;
    }

    ArrayIndex index() const
    {
        return index_;
    }

    ArrayIndex neighborIndex() const
    {
        return (*neighborIndices_)[index_];
    }

    ArrayIndex degree() const
    {
        return (ArrayIndex)neighborIndices_->size();
    }

    bool operator==(GridGraphNeighborIterator const & other) const
    {
        return index_ == other.index_;
    }

    bool operator!=(GridGraphNeighborIterator const & other) const
    {
        return index_ != other.index_;
    }

    bool isValid() const
    {
        return index_ < degree();
    }

    bool atEnd() const
    {
        return index_ >= degree();
    }

    GridGraphNeighborIterator end() const
    {
        GridGraphNeighborIterator res(*this);
        res.index_ = degree();
        return res;
    }

protected:

    // for testing only
    GridGraphNeighborIterator(std::vector<shape_type> const & neighborOffsets,
                              std::vector<index_type> const & neighborIndices,
                              vertex_descriptor const & source)
    : neighborOffsets_(&neighborOffsets)
    , neighborIndices_(&neighborIndices)
    , target_(source)
    , index_(0)
    {
        updateTarget();
    }

    void updateTarget()
    {
        if (isValid())
            target_ += (*neighborOffsets_)[index_];
    }

    std::vector<shape_type> const * neighborOffsets_;
    std::vector<index_type> const * neighborIndices_;
    vertex_descriptor target_;
    ArrayIndex index_;
};

template<int N>
class GridGraphAdjacencyIterator
{
public:
    typedef CoordAdjacency<N>                          value_type;
    typedef typename value_type::node_descriptor_type  node_descriptor_type;
    typedef typename value_type::edge_descriptor_type  edge_descriptor_type;
    typedef value_type *                               pointer;
    typedef value_type const *                         const_pointer;
    typedef value_type &                               reference;
    typedef value_type const &                         const_reference;
    typedef ArrayIndex                                 difference_type;
    typedef ArrayIndex                                 index_type;
    typedef std::forward_iterator_tag                  iterator_category;

    friend struct NeighborhoodTests<N>;

    GridGraphAdjacencyIterator()
    : adjacencyOffsets_(0)
    , neighborIndices_(0)
    , index_(0)
    {}

    //template <class DirectedTag>
    //GridGraphNeighborIterator(GridGraph<N, DirectedTag> const & g, typename GridGraph<N, DirectedTag>::Node const & v)
    //    : neighborOffsets_(0),
    //    neighborIndices_(0),
    //    target_(v),
    //    index_(0)
    //{
    //    unsigned int nbtype = g.get_border_type(v);
    //    neighborOffsets_ = &(*g.neighborIncrementArray())[nbtype];
    //    neighborIndices_ = &(*g.neighborIndexArray(BackEdgesOnly))[nbtype];
    //    updateTarget();
    //}

    //template <class DirectedTag>
    //GridGraphNeighborIterator(GridGraph<N, DirectedTag> const & g, typename GridGraph<N, DirectedTag>::NodeIt const & v)
    //    : neighborOffsets_(0),
    //    neighborIndices_(0),
    //    target_(v),
    //    index_(0)
    //{
    //    unsigned int nbtype = g.get_border_type(v);
    //    neighborOffsets_ = &(*g.neighborIncrementArray())[nbtype];
    //    neighborIndices_ = &(*g.neighborIndexArray(BackEdgesOnly))[nbtype];
    //    updateTarget();
    //}

    // TODO: implement a "goto-neighbor" operation
    // yielding a vertex_iterator! -> useful for
    // watershed algo.

    GridGraphAdjacencyIterator & operator++()
    {
        ++index_;
        updateAdjacency();
        return *this;
    }

    GridGraphAdjacencyIterator operator++(int)
    {
        GridGraphAdjacencyIterator ret(*this);
        ++*this;
        return ret;
    }

    const_reference operator*() const
    {
        return adjacency_;
    }

    const_pointer operator->() const
    {
        return &adjacency_;
    }

    ArrayIndex index() const
    {
        return index_;
    }

    ArrayIndex neighborIndex() const
    {
        return (*neighborIndices_)[index_];
    }

    ArrayIndex degree() const
    {
        return (ArrayIndex)neighborIndices_->size();
    }

    bool operator==(GridGraphAdjacencyIterator const & other) const
    {
        return index_ == other.index_;
    }

    bool operator!=(GridGraphAdjacencyIterator const & other) const
    {
        return index_ != other.index_;
    }

    bool isValid() const
    {
        return index_ < degree();
    }

    bool atEnd() const
    {
        return index_ >= degree();
    }

    GridGraphAdjacencyIterator end() const
    {
        GridGraphAdjacencyIterator res(*this);
        res.index_ = degree();
        return res;
    }

protected:

    // for testing only
    GridGraphAdjacencyIterator(std::vector<edge_descriptor_type> const & adjacencyOffsets,
                               std::vector<index_type> const & neighborIndices,
                               node_descriptor_type const & source,
                               bool directed = false)
    : adjacencyOffsets_(&adjacencyOffsets)
    , neighborIndices_(&neighborIndices)
    , adjacency_(source)
    , directed_(directed)
    , index_(0)
    {
        updateAdjacency();
    }

    void updateAdjacency()
    {
        if (isValid())
            adjacency_.move((*neighborIndices_)[index_], (*adjacencyOffsets_)[index_], directed_);
    }

    std::vector<edge_descriptor_type> const * adjacencyOffsets_;
    std::vector<index_type> const * neighborIndices_;
    value_type adjacency_;
    bool directed_;
    ArrayIndex index_;
};

#if 0
template <int N>
class GridGraphBase
{
public:
    typedef GridGraphBase<N>               self_type;

    /** \brief Dimension of the grid.
    */
    static const int dimension = N;

    /** \brief Shape type of the graph and a node property map.
    */
    using shape_type = Shape<N>;

    using coord_adjacency_type = CoordAdjacency<N>;
    using node_descriptor_type = typename coord_adjacency_type::node_descriptor_type;
    using edge_descriptor_type = typename coord_adjacency_type::edge_descriptor_type;

    /** \brief Shape type of an edge property map (must have one additional dimension).
    */
    typedef typename MultiArrayShape<N + 1>::type     edge_propmap_shape_type;

    /** \brief Type of node and edge IDs.
    */
    typedef ArrayIndex                         index_type;

    ////////////////////////////////////////////////////////////////////

    // dummy default constructor to satisfy adjacency_graph concept
    GridGraphBase()
        : max_node_id_(-1),
        max_arc_id_(-1),
        max_edge_id_(-1)
    {}

    /** \brief Construct a grid graph with given \a shape and neighborhood type \a ntype.

    The shape must have type <tt>MultiArrayShape<N>::type</tt> with the appropriate
    dimension <tt>N</tt>. The neighborhood type can take the values
    <tt>DirectNeighborhood</tt> to use only the axis-aligned edges (2N-neighborhood)
    and <tt>IndirectNeighborhood</tt> to use all diagonal edges as well
    ((3<sup>N</sup>-1)-neighborhood).
    */
    GridGraphBase(shape_type const &shape, 
                  NeighborhoodType ntype = DirectNeighborhood, 
                  tags::DirectedProxy directed = { false },
                  MemoryOrder order = C_ORDER)
    : shape_(shape)
    , num_vertices_(prod(shape))
    , num_edges_(gridGraphEdgeCount(shape, ntype, directed.value))
    , max_node_id_(num_vertices_ - 1)
    , max_arc_id_(-2)
    , max_edge_id_(-2)
    , neighborhoodType_(ntype)
    {
        // populate the neighborhood tables:
        // FIXME: this might be static (but make sure that it works with multi-threading)
        detail::makeArrayNeighborhood(neighborOffsets_, neighborExists_, neighborhoodType_);
        detail::computeNeighborOffsets(neighborOffsets_, neighborExists_, incrementalOffsets_,
            edgeDescriptorOffsets_, neighborIndices_, backIndices_, is_directed);

        // compute the neighbor offsets per neighborhood type
        // detail::makeArraySubNeighborhood(neighborhood[0], neighborExists, shape_type(1), neighborhoodIndices);
    }

    /** \brief Get the ID (i.e. scan-order index) for node desciptor \a v (API: LEMON).
    */
    index_type id(Node const & v) const
    {
        return detail::CoordinateToScanOrder<N>::exec(shape(), v);
    }

    index_type id(NodeIt const & v) const
    {
        return v.scanOrderIndex();
    }

    index_type id(neighbor_vertex_iterator const & v) const
    {
        return id(*v);
    }

    index_type id(back_neighbor_vertex_iterator const & v) const
    {
        return id(*v);
    }

    /** \brief Get node descriptor for given node ID \a i (API: LEMON).

    Return <tt>Node(lemon::INVALID)</tt> when the ID does not exist in this graph.
    */
    Node nodeFromId(index_type i) const
    {
        if (i < 0 || i > maxNodeId())
            return Node(lemon::INVALID);

        Node res(SkipInitialization);
        detail::ScanOrderToCoordinate<N>::exec(i, shape(), res);
        return res;
    }

    /** \brief Get the maximum ID of any node in this graph (API: LEMON).
    */
    index_type maxNodeId() const
    {
        return prod(shape()) - 1;
    }

    /** \brief Get the grid cordinate of the given node \a v (convenience function).
    */
    Node const & pos(Node const & v) const
    {
        return v;
    }

    /** \brief Get vertex iterator pointing to the first vertex of this graph (convenience function,<br/>
    the boost::graph API provides the free function <tt>boost::vertices(graph)</tt>,<br/>
    LEMON uses <tt>Graph::NodeIt(graph)</tt>).
    */
    vertex_iterator get_vertex_iterator() const
    {
        return vertex_iterator(shape_);
    }

    /** \brief Get vertex iterator pointing to the given vertex (API: VIGRA).
    */
    vertex_iterator get_vertex_iterator(vertex_descriptor const & v) const
    {
        return vertex_iterator(shape_) + v;
    }

    /** \brief Get vertex iterator pointing beyond the valid range of vertices of this graph (convenience function,<br/>
    the boost::graph API provides the free function <tt>boost::vertices(graph)</tt>,<br/>
    LEMON uses the special value <tt>lemon::INVALID</tt> instead).
    */
    vertex_iterator get_vertex_end_iterator() const
    {
        return get_vertex_iterator().getEndIterator();
    }

    /** \brief Get an iterator pointing to the first neighbor of the given vertex (convenience function,<br/>
    the boost::graph API provides the free function <tt>boost::adjacent_vertices(v, graph)</tt>,<br/>
    LEMON uses <tt>Graph::ArcIt(g, v)</tt>).
    */
    neighbor_vertex_iterator get_neighbor_vertex_iterator(vertex_descriptor const & v) const
    {
        return neighbor_vertex_iterator(*this, v);
    }

    /** \brief Get an iterator pointing beyond the range of neighbors of the given vertex (convenience function,<br/>
    the boost::graph API provides the free function <tt>boost::adjacent_vertices(v, graph)</tt>,<br/>
    LEMON uses the speical value <tt>lemon::INVALID</tt> instead).
    */
    neighbor_vertex_iterator get_neighbor_vertex_end_iterator(vertex_descriptor const & v) const
    {
        return get_neighbor_vertex_iterator(v).getEndIterator();
    }

    /** \brief Get an iterator pointing to the first backward neighbor of the given vertex (API: VIGRA,<br/>
    in analogy to the boost::graph API, we also provide a free function <tt>boost::back_adjacent_vertices(v, g)</tt>,<br/>
    and the LEMON analogue is <tt>Graph::OutBackArcIt(graph, v)</tt>).
    */
    back_neighbor_vertex_iterator get_back_neighbor_vertex_iterator(vertex_descriptor const & v) const
    {
        return back_neighbor_vertex_iterator(*this, v);
    }

    /** \brief Get an iterator pointing beyond the range of backward neighbors of the given vertex (API: VIGRA,<br/>
    in analogy to the boost::graph API, we also provide a free function <tt>boost::back_adjacent_vertices(v, g)</tt>,<br/>
    and LEMON just uses <tt>lemon::INVALID</tt> instead).
    */
    back_neighbor_vertex_iterator get_back_neighbor_vertex_end_iterator(vertex_descriptor const & v) const
    {
        return get_back_neighbor_vertex_iterator(v).getEndIterator();
    }

    // --------------------------------------------------
    // support for VertexListGraph:

    /** \brief Get the number of vertices in this graph (convenience function,
    the boost::graph API provides the free function <tt>boost::num_vertices(graph)</tt>).
    */
    vertices_size_type num_vertices() const
    {
        return num_vertices_;
    }

    /** \brief Get the number of nodes in this graph (API: LEMON).
    */
    vertices_size_type nodeNum() const
    {
        return num_vertices();
    }

    // --------------------------------------------------
    // support for IncidenceGraph:

    /** \brief Get the ID (i.e. scan-order index in an edge property map) for the
    given edges descriptor \a e (API: LEMON).
    */
    index_type id(Edge const & e) const
    {
        return detail::CoordinateToScanOrder<N + 1>::exec(edge_propmap_shape(), e);
    }

    index_type id(EdgeIt const & e) const
    {
        return id(*e);
    }

    index_type id(IncEdgeIt const & e) const
    {
        return id(*e);
    }

    index_type id(IncBackEdgeIt const & e) const
    {
        return id(*e);
    }

    /** \brief Get the edge descriptor for the given edge ID \a i (API: LEMON).

    Return <tt>Edge(lemon::INVALID)</tt> when the ID does not exist
    in this graph.
    */
    Edge edgeFromId(index_type i) const
    {
        if (i < 0 || i > maxEdgeId())
            return Edge(lemon::INVALID);

        Edge res(SkipInitialization);
        detail::ScanOrderToCoordinate<N + 1>::exec(i, edge_propmap_shape(), res);

        unsigned int b = detail::BorderTypeImpl<N>::exec(res.template subarray<0, N>(), shape());
        if (neighborExists_[b][res[N]])
            return res;
        else
            return Edge(lemon::INVALID);
    }

    /** \brief Get the maximum ID of any edge in this graph (API: LEMON).
    */
    index_type maxEdgeId() const
    {
        if (max_edge_id_ == -2) // -2 means uninitialized
            const_cast<GridGraph *>(this)->computeMaxEdgeAndArcId();
        return max_edge_id_;
    }

    /* Initial computation of the max_arc_id_ and max_edge_id_ (call in the constructor and
    whenever the shape changes).
    */
    void computeMaxEdgeAndArcId()
    {
        if (edgeNum() == 0)
        {
            max_arc_id_ = -1;
            max_edge_id_ = -1;
        }
        else
        {
            Node lastNode = shape() - shape_type(1);
            index_type n = neighborIndices_[get_border_type(lastNode)][0];
            Arc a(neighbor(lastNode, n), oppositeIndex(n), false);
            max_arc_id_ = detail::CoordinateToScanOrder<N + 1>::exec(arc_propmap_shape(), a);

            if (is_directed)
            {
                max_edge_id_ = max_arc_id_;
            }
            else
            {
                Arc a(lastNode, backIndices_[get_border_type(lastNode)].back(), false);
                max_edge_id_ = detail::CoordinateToScanOrder<N + 1>::exec(edge_propmap_shape(), a);
            }
        }
    }

    /** \brief Get the ID (i.e. scan-order index an an arc property map) for
    the given ar \a a (API: LEMON).
    */
    index_type id(Arc const & a) const
    {
        return detail::CoordinateToScanOrder<N + 1>::exec(arc_propmap_shape(), directedArc(a));
    }

    index_type id(ArcIt const & a) const
    {
        return id(*a);
    }

    index_type id(OutArcIt const & a) const
    {
        return id(*a);
    }

    index_type id(OutBackArcIt const & a) const
    {
        return id(*a);
    }

    /** \brief Get an arc descriptor for the given arc ID \a i (API: LEMON).

    Return <tt>Arc(lemon::INVALID)</tt> when the ID does not exist
    in this graph.
    */
    Arc arcFromId(index_type i) const
    {
        if (i < 0 || i > maxArcId())
            return Arc(lemon::INVALID);

        Arc res;
        detail::ScanOrderToCoordinate<N + 1>::exec(i, arc_propmap_shape(), res);
        unsigned int b = detail::BorderTypeImpl<N>::exec(res.template subarray<0, N>(), shape());
        if (neighborExists_[b][res[N]])
            return undirectedArc(res);
        else
            return Arc(lemon::INVALID);
    }

    /** \brief Get the maximal ID af any arc in this graph (API: LEMON).
    */
    index_type maxArcId() const
    {
        if (max_arc_id_ == -2) // -2 means uninitialized
            const_cast<GridGraph *>(this)->computeMaxEdgeAndArcId();
        return max_arc_id_;
    }

    /** \brief Return <tt>true</tt> when the arc is looking on the underlying
    edge in its natural (i.e. forward) direction, <tt>false</tt> otherwise (API: LEMON).
    */
    bool direction(Arc const & a) const
    {
        return !a.isReversed();
    }

    /** \brief Create an arc for the given edge \a e, oriented along the
    edge's natural (<tt>forward = true</tt>) or reversed
    (<tt>forward = false</tt>) direction (API: LEMON).
    */
    Arc direct(Edge const & e, bool forward) const
    {
        if (!is_directed || forward)
            return Arc(e, !forward);
        else
            return Arc(v(e), oppositeIndex(e[N]), true);
    }

    /** \brief Create an arc for the given edge \a e oriented
    so that node \a n is the starting node of the arc (API: LEMON), or
    return <tt>lemon::INVALID</tt> if the edge is not incident to this node.
    */
    Arc direct(Edge const & e, Node const & n) const
    {
        if (u(e) == n)
            return direct(e, true);
        if (v(e) == n)
            return direct(e, false);
        return Arc(lemon::INVALID);
    }

    /** \brief Return the opposite node of the given node \a n
    along edge \a e (API: LEMON), or return <tt>lemon::INVALID</tt>
    if the edge is not incident to this node.
    */
    Node oppositeNode(Node const & n, Edge const & e) const
    {
        Node start(u(e)), end(v(e));
        if (n == start)
            return end;
        if (n == end)
            return start;
        return Node(lemon::INVALID);
    }

    /** \brief Create an arc referring to the same edge as the given
    arc \a a, but with reversed direction (API: LEMON).
    */
    Arc oppositeArc(Arc const & a) const
    {
        return is_directed
            ? Arc(neighbor(a.vertexDescriptor(), a.edgeIndex()), oppositeIndex(a.edgeIndex()), false)
            : Arc(a, !a.isReversed());
    }

    // internal function
    // transforms the arc into its directed form (i.e. a.isReversed() is
    // guaranteed to be false in the returned arc).
    Arc directedArc(Arc const & a) const
    {
        return a.isReversed()
            ? Arc(neighbor(a.vertexDescriptor(), a.edgeIndex()), oppositeIndex(a.edgeIndex()), false)
            : a;
    }

    // internal function
    // transforms the arc into its undirected form (i.e. a.isReversed() will
    // be true in the returned arc if this graph is undirected and the arc
    // traverses the edge backwards).
    Arc undirectedArc(Arc const & a) const
    {
        return a.edgeIndex() < maxUniqueDegree()
            ? a
            : Arc(neighbor(a.vertexDescriptor(), a.edgeIndex()), oppositeIndex(a.edgeIndex()), true);
    }

    /** \brief Return the start node of the edge the given iterator is referring to (API: LEMON).
    */
    Node baseNode(IncEdgeIt const & e)  const
    {
        return source(e.arcDescriptor());
    }

    /** \brief Return the start node of the edge the given iterator is referring to (API: VIGRA).
    */
    Node baseNode(IncBackEdgeIt const & e)  const
    {
        return source(e.arcDescriptor());
    }

    /** \brief Return the start node of the edge the given iterator is referring to (API: LEMON).
    */
    Node baseNode(OutArcIt const & a)  const
    {
        return source(*a);
    }

    /** \brief Return the start node of the edge the given iterator is referring to (API: VIGRA).
    */
    Node baseNode(OutBackArcIt const & a)  const
    {
        return source(*a);
    }

    /** \brief Return the end node of the edge the given iterator is referring to (API: LEMON).
    */
    Node runningNode(IncEdgeIt const & e)  const
    {
        return target(e.arcDescriptor());
    }

    /** \brief Return the end node of the edge the given iterator is referring to (API: VIGRA).
    */
    Node runningNode(IncBackEdgeIt const & e)  const
    {
        return target(e.arcDescriptor());
    }

    /** \brief Return the end node of the edge the given iterator is referring to (API: LEMON).
    */
    Node runningNode(OutArcIt const & a)  const
    {
        return target(*a);
    }

    /** \brief Return the end node of the edge the given iterator is referring to (API: VIGRA).
    */
    Node runningNode(OutBackArcIt const & a)  const
    {
        return target(*a);
    }

    /** \brief Get the start node of the given arc \a a (API: LEMON).
    */
    Node source(Arc const & a) const
    {
        return source_or_target(a, true);
    }

    /** \brief Get the end node of the given arc \a a (API: LEMON).
    */
    Node target(Arc const & a) const
    {
        return source_or_target(a, false);
    }

    /** \brief Get the start node of the given edge \a e (API: LEMON,<br/>
    the boost::graph API provides the free function <tt>boost::source(e, graph)</tt>).
    */
    Node u(Edge const & e) const
    {
        return Node(e.template subarray<0, N>());
    }

    /** \brief Get the end node of the given edge \a e (API: LEMON,<br/>
    the boost::graph API provides the free function <tt>boost::target(e, graph)</tt>).
    */
    Node v(Edge const & e) const
    {
        return Node(e.template subarray<0, N>()) + neighborOffsets_[e[N]];
    }

    /** \brief Get an iterator pointing to the first outgoing edge of the given vertex (convenience function,<br/>
    the boost::graph API provides the free function <tt>boost::out_edges(v, graph)</tt>,<br/>
    LEMON uses <tt>Graph::OutArcIt(g, v)</tt>).
    */
    out_edge_iterator get_out_edge_iterator(vertex_descriptor const & v) const
    {
        return out_edge_iterator(*this, v);
    }

    /** \brief Get an iterator pointing beyond the range of outgoing edges of the given vertex (convenience function,<br/>
    the boost::graph API provides the free function <tt>boost::out_edges(v, graph)</tt>,<br/>
    LEMON uses the special value <tt>lemon::INVALID</tt> instead).
    */
    out_edge_iterator get_out_edge_end_iterator(vertex_descriptor const & v) const
    {
        return get_out_edge_iterator(v).getEndIterator();
    }

    /** \brief Get an iterator pointing to the first outgoing backward edge of the given vertex (API: VIGRA,<br/>
    in analogy to the boost::graph API, we also provide a free function <tt>boost::out_back_edges(v, g)</tt>,<br/>
    and the LEMON analogue is <tt>Graph::IncBackEdgeIt(graph, v)</tt>).
    */
    out_back_edge_iterator get_out_back_edge_iterator(vertex_descriptor const & v) const
    {
        return out_back_edge_iterator(*this, v);
    }

    /** \brief Get an iterator pointing beyond the range of outgoing backward edges of the given vertex (API: VIGRA,<br/>
    in analogy to the boost::graph API, we also provide a free function <tt>boost::out_back_edges(v, g)</tt>,<br/>
    and LEMON uses the special value <tt>lemon::INVALID</tt> instead).
    */
    out_back_edge_iterator get_out_back_edge_end_iterator(vertex_descriptor const & v) const
    {
        return get_out_back_edge_iterator(v).getEndIterator();
    }

    /** \brief Get an iterator pointing to the first incoming edge of the given vertex (convenience function,<br/>
    the boost::graph API provides the free function <tt>boost::in_edges(v, graph)</tt>,<br/>
    LEMON uses <tt>Graph::InArcIt(g, v)</tt>).
    */
    in_edge_iterator get_in_edge_iterator(vertex_descriptor const & v) const
    {
        return in_edge_iterator(*this, v);
    }

    /** \brief Get an iterator pointing beyond the range of incoming edges of the given vertex (convenience function,<br/>
    the boost::graph API provides the free function <tt>boost::in_edges(v, graph)</tt>,<br/>
    LEMON uses the special value <tt>lemon::INVALID</tt> instead).
    */
    in_edge_iterator get_in_edge_end_iterator(vertex_descriptor const & v) const
    {
        return get_in_edge_iterator(v).getEndIterator();
    }

    /** \brief Get the number of outgoing edges of the given vertex (convenience function,<br/>
    the boost::graph API provides the free function <tt>boost::out_degree(v, graph)</tt>,<br/>
    LEMON uses a special property map <tt>lemon::OutDegMap<Graph></tt>).
    */
    degree_size_type out_degree(vertex_descriptor const & v) const
    {
        return (degree_size_type)neighborIndices_[get_border_type(v)].size();
    }

    /** \brief Get the number of outgoing backward edges of the given vertex (API: VIGRA).
    */
    degree_size_type back_degree(vertex_descriptor const & v) const
    {
        return (degree_size_type)backIndices_[get_border_type(v)].size();
    }

    /** \brief Get the number of outgoing forward edges of the given vertex (API: VIGRA).
    */
    degree_size_type forward_degree(vertex_descriptor const & v) const
    {
        unsigned int bt = get_border_type(v);
        return (degree_size_type)(neighborIndices_[bt].size() - backIndices_[bt].size());
    }

    /** \brief Get the number of incoming edges of the given vertex (convenience function,<br/>
    the boost::graph API provides the free function <tt>boost::in_degree(v, graph)</tt>,<br/>
    LEMON uses a special property map <tt>lemon::InDegMap<Graph></tt>).
    */
    degree_size_type in_degree(vertex_descriptor const & v) const
    {
        return out_degree(v);
    }

    /** \brief Get the total number of edges (incoming plus outgoing) of the given vertex (convenience function,<br/>
    the boost::graph API provides the free function <tt>boost::degree(v, graph)</tt>,<br/>
    LEMON has no analogue).
    */
    degree_size_type degree(vertex_descriptor const & v) const
    {
        return is_directed
            ? 2 * out_degree(v)
            : out_degree(v);
    }

    // --------------------------------------------------
    // support for EdgeListGraph:

    /** \brief Get the number of edges in this graph (convenience function,
    boost::graph API provides the free function <tt>boost::num_edges(graph)</tt>).
    */
    edges_size_type num_edges() const
    {
        return num_edges_;
    }

    /** \brief Get the number of edges in this graph (API: LEMON).
    */
    edges_size_type edgeNum() const
    {
        return num_edges();
    }

    /** \brief Get the number of arc in this graph (API: LEMON).
    */
    edges_size_type arcNum() const
    {
        return is_directed
            ? num_edges()
            : 2 * num_edges();
    }

    /** \brief Get edge iterator pointing to the first edge of the graph (convenience function,<br/>
    the boost::graph API provides the free function <tt>boost::edges(graph)</tt>,<br/>
    LEMON uses <tt>Graph::EdgeIt(graph)</tt>).
    */
    edge_iterator get_edge_iterator() const
    {
        return edge_iterator(*this);
    }

    /** \brief Get edge iterator pointing beyond the valid range of edges of this graph (convenience function,<br/>
    the boost::graph API provides the free function <tt>boost::vertices(graph)</tt>,<br/>
    LEMON uses the special value <tt>lemon::INVALID</tt> instead).
    */
    edge_iterator get_edge_end_iterator() const
    {
        return get_edge_iterator().getEndIterator();
    }

    // --------------------------------------------------
    // support for AdjacencyMatrix concept:

    /** \brief Get a descriptor for the edge connecting vertices \a u and \a v,<br/>
    or <tt>(lemon::INVALID, false)</tt> if no such edge exists (convenience function,<br/>
    the boost::graph API provides the free function <tt>boost::edge(u, v, graph)</tt>).
    */
    std::pair<edge_descriptor, bool>
        edge(vertex_descriptor const & u, vertex_descriptor const & v) const
    {
        std::pair<edge_descriptor, bool> res(lemon::INVALID, false);

        neighbor_vertex_iterator i = get_neighbor_vertex_iterator(u),
            end = i.getEndIterator();
        for (; i != end; ++i)
        {
            if (*i == v)
            {
                res.first = make_edge_descriptor(u, i.neighborIndex());
                res.second = true;
                break;
            }
        }
        return res;
    }

    /** \brief Get a descriptor for the edge connecting vertices \a u and \a v,<br/>or <tt>lemon::INVALID</tt> if no such edge exists (API: LEMON).
    */
    Edge findEdge(Node const & u, Node const & v, Edge const & = lemon::INVALID) const
    {
        return this->edge(u, v).first;
    }

    /** \brief Get a descriptor for the arc connecting vertices \a u and \a v,<br/>or <tt>lemon::INVALID</tt> if no such edge exists (API: LEMON).
    */
    Arc findArc(Node const & u, Node const & v, Arc const & = lemon::INVALID) const
    {
        return this->edge(u, v).first;
        // std::pair<edge_descriptor, bool> res(edge(u, v));
        // return res.second
        // ? res.first
        // : Arc(lemon::INVALID);
    }

    /** \brief Create a property map that returns the coordinate of each node (API: LEMON GridGraph).
    */
    IndexMap indexMap() const
    {
        return IndexMap();
    }

    // --------------------------------------------------
    // other helper functions:

    bool isDirected() const
    {
        return is_directed;
    }

    degree_size_type maxDegree() const
    {
        return (degree_size_type)neighborOffsets_.size();
    }

    degree_size_type maxUniqueDegree() const
    {
        return is_directed
            ? maxDegree()
            : maxDegree() / 2;
    }

    shape_type const & shape() const
    {
        return shape_;
    }

    edge_propmap_shape_type edge_propmap_shape() const
    {
        edge_propmap_shape_type res(SkipInitialization);
        res.template subarray<0, N>() = shape_;
        res[N] = maxUniqueDegree();
        return res;
    }

    edge_propmap_shape_type arc_propmap_shape() const
    {
        edge_propmap_shape_type res(SkipInitialization);
        res.template subarray<0, N>() = shape_;
        res[N] = maxDegree();
        return res;
    }

    unsigned int get_border_type(vertex_descriptor const & v) const
    {
        return detail::BorderTypeImpl<N>::exec(v, shape_);
    }

    unsigned int get_border_type(vertex_iterator const & v) const
    {
        return v.borderType();
    }

    index_type oppositeIndex(index_type neighborIndex) const
    {
        return  maxDegree() - neighborIndex - 1;
    }

    /* the given neighborIndex must be valid for the given vertex,
    otherwise this function will crash
    */
    edge_descriptor make_edge_descriptor(vertex_descriptor const & v,
        index_type neighborIndex) const
    {
        if (neighborIndex < maxUniqueDegree())
            return edge_descriptor(v, neighborIndex, false);
        else
            return edge_descriptor(neighbor(v, neighborIndex), oppositeIndex(neighborIndex), true);
    }

    shape_type const & neighborOffset(index_type neighborIndex) const
    {
        return neighborOffsets_[neighborIndex];
    }

    vertex_descriptor neighbor(vertex_descriptor const & v, index_type neighborIndex) const
    {
        return v + neighborOffsets_[neighborIndex];
    }

    vertex_descriptor
        source_or_target(edge_descriptor const & e, bool return_source) const
    {
        // source is always the attached node (first coords) unless the
        // edge has been reversed.
        if ((return_source && e.isReversed()) ||
            (!return_source && !e.isReversed()))
        {
            return neighbor(e.vertexDescriptor(), e.edgeIndex());
        }
        else
        {
            return e.vertexDescriptor();
        }
    }

    NeighborOffsetArray const * neighborOffsetArray() const
    {
        return &neighborOffsets_;
    }

    RelativeNeighborOffsetsArray const * neighborIncrementArray() const
    {
        return &incrementalOffsets_;
    }

    RelativeEdgeOffsetsArray const * edgeIncrementArray() const
    {
        return &edgeDescriptorOffsets_;
    }

    IndexArray const * neighborIndexArray(bool backEdgesOnly) const
    {
        return backEdgesOnly
            ? &backIndices_
            : &neighborIndices_;
    }

    NeighborExistsArray const * neighborExistsArray() const
    {
        return &neighborExists_;
    }

protected:
    NeighborOffsetArray neighborOffsets_;
    NeighborExistsArray neighborExists_;
    IndexArray neighborIndices_, backIndices_;
    RelativeNeighborOffsetsArray incrementalOffsets_;
    RelativeEdgeOffsetsArray edgeDescriptorOffsets_;
    shape_type shape_;
    MultiArrayIndex num_vertices_, num_edges_, max_node_id_, max_arc_id_, max_edge_id_;
    NeighborhoodType neighborhoodType_;
};

#endif

} // namespace vigra

template <int N>
struct NeighborhoodTests
{
    typedef Shape<N> S;
    typedef Shape<(N==runtime_size) ? N : N+1> Edge;

    static const int ndim = (N == runtime_size)
        ? 3
        : N;
    
    std::vector<S> neighborOffsets;
    std::vector<std::vector<bool>> neighborExists;
    std::vector<std::vector<S> > neighborIncrements, backOffsets, forwardOffsets;
    std::vector<std::vector<GridGraphArcDescriptor<N> > > arcIncrements, backEdgeDescrOffsets, forwardEdgeDescrOffsets;
    std::vector<std::vector<Edge> > adjacencyIncrements;
    std::vector<std::vector<ArrayIndex> > neighborIndices, backIndices, forwardIndices;

    NeighborhoodTests()
    {}
    
    void testVertexIterator()
    {
        CoordinateIterator<N> i(S(tags::size=ndim, 3)), iend = i.end();
        
        for(; i != iend; ++i)
        {
            // create all possible array shapes from 1**N to 3**N
            S s = *i + 1;
            CoordinateIterator<N> vi(s), viend = vi.end();
            
            ArrayND<N, int> vertex_map(s);

            for(; vi != viend; ++vi)
            {
                should(vi.isValid() && !vi.atEnd());
                vertex_map[*vi] += 1;
            }
            
            should(!vi.isValid() && vi.atEnd());
            
            // check that all vertice are found
            auto minmax = vertex_map.minmax();
            
            shouldEqual(minmax[0], 1);
            shouldEqual(minmax[1], 1);
        }
    }   

    template <NeighborhoodType neighborhoodType, MemoryOrder memoryOrder>
    void testNeighborhood()
    {
        S shape(tags::size = ndim, 3);
        
        detail::makeNeighborhoodND(ndim, neighborhoodType, memoryOrder, neighborOffsets);
        detail::makeBorderNeighborhoodND(neighborOffsets, neighborIndices, backIndices);

        const int degree = (neighborhoodType == DirectNeighborhood)
                                 ? 2 * ndim
                                 : pow(3, ndim) - 1;
        shouldEqual(neighborOffsets.size(), degree);
        shouldEqual(gridGraphMaxDegree(ndim, neighborhoodType), degree);
        shouldEqual(neighborIndices.size(), pow(2, 2 * ndim));

        ArrayND<ndim, int> scanOrder(shape, memoryOrder), found(shape, memoryOrder);
        std::iota(scanOrder.begin(), scanOrder.end(), 0);
        found[scanOrder.shape() / 2] = 1;

        int scanOrderIndex = -1;
        S forward(tags::size = ndim), backward(tags::size = ndim), 
          strides = shapeToStrides(scanOrder.shape(), memoryOrder);
        for (int k = 0; k<degree; ++k)
        {
            // test that neighbors are listed in scan order
            auto target = neighborOffsets[k] + 1;
            should(scanOrderIndex < scanOrder[target]);
            scanOrderIndex = scanOrder[target];
            shouldEqual(scanOrderIndex, nodeIdFromCoord(target, strides));
            shouldEqual(target, nodeCoordFromId(scanOrderIndex, shape, memoryOrder));
            shouldEqual(dot(strides, neighborOffsets[k]), scanOrderIndex - scanOrder.size() / 2);

            if (neighborhoodType == DirectNeighborhood)
            {
                // test that offset is +-1 in exactly one direction
                shouldEqual(sum(abs(neighborOffsets[k])), 1);
            }
            else
            {
                // check that offset is at most +-1 in any direction
                shouldEqual(max(abs(neighborOffsets[k])), 1);
            }

            // mark neighbor as found
            found[target] = 1;

            if (k < degree / 2)
            {
                should(dot(strides, neighborOffsets[k]) < 0); // check that backward neighbors are first
                backward += neighborOffsets[k];               // register backward neighbors
            }
            else
            {
                should(dot(strides, neighborOffsets[k]) > 0); // check that forward neighbors are last
                forward += neighborOffsets[k];                // register forward neighbors
            }

            shouldEqual(neighborOffsets[k], -neighborOffsets[degree - 1 - k]); // check index of opposite neighbor
        }

        if (neighborhoodType == DirectNeighborhood)
        {
            shouldEqual(backward, S(tags::size = ndim, -1)); // check that all backward neighbors were found
            shouldEqual(forward, S(tags::size = ndim, 1));   // check that all forward neighbors were found
        }
        else
        {
            should(found.all()); // check that all neighbors were found
        }

        // check neighborhoods at ROI border
        ArrayND<1, uint8_t> checkNeighborCodes(Shape<1>(neighborIndices.size()), (uint8_t)0);
        CoordinateIterator<N> i(S(tags::size = ndim, 3));
        for (; i.isValid(); ++i)
        {
            // create all possible array shapes from 1**N to 3**N
            // and check neighborhood of all pixels
            CoordinateIterator<N> vi(*i + 1);
            for (; vi.isValid(); ++vi)
            {
                int borderType = vi.borderType();
                checkNeighborCodes[borderType] = 1;

                int j = 0, jb = 0;
                for (int k = 0; k<degree; ++k)
                {
                    // check that neighborIndices and backIndices contain the correct neighbors
                    if (vi.isInside(vi.coord() + neighborOffsets[k]))
                    {
                        shouldEqual(neighborIndices[borderType][j], k);
                        ++j;
                        if (k < degree / 2)
                        {
                            shouldEqual(backIndices[borderType][jb], k);
                            ++jb;
                        }
                    }
                    else if(j < neighborIndices[borderType].size())
                    {
                        should(neighborIndices[borderType][j] != k);
                    }
                }
                shouldEqual(j, neighborIndices[borderType].size());
                shouldEqual(jb, backIndices[borderType].size());
            }
        }

        should(checkNeighborCodes.all()); // check that all possible neighborhoods have been tested
    }

    template <NeighborhoodType NType, MemoryOrder memoryOrder>
    void testNeighborhoodIterator()
    {
        detail::makeNeighborhoodND(ndim, NType, memoryOrder, neighborOffsets);
        detail::makeBorderNeighborhoodND(neighborOffsets, neighborIndices, backIndices);
        detail::computeNeighborIncrements(neighborOffsets, neighborIndices, neighborIncrements, arcIncrements, adjacencyIncrements, false);

        // check neighborhoods at ROI border
        CoordinateIterator<N> i(S(tags::size = ndim, 3), memoryOrder);
        int degree = neighborOffsets.size();

        USETICTOC;
        std::cerr << "N: " << N << ", neighborhood: " << NType << ", order: " << memoryOrder<< ": ";
        TIC;
        for(; i.isValid(); ++i)
        {
            // create all possible array shapes from 1**N to 3**N
            // check neighborhood of all pixels
            S shape = *i + 1,
            //S shape(tags::size = ndim, 3),
                strides = shapeToStrides(shape, memoryOrder);
            CoordinateIterator<N> vi(shape, memoryOrder);

            Shape<> edge_offsets = edgeOffsets(shape, neighborOffsets, true),
                    edge_offsets2 = edge_offsets;
            std::vector<Shape<N>> edge_shapes, edge_pivots, edge_strides;
            for (int k = 0; k < edge_offsets2.size(); ++k)
            {
                auto p = min(neighborOffsets[k], 0);
                auto sh = shape - abs(neighborOffsets[k]);
                auto st = shapeToStrides(sh, memoryOrder);
                edge_pivots.push_back(p);
                edge_shapes.push_back(sh);
                edge_strides.push_back(st);
                edge_offsets2[k] += dot(p, st);
            }
            //std::cerr << edge_offsets << "\n";
            auto directed_edges = gridGraphEdgeCount(shape, NType, true);
            auto undirected_edges = gridGraphEdgeCount(shape, NType, false);
            ArrayND<1, int> foundDirectedEdgeIds(directed_edges),
                            foundUndirectedEdgeIds(undirected_edges);
            ArrayND<ndim+1, int> foundDirectedEdges(shape.insert(ndim, degree)),
                                 foundUndirectedEdges(shape.insert(ndim, degree/2));

            for(; vi.isValid(); ++vi)
            {
                int borderType = vi.borderType();
                
                {

                    GridGraphNeighborIterator<N> ni(neighborIncrements[borderType],
                                                    neighborIndices[borderType], 
                                                    vi.coord()),
                                                 nend = ni.end();
                    GridGraphAdjacencyIterator<N> di(adjacencyIncrements[borderType],
                                                     neighborIndices[borderType],
                                                     vi.coord(), true),
                                                  dend = di.end();
                    GridGraphAdjacencyIterator<N> ai(adjacencyIncrements[borderType],
                                                     neighborIndices[borderType],
                                                     vi.coord(), false),
                                                  aend = ai.end();

                    for(int k=0; k<neighborIndices[borderType].size(); ++k)
                    {
                        auto index = neighborIndices[borderType][k];
                        auto source = vi.coord();
                        auto target = source + neighborOffsets[index];

                        should(ni != nend);
                        should(ni.isValid() && !ni.atEnd());
                        shouldEqual(target, *ni);
                        ++ni;

                        should(di != dend);
                        should(di.isValid() && !di.atEnd());
                        shouldEqual(source, di->source());
                        shouldEqual(target, di->target());
                        shouldEqual(target, di->node());
                        shouldEqual(source.insert(ndim, index), di->edge());
                        auto nodeId = nodeIdFromCoord(di->source(), strides);
                        auto edgeId = edgeIdFromCoord(di->edge(), edge_strides[di->edge().back()], edge_offsets2);
                        should(edgeId >= 0 && edgeId < directed_edges);
                        auto edge = edgeCoordFromId(edgeId, edge_shapes, memoryOrder, edge_offsets, edge_offsets2);
                        edge.subarray(0, ndim) -= min(neighborOffsets[edge.back()], 0);
                        auto edge2 = edgeCoordFromId2(edgeId, edge_shapes, memoryOrder, edge_offsets, edge_pivots);
                        shouldEqual(edge, edge2);
                        //std::cerr << edge << " " << edge2 << " " << (edge == edge2) << "\n";
                        //std::cerr << di->source() << " " << di->edge() << " " << edgeId << "\n";
                        //std::cerr << "        " << edge << "\n";
                        shouldEqual(di->edge(), edge);
                        ++foundDirectedEdgeIds(edgeId);
                        ++foundDirectedEdges[edge];
                        ++di;

                        should(ai != aend);
                        should(ai.isValid() && !ai.atEnd());
                        shouldEqual(source, ai->source());
                        shouldEqual(target, ai->target());
                        shouldEqual(target, ai->node());
                        if(index < degree / 2)
                            shouldEqual(source.insert(ndim, index), ai->edge());
                        else
                            shouldEqual(target.insert(ndim, degree - 1 - index), ai->edge());
                        //std::cerr << "adjacency: " << ai->arc_ << " " << ai->reverse_arc_ << " " << ai->arc_index_ 
                        //    << ai->edge() << "\n";
                        //edgeId = edgeIdFromCoord(ai->edge(), shape, neighborOffsets, undirected_edge_offsets, memoryOrder);
                        //std::cerr << ai->source() << " " << ai->edge() << " " << nodeId << " " << edgeId << "\n";
                        edgeId = edgeIdFromCoord(ai->edge(), edge_strides[ai->edge().back()], edge_offsets2);
                        should(edgeId >= 0 && edgeId < undirected_edges);
                        edge = edgeCoordFromId(edgeId, edge_shapes, memoryOrder, edge_offsets, edge_offsets2);
                        edge.template subarray<0, ndim>() -= min(neighborOffsets[edge.back()], 0);
                        edge2 = edgeCoordFromId2(edgeId, edge_shapes, memoryOrder, edge_offsets, edge_pivots);
                        shouldEqual(edge, edge2);
                        shouldEqual(ai->edge(), edge);
                        ++foundUndirectedEdgeIds(edgeId);
                        ++foundUndirectedEdges[edge];
                        ++ai;
                    }
                    should(ni == nend);
                    should(ni.atEnd() && !ni.isValid());
                    should(di == aend);
                    should(di.atEnd() && !di.isValid());
                    should(ai == aend);
                    should(ai.atEnd() && !ai.isValid());
                }
                
                {
                    GridGraphNeighborIterator<N> ni(neighborIncrements[borderType], 
                                                    backIndices[borderType], 
                                                    vi.coord()),
                                                 nend = ni.end();
                    
                    for(int k=0; k<backIndices[borderType].size(); ++k)
                    {
                        should(ni.isValid() && !ni.atEnd());
                        shouldEqual(vi.coord()+neighborOffsets[backIndices[borderType][k]], *ni);
                        ++ni;
                    }
                    should(ni == nend);
                    should(ni.atEnd() && !ni.isValid());
                }
            }
            if (directed_edges > 0)
            {
                should(foundDirectedEdgeIds == 1);
                for (int k = 0; k < directed_edges; ++k)
                {
                    auto edge = edgeCoordFromId(k, edge_shapes, memoryOrder, edge_offsets, edge_offsets2);
                    edge.template subarray<0, ndim>() -= min(neighborOffsets[edge.back()], 0);
                    shouldEqual(foundDirectedEdges[edge], 1);
                    foundDirectedEdges[edge] = 0;
                }
                should(foundDirectedEdges == 0);
            }
            if (undirected_edges > 0)
            {
                should(foundUndirectedEdgeIds == 2);
                for (int k = 0; k < undirected_edges; ++k)
                {
                    auto edge = edgeCoordFromId(k, edge_shapes, memoryOrder, edge_offsets, edge_offsets2);
                    edge.template subarray<0, ndim>() -= min(neighborOffsets[edge.back()], 0);
                    shouldEqual(foundUndirectedEdges[edge], 2);
                    foundUndirectedEdges[edge] = 0;
                }
                should(foundUndirectedEdges == 0);
            }
        }
        TOC;
    }
    
#if 0    

    template <NeighborhoodType NType>
    void testOutArcIteratorDirected()
    {
        detail::makeArrayNeighborhood(neighborOffsets, neighborExists, NType);
        detail::computeNeighborOffsets(neighborOffsets, neighborExists, relativeOffsets, edgeDescrOffsets, neighborIndices, backIndices, true);

        // check neighborhoods at ROI border
        CoordinateIterator<N> i(Shape(3)), iend = i.getEndIterator();
        ArrayND<N, int> a(Shape(3));
        typedef typename ArrayND<N, int>::view_type View;
        
        for(; i != iend; ++i)
        {
            // create all possible array shapes from 1**N to 3**N
            View va = a.subarray(Shape(), *i+Shape(1)); 
            
            // check neighborhood of all pixels
            typename View::iterator vi = va.begin(), viend = vi.getEndIterator();
            for(; vi != viend; ++vi)
            {
                int borderType = vi.borderType();
                
                {
                    GridGraphOutArcIterator<N, false> ni(edgeDescrOffsets[borderType], 
                                                         neighborIndices[borderType], 
                                                         backIndices[borderType], 
                                                         vi.point()),
                                                      nend = ni.getEndIterator();
                    
                    for(int k=0; k<neighborExists[borderType].size(); ++k)
                    {
                        if(neighborExists[borderType][k])
                        {
                            should(ni.isValid() && !ni.atEnd());
                            shouldEqual(vi.point(), ni->vertexDescriptor());
                            shouldEqual(k, ni->edgeIndex());
                            shouldEqual(k, ni.neighborIndex());
                            should(!ni->isReversed());
                            ++ni;
                        }
                    }
                    should(ni == nend);
                    should(ni.atEnd() && !ni.isValid());
                }
                        
                {
                    GridGraphOutArcIterator<N, true> ni(edgeDescrOffsets[borderType], 
                                                        neighborIndices[borderType], 
                                                        backIndices[borderType], 
                                                        vi.point()),
                                                     nend = ni.getEndIterator();
                    
                    for(int k=0; k<neighborExists[borderType].size()/2; ++k)
                    {
                        if(neighborExists[borderType][k])
                        {
                            should(ni.isValid() && !ni.atEnd());
                            shouldEqual(vi.point(), ni->vertexDescriptor());
                            shouldEqual(k, ni->edgeIndex());
                            shouldEqual(k, ni.neighborIndex());
                            should(!ni->isReversed());
                            ++ni;
                        }
                    }
                    should(ni == nend);
                    should(ni.atEnd() && !ni.isValid());
                }
            }
        }
    }
    
    template <NeighborhoodType NType>
    void testOutArcIteratorUndirected()
    {
        detail::makeArrayNeighborhood(neighborOffsets, neighborExists, NType);
        detail::computeNeighborOffsets(neighborOffsets, neighborExists, relativeOffsets, edgeDescrOffsets, neighborIndices, backIndices, false);

        // check neighborhoods at ROI border
        CoordinateIterator<N> i(Shape(3)), iend = i.getEndIterator();
        ArrayND<N, int> a(Shape(3));
        typedef typename ArrayND<N, int>::view_type View;
        
        for(; i != iend; ++i)
        {
            // create all possible array shapes from 1**N to 3**N
            View va = a.subarray(Shape(), *i+Shape(1)); 
            
            // check neighborhood of all pixels
            typename View::iterator vi = va.begin(), viend = vi.getEndIterator();
            for(; vi != viend; ++vi)
            {
                int borderType = vi.borderType();
                
                {
                    GridGraphOutArcIterator<N> ni(edgeDescrOffsets[borderType], 
                                                  neighborIndices[borderType], 
                                                  backIndices[borderType], 
                                                  vi.point()),
                                                nend = ni.getEndIterator();
                    
                    for(int k=0; k<neighborExists[borderType].size(); ++k)
                    {
                        if(neighborExists[borderType][k])
                        {
                            should(ni.isValid() && !ni.atEnd());
                            shouldEqual(k, ni.neighborIndex());
                            if(k < neighborExists[borderType].size() / 2)
                            {
                                shouldEqual(vi.point(), ni->vertexDescriptor());
                                shouldEqual(k, ni->edgeIndex());
                                should(!ni->isReversed());
                            }
                            else
                            {
                                shouldEqual(vi.point()+neighborOffsets[k], ni->vertexDescriptor());
                                shouldEqual(k, (int)neighborOffsets.size() - ni->edgeIndex() - 1);
                                should(ni->isReversed());
                            }
                            ++ni;
                        }
                    }
                    should(ni == nend);
                    should(ni.atEnd() && !ni.isValid());
                }
                        
                {
                    GridGraphOutArcIterator<N, true> ni(edgeDescrOffsets[borderType], 
                                                        neighborIndices[borderType], 
                                                        backIndices[borderType], 
                                                        vi.point()),
                                                     nend = ni.getEndIterator();
                    
                    for(int k=0; k<neighborExists[borderType].size()/2; ++k)
                    {
                        if(neighborExists[borderType][k])
                        {
                            should(ni.isValid() && !ni.atEnd());
                            shouldEqual(k, ni.neighborIndex());
                            shouldEqual(vi.point(), ni->vertexDescriptor());
                            shouldEqual(k, ni->edgeIndex());
                            should(!ni->isReversed());
                            ++ni;
                        }
                    }
                    should(ni == nend);
                    should(ni.atEnd() && !ni.isValid());
                }
            }
        }
    }
    
    template <NeighborhoodType NType>
    void testArcIteratorDirected()
    {
        detail::makeArrayNeighborhood(neighborOffsets, neighborExists, NType);
        detail::computeNeighborOffsets(neighborOffsets, neighborExists, relativeOffsets, edgeDescrOffsets, neighborIndices, backIndices, true);

        // check neighborhoods at ROI border
        CoordinateIterator<N> i(Shape(3)), iend = i.getEndIterator();
        typedef typename MultiArrayShape<N+1>::type EdgeMapShape;
        
        for(; i != iend; ++i)
        {
            // create all possible array shapes from 1**N to 3**N
            Shape s = *i + Shape(1);
            
            EdgeMapShape es(gridGraphMaxDegree(N, NType));
            es.template subarray<0,N>() = s;
            ArrayND<N+1, int> edge_map(es);
            
            GridGraphArcIterator<N, false> ni(edgeDescrOffsets, neighborIndices, backIndices, s),
                                           nend = ni.getEndIterator();
            
            for(; ni != nend; ++ni)
            {
                should(ni.isValid() && !ni.atEnd());
                edge_map[*ni] += 1;
            }
            
            should(!ni.isValid() && ni.atEnd());
                
            // check neighborhood of all pixels
            CoordinateIterator<N> vi(s), viend = vi.getEndIterator();
            for(; vi != viend; ++vi)
            {
                int borderType = vi.borderType();
                es.template subarray<0,N>() = *vi;
                
                for(es[N]=0; es[N]<(MultiArrayIndex)neighborExists[borderType].size(); ++es[N])
                {
                    if(neighborExists[borderType][es[N]])
                        shouldEqual(edge_map[es], 1);
                    else
                        shouldEqual(edge_map[es], 0);
                }
            }
            
            shouldEqual(edge_map.template sum<int>(), gridGraphEdgeCount(s, NType, true));
        }
    }
    
    template <NeighborhoodType NType>
    void testArcIteratorUndirected()
    {
        detail::makeArrayNeighborhood(neighborOffsets, neighborExists, NType);
        detail::computeNeighborOffsets(neighborOffsets, neighborExists, relativeOffsets, edgeDescrOffsets, neighborIndices, backIndices, false);

        // check neighborhoods at ROI border
        CoordinateIterator<N> i(Shape(3)), iend = i.getEndIterator();
        typedef typename MultiArrayShape<N+1>::type EdgeMapShape;
        
        for(; i != iend; ++i)
        {
            // create all possible array shapes from 1**N to 3**N
            Shape s = *i + Shape(1);
            
            EdgeMapShape es(gridGraphMaxDegree(N, NType) / 2);
            es.template subarray<0,N>() = s;
            ArrayND<N+1, int> edge_map(es);
            
            GridGraphArcIterator<N, true> ni(edgeDescrOffsets, neighborIndices, backIndices, s),
                                          nend = ni.getEndIterator();
            
            for(; ni != nend; ++ni)
            {
                should(ni.isValid() && !ni.atEnd());
                edge_map[*ni] += 1;
            }
            
            should(!ni.isValid() && ni.atEnd());
                
            // check neighborhood of all pixels
            CoordinateIterator<N> vi(s), viend = vi.getEndIterator();
            for(; vi != viend; ++vi)
            {
                int borderType = vi.borderType();
                es.template subarray<0,N>() = *vi;
                
                for(es[N]=0; es[N]<(MultiArrayIndex)neighborExists[borderType].size()/2; ++es[N])
                {
                    if(neighborExists[borderType][es[N]])
                        shouldEqual(edge_map[es], 1);
                    else
                        shouldEqual(edge_map[es], 0);
                }
            }
            
            shouldEqual(edge_map.template sum<int>(), gridGraphEdgeCount(s, NType, false));
        }
    }
#endif
};

#if 0
template <int N>
struct GridGraphTests
{
    typedef typename MultiArrayShape<N>::type Shape;
    
    template <class DirectedTag, NeighborhoodType NType>
    void testBasics()
    {
        using namespace boost_graph;
        typedef GridGraph<N, DirectedTag> G;
        
        static const bool directed = IsSameType<DirectedTag, directed_tag>::value;
        
        CoordinateIterator<N> i(Shape(3)), iend = i.getEndIterator();        
        for(; i != iend; ++i)
        {
            // create all possible array shapes from 1**N to 3**N
            Shape s = *i + Shape(1);
            G g(s, NType);
            
#ifdef WITH_LEMON
            shouldEqual(directed, !lemon::UndirectedTagIndicator<G>::value);
#endif
            
            should(typename G::vertex_descriptor(lemon::INVALID) == lemon::INVALID);
            should(typename G::edge_descriptor(lemon::INVALID) == lemon::INVALID);
        
            shouldEqual(g.shape(), s);
            shouldEqual(g.isDirected(), directed);
            shouldEqual(g.num_vertices(), prod(s));
            shouldEqual(num_vertices(g), prod(s));
            shouldEqual(g.num_edges(), gridGraphEdgeCount(s, NType, directed));
            shouldEqual(num_edges(g), gridGraphEdgeCount(s, NType, directed));
            shouldEqual(g.maxDegree(), gridGraphMaxDegree(N, NType));
            
            shouldEqual(g.num_vertices(), g.nodeNum());
            shouldEqual(g.num_edges(), g.edgeNum());
            shouldEqual(g.arcNum(), directed ? g.edgeNum() : 2*g.edgeNum());
        }
#ifdef WITH_BOOST_GRAPH
        BOOST_CONCEPT_ASSERT(( boost::GraphConcept<G> ));
        BOOST_CONCEPT_ASSERT(( boost::IncidenceGraphConcept<G> ));
        BOOST_CONCEPT_ASSERT(( boost::BidirectionalGraphConcept<G> ));
        BOOST_CONCEPT_ASSERT(( boost::AdjacencyGraphConcept<G> ));
        BOOST_CONCEPT_ASSERT(( boost::VertexListGraphConcept<G> ));
        BOOST_CONCEPT_ASSERT(( boost::EdgeListGraphConcept<G> ));
        BOOST_CONCEPT_ASSERT(( boost::AdjacencyMatrixConcept<G> ));
#endif
    }
    
    template <class DirectedTag, NeighborhoodType NType>
    void testVertexIterator()
    {
        using namespace boost_graph;
        
        typedef GridGraph<N, DirectedTag> Graph;

        CoordinateIterator<N> i(Shape(3)), iend = i.getEndIterator();        
        for(; i != iend; ++i)
        {
            // create all possible array shapes from 1**N to 3**N
            Shape s = *i + Shape(1);
            Graph g(s, NType);
            typename Graph::template NodeMap<int> vertexMap(g);
            lemon::InDegMap<Graph>  inDegreeMap(g);
            lemon::OutDegMap<Graph> outDegreeMap(g);
            typename Graph::IndexMap indexMap(g.indexMap());
        
            int count = 0;
        
            typename Graph::vertex_iterator j = g.get_vertex_iterator(), 
                                            end = g.get_vertex_end_iterator();
            typename Graph::NodeIt lj(g);
                                            
            should(j == vertices(g).first);
            should(end == vertices(g).second);
            should(j == typename Graph::vertex_iterator(g));
            for(; j != end; ++j, ++lj, ++count)
            {
                should(j.isValid() && !j.atEnd());
                should(j == lj);
                should(j != lemon::INVALID);
                should(!(j == lemon::INVALID));
                
                shouldEqual(j.scanOrderIndex(), g.id(j));
                shouldEqual(j.scanOrderIndex(), g.id(*j));
                shouldEqual(*j, g.nodeFromId(g.id(*j)));
                
                shouldEqual(g.out_degree(j), g.out_degree(*j));
                shouldEqual(g.out_degree(j), out_degree(*j, g));
                shouldEqual(g.forward_degree(j), g.forward_degree(*j));
                shouldEqual(g.back_degree(j), g.back_degree(*j));
                shouldEqual(g.forward_degree(j) + g.back_degree(j), g.out_degree(j));
                shouldEqual(g.in_degree(j), g.out_degree(j));
                shouldEqual(g.degree(j), g.isDirected() ? 2*g.out_degree(j) : g.out_degree(j));
                shouldEqual(g.out_degree(j), boost_graph::out_degree(*j, g));
                shouldEqual(g.in_degree(j), boost_graph::in_degree(*j, g));
                shouldEqual(g.out_degree(j), outDegreeMap[*j]);
                shouldEqual(g.in_degree(j), inDegreeMap[*j]);

                shouldEqual(*j, indexMap[*j]);

                put(vertexMap, *j, get(vertexMap, *j) + 1); // same as: vertexMap[*j] += 1;
            }
            should(!j.isValid() && j.atEnd());
            should(j == lj);
            should(j == lemon::INVALID);
            should(!(j != lemon::INVALID));
            
            // check that all vertices are found exactly once
            shouldEqual(count, g.num_vertices());
            int min = NumericTraits<int>::max(), max = NumericTraits<int>::min();
            vertexMap.minmax(&min, &max);
            
            shouldEqual(min, 1);
            shouldEqual(max, 1);
        }
    }
    
    template <class DirectedTag, NeighborhoodType NType>
    void testNeighborIterator()
    {
        using namespace boost_graph;
        
        static const bool directed = IsSameType<DirectedTag, directed_tag>::value;
        
        typedef GridGraph<N, DirectedTag> Graph;
        typedef typename Graph::Node Node;
        
        CoordinateIterator<N> i(Shape(3)), iend = i.getEndIterator();        
        for(; i != iend; ++i)
        {
            // create all possible array shapes from 1**N to 3**N
            Shape s = *i + Shape(1);
            Graph g(s, NType);
            
            ArrayND<N, int> vertexMap(s);
            typename Graph::template EdgeMap<int> edgeMap(g);
            
            typename Graph::template ArcMap<int> arcIdMap(g);            
            typename Graph::template EdgeMap<int> edgeIdMap(g);            
            std::vector<unsigned char> arcExistsMap(g.maxArcId()+1, 0);
            std::vector<unsigned char> edgeExistsMap(g.maxEdgeId()+1, 0);

            linearSequence(arcIdMap.begin(), arcIdMap.end());
            linearSequence(edgeIdMap.begin(), edgeIdMap.end());
            
            shouldEqual((edgeMap.shape().template subarray<0, N>()), s);
            shouldEqual(edgeMap.shape(N), g.maxUniqueDegree());
            shouldEqual((arcIdMap.shape().template subarray<0, N>()), s);
            shouldEqual(arcIdMap.shape(N), g.maxDegree());
            shouldEqual(edgeMap.shape(), edgeIdMap.shape());

            int totalCount = 0;
        
            typename Graph::vertex_iterator j = g.get_vertex_iterator(), 
                                            end = g.get_vertex_end_iterator();
            for(; j != end; ++j)
            {
                typename Graph::neighbor_vertex_iterator n = g.get_neighbor_vertex_iterator(j), 
                                                         nend = g.get_neighbor_vertex_end_iterator(j);
                typename Graph::out_edge_iterator        e = g.get_out_edge_iterator(j), 
                                                         eend = g.get_out_edge_end_iterator(j);
                typename Graph::in_edge_iterator         i = g.get_in_edge_iterator(j), 
                                                         iend = g.get_in_edge_end_iterator(j);
                typename Graph::IncEdgeIt                le(g, j);
                typename Graph::OutArcIt                 la(g, j);
                typename Graph::InArcIt                  li(g, j);

                should(n == g.get_neighbor_vertex_iterator(*j));
                should(nend == g.get_neighbor_vertex_end_iterator(*j));
                should(n == adjacent_vertices(*j, g).first);
                should(nend == adjacent_vertices(*j, g).second);
                should(n == adjacent_vertices_at_iterator(j, g).first);
                should(nend == adjacent_vertices_at_iterator(j, g).second);
                should(e == g.get_out_edge_iterator(*j));
                should(eend == g.get_out_edge_end_iterator(*j));
                should(e == out_edges(*j, g).first);
                should(eend == out_edges(*j, g).second);
                should(i == in_edges(*j, g).first);
                should(iend == in_edges(*j, g).second);

                int count = 0;
                for(; n != nend; ++n, ++e, ++i, ++la, ++le, ++li, ++count)
                {
                    should(*n != *j);
                    should(n.isValid() && !n.atEnd());
                    should(n != lemon::INVALID);
                    should(!(n == lemon::INVALID));
                    should(e.isValid() && !e.atEnd());
                    should(e == la);
                    should(e == le);
                    should(e != eend);
                    should(e != lemon::INVALID);
                    should(!(e == lemon::INVALID));
                    should(i != lemon::INVALID);
                    should(!(i == lemon::INVALID));
                    should(i == li);
                    
                    shouldEqual(source(*e, g), *j);
                    shouldEqual(target(*e, g), *n);
                    vertexMap[*n] += 1;
                    edgeMap[*e] += 1;
                    
                    shouldEqual(source(*i, g), *n);
                    shouldEqual(target(*i, g), *j);
                    
                    typename Graph::Arc a  = g.findArc(*j, *n),
                                        oa = g.findArc(*n, *j);
                    should(a == *la);
                    shouldEqual(source(oa, g), *n);
                    shouldEqual(target(oa, g), *j);
                    shouldEqual(g.baseNode(la), *j);
                    shouldEqual(g.runningNode(la), *n);
                    shouldEqual(g.oppositeArc(a), oa);
                    
                    typename Graph::Edge ge = g.findEdge(*j, *n),
                                         oe = g.findEdge(*n, *j);
                    should(ge == *le);
                    shouldEqual(g.baseNode(le), *j);
                    shouldEqual(g.runningNode(le), *n);
                    
                    typename Graph::Node u = g.u(oe), 
                                         v = g.v(oe);
                    if(directed)
                    {
                        should(g.direction(*la));
                        shouldEqual(g.direct(*le, true), *la);
                        shouldEqual(g.direct(*le, false), g.oppositeArc(*la));
                    }
                    else
                    {
                        should(oe == *le);
                        if(g.id(n) < g.id(j))
                        {
                            std::swap(u, v);
                            should(g.direction(*la));
                            shouldEqual(g.direct(*le, true), *la);
                            shouldEqual(g.direct(*le, false), g.oppositeArc(*la));
                        }
                        else
                        {
                            should(!g.direction(*la));
                            shouldEqual(g.direct(*le, false), *la);
                            shouldEqual(g.direct(*le, true), g.oppositeArc(*la));
                        }
                    }

                    shouldEqual(u, *n);
                    shouldEqual(v, *j);
                    shouldEqual(g.direct(*le, v), *la);
                    shouldEqual(g.direct(*le, u), g.oppositeArc(*la));
                    shouldEqual(g.oppositeNode(u, *le), v);
                    shouldEqual(g.oppositeNode(v, *le), u);
                    shouldEqual(g.oppositeNode(lemon::INVALID, *le), Node(lemon::INVALID));
                    
                    MultiArrayIndex arcId = g.id(*la);
                    shouldEqual(arcIdMap[*la], arcId);
                    arcExistsMap[arcId] = 1;  // mark arc as found

                    MultiArrayIndex edgeId = g.id(*le);
                    shouldEqual(edgeIdMap[*la], g.id((typename Graph::Edge &)*la));
                    shouldEqual(edgeIdMap[*le], g.id(*le));
                    edgeExistsMap[edgeId] = 1;  // mark edge as found

                    shouldEqual(*la, g.arcFromId(g.id(*la)));
                    shouldEqual(*le, g.edgeFromId(g.id(*le)));
                }
                should(!n.isValid() && n.atEnd());
                should(n == lemon::INVALID);
                should(!(n != lemon::INVALID));
                should(!e.isValid() && e.atEnd());
                should(e == eend);
                should(e == la);
                should(e == le);
                should(e == lemon::INVALID);
                should(!(e != lemon::INVALID));
                should(i == li);
                should(i == iend);
                should(i == lemon::INVALID);
                should(!(i != lemon::INVALID));
                
                shouldEqual(count, g.out_degree(j));
                
                totalCount += count;
            }
            
            // check that all neighbors are found
            if(!directed)
                totalCount /= 2;
            shouldEqual(totalCount, g.num_edges());
            
            int min = NumericTraits<int>::max(), max = NumericTraits<int>::min();
            edgeMap.minmax(&min, &max);
            shouldEqual(min, 0);
            shouldEqual(max, g.num_edges() ? !directed ? 2 : 1 : 0);

            j = g.get_vertex_iterator();
            for(; j != end; ++j)
            {
                shouldEqual(vertexMap[*j], g.in_degree(j));
                if(directed)
                    shouldEqual(edgeMap.bindInner(*j).template sum<int>(), g.out_degree(j));
                else
                    shouldEqual(edgeMap.bindInner(*j).template sum<int>(), 2*g.back_degree(j));
            }

            for(int id=0; id <= g.maxEdgeId(); ++id)
            {
                typename Graph::Edge e = g.edgeFromId(id);
                if(edgeExistsMap[id])
                    should(e != lemon::INVALID);
                else
                    should(e == lemon::INVALID);
            }

            for(int id=0; id <= g.maxArcId(); ++id)
            {
                typename Graph::Arc a = g.arcFromId(id);
                if(arcExistsMap[id])
                    should(a != lemon::INVALID);
                else
                    should(a == lemon::INVALID);
            }
        }
    }
    
    template <class DirectedTag, NeighborhoodType NType>
    void testBackNeighborIterator()
    {
        using namespace boost_graph;
        
        static const bool directed = IsSameType<DirectedTag, directed_tag>::value;
        
        typedef GridGraph<N, DirectedTag> Graph;
        
        CoordinateIterator<N> i(Shape(3)), iend = i.getEndIterator();        
        for(; i != iend; ++i)
        {
            // create all possible array shapes from 1**N to 3**N
            Shape s = *i + Shape(1);
            Graph g(s, NType);
            
            ArrayND<N, int> vertexMap(s);
            typename Graph::template EdgeMap<int> edgeMap(g);
            typename Graph::template ArcMap<int> arcMap(g);
            
            typename Graph::template ArcMap<int> arcIdMap(g);            
            typename Graph::template EdgeMap<int> edgeIdMap(g);            

            linearSequence(arcIdMap.begin(), arcIdMap.end());
            linearSequence(edgeIdMap.begin(), edgeIdMap.end());
            
            shouldEqual((edgeMap.shape().template subarray<0, N>()), s);
            shouldEqual(edgeMap.shape(N), g.maxUniqueDegree());
            shouldEqual((arcIdMap.shape().template subarray<0, N>()), s);
            shouldEqual(arcIdMap.shape(N), g.maxDegree());
            shouldEqual(edgeMap.shape(), edgeIdMap.shape());
            
            int totalCount = 0;
        
            typename Graph::vertex_iterator j = g.get_vertex_iterator(), 
                                            end = g.get_vertex_end_iterator();
            for(; j != end; ++j)
            {
                typename Graph::back_neighbor_vertex_iterator n = g.get_back_neighbor_vertex_iterator(j), 
                                                              nend = g.get_back_neighbor_vertex_end_iterator(j);
                typename Graph::out_back_edge_iterator        e = g.get_out_back_edge_iterator(j), 
                                                              eend = g.get_out_back_edge_end_iterator(j);
                typename Graph::IncBackEdgeIt                 le(g, j);
                typename Graph::OutBackArcIt                  la(g, j);

                should(n == g.get_back_neighbor_vertex_iterator(*j));
                should(nend == g.get_back_neighbor_vertex_end_iterator(*j));
                should(n == back_adjacent_vertices(*j, g).first);
                should(nend == back_adjacent_vertices(*j, g).second);
                should(e == g.get_out_back_edge_iterator(*j));
                should(eend == g.get_out_back_edge_end_iterator(*j));
                should(e == out_back_edges(*j, g).first);
                should(eend == out_back_edges(*j, g).second);

                int count = 0;
                for(; n != nend; ++n, ++e, ++la, ++le, ++count)
                {
                    should(*n != *j);
                    should(n.isValid() && !n.atEnd());
                    should(n != lemon::INVALID);
                    should(!(n == lemon::INVALID));
                    should(e.isValid() && !e.atEnd());
                    should(e == la);
                    should(e == le);
                    should(e != eend);
                    should(e != lemon::INVALID);
                    should(!(e == lemon::INVALID));
                    
                    shouldEqual(source(*e, g), *j);
                    shouldEqual(target(*e, g), *n);
                    vertexMap[*n] += 1;
                    edgeMap[*e] += 1;
                    arcMap[*la] += 1;
                    
                    typename Graph::Arc a  = g.findArc(*j, *n),
                                        oa = g.findArc(*n, *j);
                    should(a == *la);
                    shouldEqual(source(oa, g), *n);
                    shouldEqual(target(oa, g), *j);
                    shouldEqual(g.baseNode(la), *j);
                    shouldEqual(g.runningNode(la), *n);
                    shouldEqual(g.oppositeArc(a), oa);
                    
                    typename Graph::Edge ge = g.findEdge(*j, *n),
                                         oe = g.findEdge(*n, *j);
                    should(ge == *le);
                    shouldEqual(g.baseNode(le), *j);
                    shouldEqual(g.runningNode(le), *n);
                    
                    typename Graph::Node u = g.u(oe), 
                                         v = g.v(oe);
                    if(!directed)
                    {
                        should(oe == *le);
                        if(g.id(n) < g.id(j))
                            std::swap(u, v);
                    }
                    shouldEqual(u, *n);
                    shouldEqual(v, *j);
                    
                    shouldEqual(arcIdMap[*la], g.id(la));
                    shouldEqual(edgeIdMap[*la], g.id((typename Graph::Edge &)*la));
                    shouldEqual(edgeIdMap[*le], g.id(*le));
                    shouldEqual(*la, g.arcFromId(g.id(*la)));
                    shouldEqual(*le, g.edgeFromId(g.id(*le)));
                }
                should(!n.isValid() && n.atEnd());
                should(n == lemon::INVALID);
                should(!(n != lemon::INVALID));
                should(!e.isValid() && e.atEnd());
                should(e == eend);
                should(e == la);
                should(e == le);
                should(e == lemon::INVALID);
                should(!(e != lemon::INVALID));
                
                shouldEqual(count, g.back_degree(j));
                
                totalCount += count;
            }
            
            // check that all neighbors are found
            if(directed)
                totalCount *= 2;
            shouldEqual(totalCount, g.num_edges());
            
            int min = NumericTraits<int>::max(), max = NumericTraits<int>::min();
            edgeMap.minmax(&min, &max);
            shouldEqual(min, 0);
            shouldEqual(max, g.num_edges() ? 1 : 0);

            typename Graph::edge_propmap_shape_type estart, estop;
            estop.template subarray<0,N>() = s;
            estop[N] = g.maxDegree() / 2;
            shouldEqualSequence(edgeMap.begin(), edgeMap.end(), arcMap.subarray(estart, estop).begin());
            
            estart[N] = g.maxDegree() / 2;
            estop[N] = g.maxDegree();
            min = NumericTraits<int>::max(), max = NumericTraits<int>::min();
            arcMap.subarray(estart, estop).minmax(&min, &max);
            shouldEqual(min, 0);
            shouldEqual(max, 0);

            j = g.get_vertex_iterator();
            for(; j != end; ++j)
            {
                shouldEqual(vertexMap[*j], g.forward_degree(j));
                shouldEqual(edgeMap.bindInner(*j).template sum<int>(), g.back_degree(j));
            }
        }
    }
    
    template <class DirectedTag, NeighborhoodType NType>
    void testEdgeIterator()
    {
        using namespace boost_graph;
        
        static const bool directed = IsSameType<DirectedTag, directed_tag>::value;
        
        typedef GridGraph<N, DirectedTag> Graph;
        
        CoordinateIterator<N> i(Shape(3)), iend = i.getEndIterator();        
        for(; i != iend; ++i)
        {
            // create all possible array shapes from 1**N to 3**N
            Shape s = *i + Shape(1);
            Graph g(s, NType);
            
            ArrayND<N, int> sourceVertexMap(s),targetVertexMap(s);
            ArrayND<N+1, int> edgeMap(g.edge_propmap_shape()),
                                 edgeIdMap(g.edge_propmap_shape());
            
            shouldEqual((edgeMap.shape().template subarray<0, N>()), s);
            shouldEqual(edgeMap.shape(N), g.maxUniqueDegree());
            
            linearSequence(edgeIdMap.begin(), edgeIdMap.end());
            
            typename Graph::edge_iterator e = g.get_edge_iterator(),
                                          eend = g.get_edge_end_iterator();
            typename Graph::EdgeIt el(g);

            should(e == edges(g).first);
            should(eend == edges(g).second);
            
            int count = 0;
            int maxEdgeId = -1;
            for(; e != eend; ++e, ++el, ++count)
            {
                should(e.isValid() && !e.atEnd());
                should(e == el);
                should(e != lemon::INVALID);
                should(!(e == lemon::INVALID));

                put(edgeMap, *e, get(edgeMap, *e) + 1); // same as: edgeMap[*e] += 1;
                sourceVertexMap[source(*e, g)] += 1;
                targetVertexMap[target(*e, g)] += 1;
                
                shouldEqual(edgeIdMap[*e], g.id(el));
                shouldEqual(*el, g.edgeFromId(g.id(*el)));

                if(maxEdgeId < g.id(el))
                    maxEdgeId = g.id(el);
            }
            should(!e.isValid() && e.atEnd());
            should(e == el);
            should(e == lemon::INVALID);
            should(!(e != lemon::INVALID));
            
            shouldEqual(maxEdgeId, g.maxEdgeId());
            
            // check that all neighbors are found
            shouldEqual(count, g.num_edges());
            
            int min = NumericTraits<int>::max(), max = NumericTraits<int>::min();
            edgeMap.minmax(&min, &max);
            shouldEqual(min, 0);
            shouldEqual(max, g.num_edges() ? 1 : 0);
            
            CoordinateIterator<N> j(s), end = j.getEndIterator();
            for(; j != end; ++j)
            {
                if(directed)
                {
                    shouldEqual(edgeMap.bindInner(*j).template sum<int>(), g.out_degree(j));
                    shouldEqual(sourceVertexMap[*j], g.out_degree(j));
                    shouldEqual(targetVertexMap[*j], g.out_degree(j));
                }
                else
                {
                    shouldEqual(edgeMap.bindInner(*j).template sum<int>(), g.back_degree(j));
                    shouldEqual(sourceVertexMap[*j], g.back_degree(j));
                    shouldEqual(targetVertexMap[*j], g.forward_degree(j));
                }
            }
        }
    }
    
    template <class DirectedTag, NeighborhoodType NType>
    void testArcIterator()
    {
        using namespace boost_graph;
        
        static const bool directed = IsSameType<DirectedTag, directed_tag>::value;
        
        typedef GridGraph<N, DirectedTag> Graph;
        
        CoordinateIterator<N> i(Shape(3)), iend = i.getEndIterator();        
        for(; i != iend; ++i)
        {
            // create all possible array shapes from 1**N to 3**N
            Shape s = *i + Shape(1);
            Graph g(s, NType);
            
            typename Graph::template NodeMap<int> sourceVertexMap(g), targetVertexMap(g);
            typename Graph::template EdgeMap<int> edgeMap(g);
            typename Graph::template ArcMap<int> arcMap(g);
            
            shouldEqual(sourceVertexMap.shape(), s);
            shouldEqual(targetVertexMap.shape(), s);
            shouldEqual((edgeMap.shape().template subarray<0, N>()), s);
            shouldEqual(edgeMap.shape(N), g.maxUniqueDegree());
            shouldEqual((arcMap.shape().template subarray<0, N>()), s);
            shouldEqual(arcMap.shape(N), g.maxDegree());
            
            typename Graph::ArcIt e(g);

            int count = 0;
            int maxArcId = -1;
            for(; e != lemon::INVALID; ++e, ++count)
            {
                should(e.isValid() && !e.atEnd());
                should(e != lemon::INVALID);
                should(!(e == lemon::INVALID));

                sourceVertexMap[source(*e, g)] += 1;
                targetVertexMap[target(*e, g)] += 1;
                edgeMap[*e] += 1;
                arcMap[*e] += 1;
                
                shouldEqual(*e, g.arcFromId(g.id(*e)));
                
                if(maxArcId < g.id(e))
                    maxArcId = g.id(e);
            }
            should(!e.isValid() && e.atEnd());
            should(e == lemon::INVALID);
            should(!(e != lemon::INVALID));
            
            shouldEqual(maxArcId, g.maxArcId());
            
            // check that all neighbors are found
            shouldEqual(count, g.arcNum());
            shouldEqual(count, directed ? g.edgeNum() : 2*g.edgeNum());
            
            int min = NumericTraits<int>::max(), max = NumericTraits<int>::min();
            edgeMap.minmax(&min, &max);
            shouldEqual(min, 0);
            shouldEqual(max, g.arcNum() ? directed ? 1 : 2 : 0);
            
            min = NumericTraits<int>::max();
            max = NumericTraits<int>::min();
            arcMap.minmax(&min, &max);
            shouldEqual(min, 0);
            shouldEqual(max, g.arcNum() ? 1 : 0);
            
            CoordinateIterator<N> j(s), end = j.getEndIterator();
            for(; j != end; ++j)
            {
                shouldEqual(edgeMap.bindInner(*j).template sum<int>(), directed ? g.out_degree(j) : 2*g.back_degree(j));
                shouldEqual(arcMap.bindInner(*j).template sum<int>(), g.out_degree(j));
                shouldEqual(sourceVertexMap[*j], g.out_degree(j));
                shouldEqual(targetVertexMap[*j], g.out_degree(j));
            }
        }
    }
};

template <unsigned int N>
struct GridGraphAlgorithmTests
{
    typedef typename MultiArrayShape<N>::type Shape;
    
    template <class DirectedTag, NeighborhoodType NType>
    void testLocalMinMax()
    {
        typedef GridGraph<N, DirectedTag> Graph;
        
        Graph g(Shape(3), NType);
        typename Graph::template NodeMap<int> src(g), dest(g);
        
        src[Shape(1)] = 1;
        
        should(1 == boost_graph::localMinMaxGraph(g, src, dest, 1, -9999, std::greater<int>()));
        
        shouldEqualSequence(src.begin(), src.end(), dest.begin());
        
        dest.init(0);
        
        should(1 == lemon_graph::localMinMaxGraph(g, src, dest, 1, -9999, std::greater<int>()));
        
        shouldEqualSequence(src.begin(), src.end(), dest.begin());
    }
};
#endif

template <int N>
struct GridgraphTestSuiteN
: public vigra::test_suite
{
    GridgraphTestSuiteN()
    : vigra::test_suite((std::string("GridGraph<") + std::to_string(N) + ">").c_str())
    {
        add(testCase(&NeighborhoodTests<N>::testVertexIterator));

        add(testCase((&NeighborhoodTests<N>::template testNeighborhood<DirectNeighborhood, F_ORDER>)));
        add(testCase((&NeighborhoodTests<N>::template testNeighborhood<DirectNeighborhood, C_ORDER>)));
        add(testCase((&NeighborhoodTests<N>::template testNeighborhood<IndirectNeighborhood, F_ORDER>)));
        add(testCase((&NeighborhoodTests<N>::template testNeighborhood<IndirectNeighborhood, C_ORDER>)));
        add(testCase((&NeighborhoodTests<N>::template testNeighborhoodIterator<DirectNeighborhood, F_ORDER>)));
        add(testCase((&NeighborhoodTests<N>::template testNeighborhoodIterator<DirectNeighborhood, C_ORDER>)));
        add(testCase((&NeighborhoodTests<N>::template testNeighborhoodIterator<IndirectNeighborhood, F_ORDER>)));
        add(testCase((&NeighborhoodTests<N>::template testNeighborhoodIterator<IndirectNeighborhood, C_ORDER>)));
#if 0
        
        add(testCase(&NeighborhoodTests<N>::template testOutArcIteratorDirected<DirectNeighborhood>));
        add(testCase(&NeighborhoodTests<N>::template testOutArcIteratorDirected<IndirectNeighborhood>));
        
        add(testCase(&NeighborhoodTests<N>::template testOutArcIteratorUndirected<DirectNeighborhood>));
        add(testCase(&NeighborhoodTests<N>::template testOutArcIteratorUndirected<IndirectNeighborhood>));
        
        add(testCase(&NeighborhoodTests<N>::template testArcIteratorDirected<DirectNeighborhood>));
        add(testCase(&NeighborhoodTests<N>::template testArcIteratorDirected<IndirectNeighborhood>));
        
        add(testCase(&NeighborhoodTests<N>::template testArcIteratorUndirected<DirectNeighborhood>));
        add(testCase(&NeighborhoodTests<N>::template testArcIteratorUndirected<IndirectNeighborhood>));
        
        add(testCase((&GridGraphTests<N>::template testBasics<directed_tag, IndirectNeighborhood>)));
        add(testCase((&GridGraphTests<N>::template testBasics<undirected_tag, IndirectNeighborhood>)));
        add(testCase((&GridGraphTests<N>::template testBasics<directed_tag, DirectNeighborhood>)));
        add(testCase((&GridGraphTests<N>::template testBasics<undirected_tag, DirectNeighborhood>)));
        
        add(testCase((&GridGraphTests<N>::template testVertexIterator<directed_tag, IndirectNeighborhood>)));
        add(testCase((&GridGraphTests<N>::template testVertexIterator<undirected_tag, IndirectNeighborhood>)));
        add(testCase((&GridGraphTests<N>::template testVertexIterator<directed_tag, DirectNeighborhood>)));
        add(testCase((&GridGraphTests<N>::template testVertexIterator<undirected_tag, DirectNeighborhood>)));
        
        add(testCase((&GridGraphTests<N>::template testNeighborIterator<directed_tag, IndirectNeighborhood>)));
        add(testCase((&GridGraphTests<N>::template testNeighborIterator<undirected_tag, IndirectNeighborhood>)));
        add(testCase((&GridGraphTests<N>::template testNeighborIterator<directed_tag, DirectNeighborhood>)));
        add(testCase((&GridGraphTests<N>::template testNeighborIterator<undirected_tag, DirectNeighborhood>)));
        
        add(testCase((&GridGraphTests<N>::template testBackNeighborIterator<directed_tag, IndirectNeighborhood>)));
        add(testCase((&GridGraphTests<N>::template testBackNeighborIterator<undirected_tag, IndirectNeighborhood>)));
        add(testCase((&GridGraphTests<N>::template testBackNeighborIterator<directed_tag, DirectNeighborhood>)));
        add(testCase((&GridGraphTests<N>::template testBackNeighborIterator<undirected_tag, DirectNeighborhood>)));
        
        add(testCase((&GridGraphTests<N>::template testEdgeIterator<directed_tag, IndirectNeighborhood>)));
        add(testCase((&GridGraphTests<N>::template testEdgeIterator<undirected_tag, IndirectNeighborhood>)));
        add(testCase((&GridGraphTests<N>::template testEdgeIterator<directed_tag, DirectNeighborhood>)));
        add(testCase((&GridGraphTests<N>::template testEdgeIterator<undirected_tag, DirectNeighborhood>)));
        
        add(testCase((&GridGraphTests<N>::template testArcIterator<directed_tag, IndirectNeighborhood>)));
        add(testCase((&GridGraphTests<N>::template testArcIterator<undirected_tag, IndirectNeighborhood>)));
        add(testCase((&GridGraphTests<N>::template testArcIterator<directed_tag, DirectNeighborhood>)));
        add(testCase((&GridGraphTests<N>::template testArcIterator<undirected_tag, DirectNeighborhood>)));
        
        add(testCase((&GridGraphAlgorithmTests<N>::template testLocalMinMax<undirected_tag, DirectNeighborhood>)));
#endif
    }
};

struct GridgraphTestSuite
: public vigra::test_suite
{
    GridgraphTestSuite()
    : vigra::test_suite("GridgraphTestSuite")
    {
        add(VIGRA_TEST_SUITE(GridgraphTestSuiteN<2>));
        add(VIGRA_TEST_SUITE(GridgraphTestSuiteN<3>));
        add(VIGRA_TEST_SUITE(GridgraphTestSuiteN<runtime_size>));
        //        add(VIGRA_TEST_SUITE(GridgraphTestSuiteN<4>));
    }
};

int main(int argc, char **argv)
{

    GridgraphTestSuite gridgraphTest;

    int failed = gridgraphTest.run(vigra::testsToBeExecuted(argc, argv));

    std::cout << gridgraphTest.report() << std::endl;

    return (failed != 0);
}

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
