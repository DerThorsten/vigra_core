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

#include <typeinfo>
#include <iostream>
#include <string>
#include <numeric>
#include <functional>
#include <vigra2/unittest.hxx>
#include <vigra2/algorithm_nd.hxx>
#include <vigra2/timing.hxx>

using namespace vigra;

template <int N>
struct AlgorithmNDTest
{
    typedef Shape<N> S;

    AlgorithmNDTest()
    {
    }

    void testForeachCoordinate()
    {
        S shape    = {4,3,2},
          cstrides = shapeToStrides(shape, C_ORDER),
          fstrides = shapeToStrides(shape, F_ORDER);

        should(cstrides != fstrides);

        int count = 0;
        foreachCoordinate(shape, [&count, cstrides](S const & coord) {
            shouldEqual(dot(coord, cstrides), count);
            ++count;
        });
        shouldEqual(count, 24);

        count = 0;
        foreachCoordinate(S{3,3,3}, shape+3, [&count, cstrides](S const & coord) {
            shouldEqual(dot(coord-3, cstrides), count);
            ++count;
        });
        shouldEqual(count, 24);

        count = 0;
        foreachCoordinate(shape, [&count, fstrides](S const & coord) {
            shouldEqual(dot(coord, fstrides), count);
            ++count;
        }, F_ORDER);
        shouldEqual(count, 24);

        count = 0;
        foreachCoordinate(S{3,3,3}, shape+3, [&count, fstrides](S const & coord) {
            shouldEqual(dot(coord-3, fstrides), count);
            ++count;
        }, F_ORDER);
        shouldEqual(count, 24);

        S ustrides = { 6, 1, 3 },
          order    = { 1, 2, 0 };

        count = 0;
        foreachCoordinate(shape, [&count, ustrides](S const & coord) {
            shouldEqual(dot(coord, ustrides), count);
            ++count;
        }, order);
        shouldEqual(count, 24);

        count = 0;
        foreachCoordinate(S{3,3,3}, shape+3, [&count, ustrides](S const & coord) {
            shouldEqual(dot(coord-3, ustrides), count);
            ++count;
        }, order);
        shouldEqual(count, 24);
    }

    void testAlgorithmND()
    {
        S shape    = {4,3,2};

        ArrayND<N, int> a(shape), b(shape);
        std::iota(a.begin(), a.end(), 0);

        int count = 0;
        foreachND(a, [&count](int i) {
            count += i;
        });

        shouldEqual(count, 276);

        transformND(a, b, [](int i) { return -i; });
        using namespace array_math;
        should(a == -b);

        transformND(a, b, b, std::plus<int>());
        should(b == 0);
    }

    void testSpeed()
    {
        USETICTOC;
        S shape    = {200,200,200},
          strides = shapeToStrides(shape, C_ORDER);

        int count = 0;
        std::cerr << "C order: ";
        TIC;
        foreachCoordinate(shape, [&count, strides](S const & coord) {
            count += dot(coord, strides);
        });
        TOC;

        count = 0;
        std::cerr << "F order: ";
        TIC;
        foreachCoordinate(shape, [&count, strides](S const & coord) {
            count += dot(coord, strides);
        }, F_ORDER);
        TOC;

        S order    = { 1, 2, 0 };
        count = 0;
        std::cerr << "custom order: ";
        TIC;
        foreachCoordinate(shape, [&count, strides](S const & coord) {
            count += dot(coord, strides);
        }, order);
        TOC;
    }
};

struct AlgorithmNDTestSuite
: public vigra::test_suite
{
    AlgorithmNDTestSuite()
    : vigra::test_suite("AlgorithmNDTestSuite")
    {
        addTests<3>();
        addTests<runtime_size>();
    }

    template <int N>
    void addTests()
    {
        add( testCase(&AlgorithmNDTest<N>::testForeachCoordinate));
        add( testCase(&AlgorithmNDTest<N>::testAlgorithmND));
        add( testCase(&AlgorithmNDTest<N>::testSpeed));
    }
};

int main(int argc, char ** argv)
{
    AlgorithmNDTestSuite test;

    int failed = test.run(vigra::testsToBeExecuted(argc, argv));

    std::cout << test.report() << std::endl;

    return (failed != 0);
}
