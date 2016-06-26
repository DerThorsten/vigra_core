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

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <typeinfo>
#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include <vigra2/unittest.hxx>
#include <vigra2/timing.hxx>
#include <vigra2/iterator_nd.hxx>

using namespace vigra;

template <int N>
struct IteratorNDTest
{
    typedef Shape<N>                S;
    typedef CoordinateIterator<N, F_ORDER>   CIF;
    typedef CoordinateIterator<N, C_ORDER>   CIC;
    typedef CoordinateIterator<N>   CIR;
    //S s{ 4,3,2 };
    S s{ 200,200,200 };

    IteratorNDTest()
    {
    }

    void testCoordinateIterator()
    {
        CIR iter(s, C_ORDER), end(iter.getEndIterator());
        //CIF iter(s), end(iter.getEndIterator());
        //CIC iter(s), end(iter.getEndIterator());

        std::cerr << iter.shape() << " " << end.shape() << " " << end.coord() << "\n";
        //std::cerr << iter.recursion_.axes_ << " " << iter.recursion_.minor() << "\n";
        //return;

        // while(iter.isValid())
        // {
            // std::cerr << *iter << "\n";
            // ++iter;
        // }
        USETICTOC;
        TIC;
        int count = 0;
        for (; iter.isValid(); ++iter)
        //for (; iter != end; ++iter)
        {
            count += (*iter)[0];
        }
        TOC;
        std::cerr << count << "\n";
    }

};

struct IteratorNDTestSuite
: public vigra::test_suite
{
    IteratorNDTestSuite()
    : vigra::test_suite("IteratorNDTest")
    {
        addTests<3>();
        // addTests<runtime_size>();
    }

    template <int N>
    void addTests()
    {
        add(testCase(&IteratorNDTest<N>::testCoordinateIterator));
    }
};

int main(int argc, char ** argv)
{
    IteratorNDTestSuite test;

    int failed = test.run(vigra::testsToBeExecuted(argc, argv));

    std::cout << test.report() << std::endl;

    return (failed != 0);
}
