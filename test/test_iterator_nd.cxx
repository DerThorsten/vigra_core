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
    typedef Shape<N>                         S;
    typedef CoordinateIterator<N, F_ORDER>   CIF;
    typedef CoordinateIterator<N, C_ORDER>   CIC;
    typedef CoordinateIterator<N>            CID;
    S s{ 4,3,2 };
    S slarge{ 200,200,200 };

    IteratorNDTest()
    {
    }

    void testCoordinateIterator()
    {
        {
            CID diter(s, C_ORDER), dend(diter.end()), drend(diter.rend()), dbegin(dend.begin()), drbegin(diter.rbegin());
            CIC citer(s), cend(citer.end()), crend(citer.rend()), cbegin(cend.begin()), crbegin(citer.rbegin());

            shouldEqual(diter.shape(), s);
            shouldEqual(citer.shape(), s);
            shouldEqual(dend.coord(), (S{ 4,0,0 }));
            shouldEqual(cend.coord(), (S{ 4,0,0 }));
            shouldEqual(drend.coord(), (S{ -1, 2, 1 }));
            shouldEqual(crend.coord(), (S{ -1, 2, 1 }));
            shouldEqual(dbegin.coord(), (S{ 0,0,0 }));
            shouldEqual(cbegin.coord(), (S{ 0,0,0 }));
            shouldEqual(drbegin.coord(), (S{ 3,2,1 }));
            shouldEqual(crbegin.coord(), (S{ 3,2,1 }));

            for (int i = 0; i < s[0]; ++i)
                for (int j = 0; j < s[1]; ++j)
                    for (int k = 0; k < s[2]; ++k, ++diter, ++citer)
                    {
                        shouldEqual(*diter, (S{ i,j,k }));
                        shouldEqual(*citer, (S{ i,j,k }));
                        should(diter != dend);
                        should(citer != cend);
                        should(diter.isValid());
                        should(citer.isValid());
                        shouldNot(diter.atEnd());
                        shouldNot(citer.atEnd());
                    }

            should(diter == dend);
            should(citer == cend);
            shouldNot(diter.isValid());
            shouldNot(citer.isValid());
            should(diter.atEnd());
            should(citer.atEnd());

            --diter;
            --citer;
            for (int i = s[0] - 1; i >= 0; --i)
                for (int j = s[1] - 1; j >= 0; --j)
                    for (int k = s[2] - 1; k >= 0; --k, --diter, --citer)
                    {
                        shouldEqual(*diter, (S{ i,j,k }));
                        shouldEqual(*citer, (S{ i,j,k }));
                        should(diter != drend);
                        should(citer != crend);
                        should(diter.isValid());
                        should(citer.isValid());
                        shouldNot(diter.atEnd());
                        shouldNot(citer.atEnd());
                    }
            should(diter == drend);
            should(citer == crend);
            shouldNot(diter.isValid());
            shouldNot(citer.isValid());
            should(diter.atEnd());
            should(citer.atEnd());
        }

        {
            CID diter(s, F_ORDER), dend(diter.end()), drend(diter.rend()), dbegin(dend.begin()), drbegin(diter.rbegin());
            CIF fiter(s), fend(fiter.end()), frend(fiter.rend()), fbegin(fend.begin()), frbegin(fiter.rbegin());

            shouldEqual(diter.shape(), s);
            shouldEqual(fiter.shape(), s);
            shouldEqual(dend.coord(), (S{ 0,0,2 }));
            shouldEqual(fend.coord(), (S{ 0,0,2 }));
            shouldEqual(drend.coord(), (S{ 3,2,-1 }));
            shouldEqual(frend.coord(), (S{ 3,2,-1 }));
            shouldEqual(dbegin.coord(), (S{ 0,0,0 }));
            shouldEqual(fbegin.coord(), (S{ 0,0,0 }));
            shouldEqual(drbegin.coord(), (S{ 3,2,1 }));
            shouldEqual(frbegin.coord(), (S{ 3,2,1 }));

            for (int i = 0; i < s[2]; ++i)
                for (int j = 0; j < s[1]; ++j)
                    for (int k = 0; k < s[0]; ++k, ++diter, ++fiter)
                    {
                        shouldEqual(*diter, (S{ k,j,i }));
                        shouldEqual(*fiter, (S{ k,j,i }));
                        should(diter != dend);
                        should(fiter != fend);
                        should(diter.isValid());
                        should(fiter.isValid());
                        shouldNot(diter.atEnd());
                        shouldNot(fiter.atEnd());
                    }

            should(diter == dend);
            should(fiter == fend);
            shouldNot(diter.isValid());
            shouldNot(fiter.isValid());
            should(diter.atEnd());
            should(fiter.atEnd());

            --diter;
            --fiter;
            for (int i = s[2] - 1; i >= 0; --i)
                for (int j = s[1] - 1; j >= 0; --j)
                    for (int k = s[0] - 1; k >= 0; --k, --diter, --fiter)
                    {
                        shouldEqual(*diter, (S{ k,j,i }));
                        shouldEqual(*fiter, (S{ k,j,i }));
                        should(diter != drend);
                        should(fiter != frend);
                        should(diter.isValid());
                        should(fiter.isValid());
                        shouldNot(diter.atEnd());
                        shouldNot(fiter.atEnd());
                    }
            should(diter == drend);
            should(fiter == frend);
            shouldNot(diter.isValid());
            shouldNot(fiter.isValid());
            should(diter.atEnd());
            should(fiter.atEnd());
        }

        {
            CID diter(s, S{ 2,0,1 }), dend(diter.end()), drend(diter.rend()), dbegin(dend.begin()), drbegin(diter.rbegin());

            shouldEqual(diter.shape(), s);
            shouldEqual(dend.coord(), (S{ 0,3,0 }));
            shouldEqual(drend.coord(), (S{ 3,-1,1 }));
            shouldEqual(dbegin.coord(), (S{ 0,0,0 }));
            shouldEqual(drbegin.coord(), (S{ 3,2,1 }));

            for (int i = 0; i < s[1]; ++i)
                for (int j = 0; j < s[0]; ++j)
                    for (int k = 0; k < s[2]; ++k, ++diter)
                    {
                        shouldEqual(*diter, (S{ j,i,k }));
                        should(diter != dend);
                        should(diter.isValid());
                        shouldNot(diter.atEnd());
                    }

            should(diter == dend);
            shouldNot(diter.isValid());
            should(diter.atEnd());

            --diter;
            for (int i = s[1] - 1; i >= 0; --i)
                for (int j = s[0] - 1; j >= 0; --j)
                    for (int k = s[2] - 1; k >= 0; --k, --diter)
                    {
                        shouldEqual(*diter, (S{ j,i,k }));
                        should(diter != drend);
                        should(diter.isValid());
                        shouldNot(diter.atEnd());
                    }
            should(diter == drend);
            shouldNot(diter.isValid());
            should(diter.atEnd());
        }
    }

    void testSpeed()
    {
        //CID iter(slarge, C_ORDER), end(iter.end());
        //CIF iter(slarge), end(iter.end());
        CIC iter(slarge), end(iter.end());

        //std::cerr << iter.shape() << " " << end.shape() << " " << end.coord() << "\n";
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
        addTests<runtime_size>();
    }

    template <int N>
    void addTests()
    {
        add(testCase(&IteratorNDTest<N>::testCoordinateIterator));
        add(testCase(&IteratorNDTest<N>::testSpeed));
    }
};

int main(int argc, char ** argv)
{
    IteratorNDTestSuite test;

    int failed = test.run(vigra::testsToBeExecuted(argc, argv));

    std::cout << test.report() << std::endl;

    return (failed != 0);
}
