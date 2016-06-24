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
#include <initializer_list>
#include <vigra2/unittest.hxx>
#include <vigra2/concepts.hxx>

using namespace vigra;

struct ConceptTest
{
    struct ThisIsAnArrayND
    : public ArrayNDTag
    {};

    struct ThisIsNoArrayND
    {};

    int
    checkEnableIfArray(...)
    {
        return 1;
    }

    template <class A>
    enable_if_t<ArrayNDConcept<A>::value, int>
    checkEnableIfArray(A *)
    {
        return 2;
    }

    void test()
    {
        shouldEqual(ArrayNDConcept<int>::value, false);
        shouldEqual(ArrayNDConcept<std::string>::value, false);
        shouldEqual(ArrayNDConcept<ThisIsNoArrayND>::value, false);
        shouldEqual(ArrayNDConcept<ThisIsAnArrayND>::value, true);

        shouldEqual(checkEnableIfArray((int*)0), 1);
        shouldEqual(checkEnableIfArray((std::string*)0), 1);
        shouldEqual(checkEnableIfArray((ThisIsNoArrayND*)0), 1);
        shouldEqual(checkEnableIfArray((ThisIsAnArrayND*)0), 2);

        shouldEqual(IsIterator<int>::value, false);
        shouldEqual(IsIterator<int*>::value, true);
        shouldEqual(IsIterator<std::string>::value, false);
        shouldEqual(IsIterator<decltype(std::string().begin())>::value, true);

        typedef int IntArray[3];
        shouldEqual(IsIterator<IntArray>::value, true);
        shouldEqual(IsIterator<std::initializer_list<int>::iterator>::value, true);
    }
};

struct ConceptTestSuite
: public vigra::test_suite
{
    ConceptTestSuite()
    : vigra::test_suite("ConceptTestSuite")
    {
        add( testCase(&ConceptTest::test));
    }
};

int main(int argc, char ** argv)
{
    ConceptTestSuite test;

    int failed = test.run(vigra::testsToBeExecuted(argc, argv));

    std::cout << test.report() << std::endl;

    return (failed != 0);
}
