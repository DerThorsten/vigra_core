/************************************************************************/
/*                                                                      */
/*               Copyright 2016-2017 by Ullrich Koethe                  */
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

#ifndef VIGRA2_CONVOLUTION_KERNELS_HXX
#define VIGRA2_CONVOLUTION_KERNELS_HXX

#include "config.hxx"
#include "gaussians.hxx"
#include "error.hxx"
#include "numeric_traits.hxx"
#include "mathutil.hxx"
#include "tags.hxx"
// #include <algorithm>
// #include <functional>
#include <vector>

namespace vigra {

/** \addtogroup SeparableConvolution One-dimensional and separable convolution functionality
*/
//@{

/********************************************************/
/*                                                      */
/*                      Kernel1D                        */
/*                                                      */
/********************************************************/

/** \brief Generic 1 dimensional convolution kernel.

    This kernel may be used for convolution of 1 dimensional signals or for
    separable convolution of multidimensional signals.

    Convolution functions access the kernel via a 1 dimensional random access
    iterator which they get by calling \ref center(). This iterator
    points to the center of the kernel. The kernel's size is given by its left() (<=0)
    and right() (>= 0) methods. The desired border treatment mode is
    returned by borderTreatment().

    The different init functions create a kernel with the specified
    properties. The kernel's value_type must be a linear space, i.e. it
    must define multiplication with doubles and NumericTraits.

    <b> Usage:</b>

    <b>\#include</b> \<vigra/separableconvolution.hxx\><br/>
    Namespace: vigra

    \code
    MultiArray<2, float> src(w,h), dest(w,h);
    ...

    // define Gaussian kernel with std. deviation 3.0
    Kernel1D kernel;
    kernel.initGaussian(3.0);

    // apply 1D kernel along the x-axis
    separableConvolveX(src, dest, kernel);
    \endcode
*/

template <class ARITHTYPE = double>
class Kernel1D
{
  public:
    typedef std::vector<ARITHTYPE> InternalVector;

        /** the kernel's value type
        */
    typedef typename InternalVector::value_type value_type;

        /** the kernel's reference type
        */
    typedef typename InternalVector::reference reference;

        /** the kernel's const reference type
        */
    typedef typename InternalVector::const_reference const_reference;

        /** 1D random access iterator over the kernel's values
        */
    typedef typename InternalVector::pointer iterator;

        /** const 1D random access iterator over the kernel's values
        */
    typedef typename InternalVector::const_pointer const_iterator;

        // helper class to enable initialization via overloaded operator,()
    friend struct InitProxy;

    struct InitProxy
    {
        InitProxy(Kernel1D & kernel, int count)
        : kernel_(kernel)
        , iter_(kernel.kernel_.data())
        , count_(count)
        , sum_(count)
        , norm_(kernel.norm_)
        {}

        ~InitProxy()
#ifndef _MSC_VER
            throw(ContractViolation)
#elif _MSC_VER >= 1900
            noexcept(false)
#endif
        {
            vigra_precondition(count_ == 1 || count_ == sum_,
                  "Kernel1D::initExplicitly(): "
                  "Wrong number of init values.");
            kernel_.norm_ = norm_;
            kernel_.setSymmetry();
        }

        InitProxy & operator,(value_type const & v)
        {
            if(sum_ == count_)
                norm_ = *iter_;

            norm_ += v;

            --count_;

            if(count_ > 0)
            {
                ++iter_;
                *iter_ = v;
            }
            return *this;
        }

        Kernel1D & kernel_;
        iterator iter_;
        int count_, sum_;
        value_type norm_;
    };

    static value_type one() { return NumericTraits<value_type>::one(); }

        /** Default constructor.
            Creates an identity kernel, i.e. a kernel of size 1 which copies the signal
            unchanged.
        */
    Kernel1D()
    : kernel_(1, one())
    , left_(0)
    , right_(0)
    , border_treatment_(BORDER_TREATMENT_REFLECT)
    , norm_(one())
    , symmetry_(KernelEven)
    {}

        /** Construct from kernel with different element type, e.g. double => FixedPoint16.
        */
    template <class U>
    Kernel1D(Kernel1D<U> const & k)
    : kernel_(k.center()+k.left(), k.center()+k.right()+1)
    , left_(k.left())
    , right_(k.right())
    , border_treatment_(k.borderTreatment())
    , norm_(k.norm())
    , symmetry_(k.symmetry_)
    {}

        /** Copy assignment.
        */
    Kernel1D & operator=(Kernel1D const & k) = default;

        /** Copy constructor.
        */
    Kernel1D(Kernel1D const & k) = default;

        /** Initialization.
            This initializes the kernel with the given constant. The norm becomes
            v*size().

            Instead of a single value an initializer list of length size()
            can be used like this:

            \code
            vigra::Kernel1D<float> roberts_gradient_x;

            roberts_gradient_x.initExplicitly(0, 1) = 1.0, -1.0;
            \endcode

            In this case, the norm will be set to the sum of the init values.
            An initializer list of wrong length will result in a run-time error.
        */
    InitProxy operator=(value_type const & v)
    {
        int size = right_ - left_ + 1;
        for(int i=0; i<size; ++i)
            kernel_[i] = v;
        norm_ = (double)size*v;

        return InitProxy(*this, size);
    }

    template <class T>
    Kernel1D & operator=(std::initializer_list<T> const & v)
    {
        int size = right_ - left_ + 1;
        vigra_precondition(v.size() == size,
            "Kernel1D::operator=(std::initializer_list<T>): Wrong number of init values.");

        norm_ = value_type();
        auto iv = v.begin();
        for(int k=0; k<size; ++k, ++iv)
        {
            kernel_[k] = detail::RequiresExplicitCast<value_type>::cast(*iv);
            norm_ += kernel_[k];
        }

        setSymmetry();
        return *this;
    }

        /** Destructor.
        */
    ~Kernel1D()
    {}

        /** \brief Init as a sampled Gaussian function.

            '<tt>norm</tt>' (default: 1.0) denotes the sum of all bins of the kernel
            (i.e. the kernel is corrected for the normalization error introduced
            by windowing the Gaussian to a finite interval). However,
            if <tt>norm</tt> is 0.0, the kernel is normalized to 1 by the analytic
            expression for the Gaussian, and <b>no</b> correction for the windowing
            error is performed. If <tt>window_ratio = 0.0</tt>, the radius of the filter
            window is <tt>radius = round(3.0 * std_dev)</tt>, otherwise it is
            <tt>radius = round(window_ratio * std_dev)</tt> (where <tt>window_ratio > 0.0</tt>
            is required).

            Precondition:
            \code
            std_dev >= 0.0
            \endcode

            Postconditions:
            \code
            1. left()  == -(int)(3.0*std_dev + 0.5)
            2. right() ==  (int)(3.0*std_dev + 0.5)
            3. borderTreatment() == BORDER_TREATMENT_REFLECT
            4. norm() == norm
            \endcode
        */
    void initGaussian(double std_dev, value_type norm, double window_ratio = 0.0);

        /** Init as a Gaussian function with norm 1.
         */
    void initGaussian(double std_dev)
    {
        initGaussian(std_dev, one());
    }


        /** \brief Init as Lindeberg's discrete analog of the Gaussian function.

            The radius of the kernel is always <tt>round(3*std_dev)</tt>. 'norm' denotes
            the sum of all bins of the kernel (default: 1.0).

            Precondition:
            \code
            std_dev >= 0.0
            \endcode

            Postconditions:
            \code
            1. left()  == -(int)(3.0*std_dev + 0.5)
            2. right() ==  (int)(3.0*std_dev + 0.5)
            3. borderTreatment() == BORDER_TREATMENT_REFLECT
            4. norm() == norm
            \endcode
        */
    void initDiscreteGaussian(double std_dev, value_type norm);

        /** \brief Init as Lindeberg's discrete analog of the Gaussian function
            with norm 1.
         */
    void initDiscreteGaussian(double std_dev)
    {
        initDiscreteGaussian(std_dev, one());
    }

        /** \brief Init as a Gaussian derivative of order '<tt>order</tt>'.

            '<tt>norm</tt>' (default: 1.0) denotes the norm of the kernel so
            that the following condition is fulfilled:

            \f[ \sum_{i=left()}^{right()}
                         \frac{(-i)^{order}kernel[i]}{order!} = norm
            \f]

            Thus, the kernel will be corrected for the error introduced
            by windowing the Gaussian to a finite interval. However,
            if <tt>norm</tt> is 0.0, the kernel is normalized to 1 by the analytic
            expression for the Gaussian derivative, and <b>no</b> correction for the
            windowing error is performed. If <tt>window_ratio = 0.0</tt>, the radius
            of the filter window is <tt>radius = round(3.0 * std_dev + 0.5 * order)</tt>,
            otherwise it is <tt>radius = round(window_ratio * std_dev)</tt> (where
            <tt>window_ratio > 0.0</tt> is required).

            Preconditions:
            \code
            1. std_dev >= 0.0
            2. order   >= 1
            \endcode

            Postconditions:
            \code
            1. left()  == -(int)(3.0*std_dev + 0.5*order + 0.5)
            2. right() ==  (int)(3.0*std_dev + 0.5*order + 0.5)
            3. borderTreatment() == BORDER_TREATMENT_REFLECT
            4. norm() == norm
            \endcode
        */
    void initGaussianDerivative(double std_dev, int order, value_type norm, double window_ratio = 0.0);

        /** \brief Init as a Gaussian derivative with norm 1.
         */
    void initGaussianDerivative(double std_dev, int order)
    {
        initGaussianDerivative(std_dev, order, one());
    }

        /** \brief Init an optimal 3-tap smoothing filter.

            The filter coefficients are

            \code
            [0.216, 0.568, 0.216]
            \endcode

            These values are optimal in the sense that the 3x3 filter obtained by
            separable application of this filter is the best possible 3x3 approximation
            to a Gaussian filter. The equivalent Gaussian has sigma = 0.680.

            Postconditions:
            \code
            1. left()  == -1
            2. right() ==  1
            3. borderTreatment() == BORDER_TREATMENT_REFLECT
            4. norm() == 1.0
            \endcode
        */
    void initOptimalSmoothing3()
    {
        this->initExplicitly(-1, 1) = { 0.216, 0.568, 0.216 };
        this->setBorderTreatment(BORDER_TREATMENT_REFLECT);
        this->symmetry_ = KernelEven;
    }

        /** \brief Init an optimal 3-tap smoothing filter for the Scharr derivative.

            The filter coefficients are

            \code
            [ 0.183, 0.634, 0.183 ]
            \endcode

            These values are optimal in the sense that the 3x3 derivative obtained
            by combining this filter with the symmetric difference gives the best
            approximation to rotational equivariance (steerability) among all 3x3 first
            derivative filters, see

            Hanno Scharr: Optimale Operatoren in der Digitalen Bildverarbeitung. Dissertation: Ruprecht-Karls-Universität Heidelberg, 2000

            Postconditions:
            \code
            1. left()  == -1
            2. right() ==  1
            3. borderTreatment() == BORDER_TREATMENT_REFLECT
            4. norm() == 1.0
            \endcode
        */
    void initOptimalFirstDerivativeSmoothing3()
    {
        this->initExplicitly(-1, 1) = { 0.183, 0.634, 0.183 };
        this->setBorderTreatment(BORDER_TREATMENT_REFLECT);
        this->symmetry_ = KernelEven;
    }


        /** \brief Init an optimal 3-tap smoothing filter for the Scharr Laplacian.

            The filter coefficients are

            \code
            [ 0.0939, 0.8122, 0.0939 ]
            \endcode

            These values are optimal in the sense that the 3x3 Laplacian constructed
            by combining this filter with the second difference gives the best
            approximation to rotational invariance among all 3x3 Laplacian filters,
            see

            Hanno Scharr: Optimale Operatoren in der Digitalen Bildverarbeitung. Dissertation: Ruprecht-Karls-Universität Heidelberg, 2000

            Postconditions:
            \code
            1. left()  == -1
            2. right() ==  1
            3. borderTreatment() == BORDER_TREATMENT_REFLECT
            4. norm() == 1.0
            \endcode
        */
    void initOptimalSecondDerivativeSmoothing3()
    {
        this->initExplicitly(-1, 1) = { 0.0939, 0.8122, 0.0939 };
        this->setBorderTreatment(BORDER_TREATMENT_REFLECT);
        this->symmetry_ = KernelEven;
    }

        /** \brief Init an optimal 5-tap smoothing filter.

            The filter coefficients are

            \code
            [0.03134, 0.24, 0.45732, 0.24, 0.03134]
            \endcode

            These values are optimal in the sense that the 5x5 filter obtained by separable
            application of this filter is the best possible 5x5 approximation to a
            Gaussian filter. The equivalent Gaussian has sigma = 0.867.

            Postconditions:
            \code
            1. left()  == -2
            2. right() ==  2
            3. borderTreatment() == BORDER_TREATMENT_REFLECT
            4. norm() == 1.0
            \endcode
        */
    void initOptimalSmoothing5()
    {
        this->initExplicitly(-2, 2) = { 0.03134, 0.24, 0.45732, 0.24, 0.03134 };
        this->setBorderTreatment(BORDER_TREATMENT_REFLECT);
        this->symmetry_ = KernelEven;
    }

        /** \brief Init an optimal 5-tap smoothing filter for the Scharr derivative.

            This filter must be used in conjunction with the optimal 5-tap first derivative filter (see initOptimalFirstDerivative5()), such that the derivative
            filter is applied along one dimension, and the smoothing filter along the other.
            The filter coefficients are

            \code
            [ 0.0231, 0.2413, 0.4712, 0.2413, 0.0231 ]
            \endcode

            These values are optimal in the sense that the 5x5 filter obtained by combining
            this filter with the optimal 5-tap first derivative gives the best possible
            approximation of rotational equivariance (steerability) among all
            5x5 first derivatives, see

            Hanno Scharr: Optimale Operatoren in der Digitalen Bildverarbeitung. Dissertation: Ruprecht-Karls-Universität Heidelberg, 2000

            Postconditions:
            \code
            1. left()  == -2
            2. right() ==  2
            3. borderTreatment() == BORDER_TREATMENT_REFLECT
            4. norm() == 1.0
            \endcode
        */
    void initOptimalFirstDerivativeSmoothing5()
    {
        this->initExplicitly(-2, 2) = { 0.0231, 0.2413, 0.4712, 0.2413, 0.0231 };
        this->setBorderTreatment(BORDER_TREATMENT_REFLECT);
        this->symmetry_ = KernelEven;
    }

        /** \brief Init an optimal 5-tap smoothing filter for the Scharr Laplacian.

            This filter must be used in conjunction with the optimal 5-tap second derivative
            filter (see initOptimalSecondDerivative5()), such that the derivative filter is
            applied along one dimension, and the smoothing filter along the other.
            The filter coefficients are

            \code
            [ 0.131, -0.0328, 0.8036, -0.0328, 0.131 ]
            \endcode

            These values are optimal in the sense that the 5x5 filter obtained by combining
            this filter with the optimal 5-tap second derivative gives the best possible
            rotation invariance among 5x5 Laplacian filters, see

            Hanno Scharr: Optimale Operatoren in der Digitalen Bildverarbeitung. Dissertation: Ruprecht-Karls-Universität Heidelberg, 2000

            Postconditions:
            \code
            1. left()  == -2
            2. right() ==  2
            3. borderTreatment() == BORDER_TREATMENT_REFLECT
            4. norm() == 1.0
            \endcode
        */
    void initOptimalSecondDerivativeSmoothing5()
    {
        this->initExplicitly(-2, 2) = { 0.131, -0.0328, 0.8036, -0.0328, 0.131 };
        this->setBorderTreatment(BORDER_TREATMENT_REFLECT);
        this->symmetry_ = KernelEven;
    }

        /** \brief Init a 5-tap filter as defined by Peter Burt in the context of pyramid creation.

            The filter coefficients are

            \code
            [a, 0.25, 0.5-2*a, 0.25, a]
            \endcode

            The default <tt>a = 0.04785</tt> is optimal in the sense that it minimizes the difference
            to a true Gaussian filter (which would have sigma = 0.975). For other values of <tt>a</tt>, the scale of the most similar Gaussian can be approximated by

            \code
            sigma = 5.1 * a + 0.731
            \endcode

            Preconditions:
            \code
            0 <= a <= 0.125
            \endcode

            Postconditions:
            \code
            1. left()  == -2
            2. right() ==  2
            3. borderTreatment() == BORDER_TREATMENT_REFLECT
            4. norm() == 1.0
            \endcode
        */
    void initBurtFilter(double a = 0.04785)
    {
        vigra_precondition(a >= 0.0 && a <= 0.125,
            "Kernel1D::initBurtFilter(): 0 <= a <= 0.125 required.");
        this->initExplicitly(-2, 2) = a, 0.25, 0.5 - 2.0*a, 0.25, a;
        this->setBorderTreatment(BORDER_TREATMENT_REFLECT);
        this->symmetry_ = KernelEven;
    }

        /** \brief Init a Binomial filter with given radius.

            'norm' (default: 1.0) denotes the sum of all bins of the kernel.

            Precondition:
            \code
            radius   >= 0
            \endcode

            Postconditions:
            \code
            1. left()  == -radius
            2. right() ==  radius
            3. borderTreatment() == BORDER_TREATMENT_REFLECT
            4. norm() == norm
            \endcode
        */
    void initBinomial(int radius, value_type norm);

        /** Init as a Binomial filter with norm 1.
         */
    void initBinomial(int radius)
    {
        initBinomial(radius, one());
    }

        /** \brief Init as an averaging filter.

            'norm' (default: 1.0) denotes the sum of all bins
            of the kernel. The filters size is <tt>2*radius+1</tt>.

            Precondition:
            \code
            radius   >= 0
            \endcode

            Postconditions:
            \code
            1. left()  == -radius
            2. right() ==  radius
            3. borderTreatment() == BORDER_TREATMENT_CLIP
            4. norm() == norm
            \endcode
        */
    void initAveraging(int radius, value_type norm);

        /** Init as an Averaging filter with norm 1.
         */
    void initAveraging(int radius)
    {
        initAveraging(radius, one());
    }

        /** \brief Init as the 2-tap forward difference filter.

            The filter coefficients are

            \code
            [1.0, -1.0]
            \endcode

            (note that filters are reflected by the convolution algorithm,
            and we get a forward difference after reflection).

            Postconditions:
            \code
            1. left()  == -1
            2. right() ==  0
            3. borderTreatment() == BORDER_TREATMENT_REFLECT
            4. norm() == 1.0
            \endcode
          */
    void initForwardDifference()
    {
        this->initExplicitly(-1, 0) = { 1.0, -1.0 };
        this->setBorderTreatment(BORDER_TREATMENT_REFLECT);
        this->symmetry_ = KernelAsymmetric;
    }

        /** \brief Init as the 2-tap backward difference filter.

            The filter coefficients are

            \code
            [1.0, -1.0]
            \endcode

            (note that filters are reflected by the convolution algorithm,
             and we get a forward difference after reflection).

            Postconditions:
            \code
            1. left()  == 0
            2. right() ==  1
            3. borderTreatment() == BORDER_TREATMENT_REFLECT
            4. norm() == 1.0
            \endcode
          */
    void initBackwardDifference()
    {
        this->initExplicitly(0, 1) = { 1.0, -1.0 };
        this->setBorderTreatment(BORDER_TREATMENT_REFLECT);
        this->symmetry_ = KernelAsymmetric;
    }

    void initSymmetricDifference(value_type norm )
    {
        this->initExplicitly(-1, 1) = { 0.5*norm, 0.0, -0.5*norm };
        this->setBorderTreatment(BORDER_TREATMENT_REFLECT);
        this->symmetry_ = KernelOdd;
    }

        /** \brief Init as the 3-tap symmetric difference filter.

            The filter coefficients are

            \code
            [0.5, 0, -0.5]
            \endcode

            Postconditions:
            \code
            1. left()  == -1
            2. right() ==  1
            3. borderTreatment() == BORDER_TREATMENT_REFLECT
            4. norm() == 1.0
            \endcode
          */
    void initSymmetricDifference()
    {
        initSymmetricDifference(one());
    }

        /** \brief Init the 3-tap second difference filter.

            The filter coefficients are

            \code
            [1, -2, 1]
            \endcode

            Postconditions:
            \code
            1. left()  == -1
            2. right() ==  1
            3. borderTreatment() == BORDER_TREATMENT_REFLECT
            4. norm() == 1
            \endcode
        */
    void initSecondDifference3()
    {
        this->initExplicitly(-1, 1) = { 1.0, -2.0, 1.0 };
        this->setBorderTreatment(BORDER_TREATMENT_REFLECT);
        this->symmetry_ = KernelEven;
    }

        /** \brief Init the optimal 5-tap first derivative filter for Scharr derivatives.

            This filter must be used in conjunction with the corresponding 5-tap smoothing filter
            (see initOptimalFirstDerivativeSmoothing5()), such that the derivative filter is applied
            along one dimension, and the smoothing filter along the other.
            The filter coefficients are

            \code
            [ 0.0831, 0.3338, 0.0, -0.3338, -0.0831 ]
            \endcode

            These values are optimal in the sense that the 5x5 filter obtained by combining
            this filter with the corresponding 5-tap smoothing filter gives the best
            approximation of rotational equivariance (steerability) among 5x5 first derivatives,
            see

            Scharr, Hanno. Optimale Operatoren in der Digitalen Bildverarbeitung. Dissertation: Ruprecht-Karls-Universität Heidelberg, 2000

            Postconditions:
            \code
            1. left()  == -2
            2. right() ==  2
            3. borderTreatment() == BORDER_TREATMENT_REFLECT
            4. norm() == 1.0
            \endcode
        */
    void initOptimalFirstDerivative5()
    {
        this->initExplicitly(-2, 2) = { 0.0831, 0.3338, 0.0, -0.3338, -0.0831 };
        this->setBorderTreatment(BORDER_TREATMENT_REFLECT);
        this->symmetry_ = KernelOdd;
    }

        /** \brief Init the optimal 5-tap second derivative filter for Scharr Laplacians.

            This filter must be used in conjunction with the corresponding 5-tap smoothing filter
            (see initOptimalSecondDerivativeSmoothing5()), such that the derivative filter is applied
            along one dimension, and the smoothing filter along the other.
            The filter coefficients are

            \code
            [ -0.083325, 1.3333, -2.49995, 1.3333, -0.083325 ]
            \endcode

            These values are optimal in the sense that the 5x5 filter obtained by combining
            this filter with the corresponding 5-tap smoothing filter has the best
            rotational invariance among all 5x5 Laplacian filters, see

            Scharr, Hanno. Optimale Operatoren in der Digitalen Bildverarbeitung. Dissertation: Ruprecht-Karls-Universität Heidelberg, 2000

            Postconditions:
            \code
            1. left()  == -2
            2. right() ==  2
            3. borderTreatment() == BORDER_TREATMENT_REFLECT
            4. norm() == 1.0
            \endcode
        */
    void initOptimalSecondDerivative5()
    {
        this->initExplicitly(-2, 2) = { -0.083325, 1.3333, -2.49995, 1.3333, -0.083325 };
        this->setBorderTreatment(BORDER_TREATMENT_REFLECT);
        this->symmetry_ = KernelEven;
    }

        /** \brief Init the kernel by a comma-separated initializer list.

            DEPRECATED, use operator=(std::initializer_list) instead.

            The left and right boundaries of the kernel must be passed.
            A comma-separated initializer list is given after the assignment
            operator. This function is used like this:

            \code
            // define horizontal Roberts filter
            vigra::Kernel1D<float> roberts_gradient_x;

            roberts_gradient_x.initExplicitly(0, 1) = 1.0, -1.0;
            \endcode

            The norm is set to the sum of the initializer values. If the wrong number of
            values is given, a run-time error results. It is, however, possible to give
            just one initializer. This creates an averaging filter with the given constant:

            \code
            vigra::Kernel1D<float> average5x1;

            average5x1.initExplicitly(-2, 2) = 1.0/5.0;
            \endcode

            Here, the norm is set to value*size().

            <b> Preconditions:</b>

            \code

            1. left <= 0
            2. right >= 0
            3. the number of values in the initializer list
               is 1 or equals the size of the kernel.
            \endcode
        */
    Kernel1D & initExplicitly(int left, int right)
    {
        vigra_precondition(left <= 0,
                     "Kernel1D::initExplicitly(): left border must be <= 0.");
        vigra_precondition(right >= 0,
                     "Kernel1D::initExplicitly(): right border must be >= 0.");

        right_ = right;
        left_ = left;
        symmetry_ = KernelAsymmetric;
        InternalVector(right - left + 1, value_type()).swap(kernel_);
        kernel_[-left] = one();

        return *this;
    }

        /** \brief Get iterator to center of kernel.

            Postconditions:
            \code

            center()[left()] ... center()[right()] are valid kernel positions
            \endcode
        */
    iterator center()
    {
        return &kernel_[-left()];
    }

    const_iterator center() const
    {
        return &kernel_[-left()];
    }

        /** \brief Access kernel value at specified location.

            Preconditions:
            \code

            left() <= location <= right()
            \endcode
        */
    reference operator[](int location)
    {
        return kernel_[location - left()];
    }

    const_reference operator[](int location) const
    {
        return kernel_[location - left()];
    }

        /** \brief Get left border of kernel (inclusive), always <= 0.
        */
    int left() const { return left_; }

        /** \brief Get right border of kernel (inclusive), always >= 0.
        */
    int right() const { return right_; }

        /** \brief  Get size of kernel (right() - left() + 1).
        */
    int size() const { return right_ - left_ + 1; }

        /** \brief Get default border treatment mode.

            Can be overridden in convolution functions.
        */
    BorderTreatmentMode borderTreatment() const
    { return border_treatment_; }

        /** \brief Set default border treatment mode.

            Can be overridden in convolution functions.
        */
    void setBorderTreatment( BorderTreatmentMode new_mode)
    { border_treatment_ = new_mode; }

        /** \brief Get norm of kernel.

            Typically norm() == 1.0
        */
    value_type norm() const { return norm_; }

        /** \brief Set a new norm and normalize kernel, use the normalization formula
            for the given <tt>derivativeOrder</tt>.
        */
    void
    normalize(value_type norm, unsigned int derivativeOrder = 0, double offset = 0.0);

        /** \brief Normalize kernel so that the coefficients sum to 1.
        */
    void normalize()
    {
        normalize(one());
    }

        /** \brief Get the kernel's symmetry (even, odd or assymetric).
        */
    KernelSymmetry symmetry() const
    {
        return symmetry_;
    }

        /** \brief Determine the kernel's symmetry.
        */
    void setSymmetry()
    {
        symmetry_ = checkSymmetry();
    }

        /** \brief Force kernel to be treated as asymmetric.
        */
    void setAsymmetric()
    {
        symmetry_ = KernelAsymmetric;
    }

  private:

    KernelSymmetry checkSymmetry() const
    {
        if(right_ == -left_)
        {
            bool is_even = true,
                 is_odd  = (kernel_[-left_] == value_type());
            for(int k=1; k<=right_; ++k)
            {
                if(kernel_[-left_ + k] != kernel_[-left_ - k])
                    is_even = false;
                if(kernel_[-left_ + k] != -kernel_[-left_ - k])
                    is_odd = false;
            }
            if(is_even)
                return KernelEven;
            if(is_odd)
                return KernelOdd;
        }
        return KernelAsymmetric;
    }

    InternalVector kernel_;
    int left_, right_;
    BorderTreatmentMode border_treatment_;
    value_type norm_;
    KernelSymmetry symmetry_;
};

template <class ARITHTYPE>
void Kernel1D<ARITHTYPE>::normalize(value_type norm,
                          unsigned int derivativeOrder,
                          double offset)
{
    typedef typename NumericTraits<value_type>::RealPromote TmpType;

    // find kernel sum
    auto k = kernel_.begin();
    TmpType sum = NumericTraits<TmpType>::zero();

    if(derivativeOrder == 0)
    {
        for(; k < kernel_.end(); ++k)
        {
            sum += *k;
        }
    }
    else
    {
        unsigned int faculty = 1;
        for(unsigned int i = 2; i <= derivativeOrder; ++i)
            faculty *= i;
        for(double x = left() + offset; k < kernel_.end(); ++x, ++k)
        {
            sum = TmpType(sum + *k * std::pow(-x, int(derivativeOrder)) / faculty);
        }
    }

    vigra_precondition(sum != NumericTraits<value_type>::zero(),
                    "Kernel1D<ARITHTYPE>::normalize(): "
                    "Cannot normalize a kernel with sum = 0");
    // normalize
    sum = norm / sum;
    k = kernel_.begin();
    for(; k != kernel_.end(); ++k)
    {
        *k = *k * sum;
    }

    norm_ = norm;
}

/***********************************************************************/

template <class ARITHTYPE>
void
Kernel1D<ARITHTYPE>::initGaussian(double std_dev,
                                  value_type norm,
                                  double window_ratio)
{
    vigra_precondition(std_dev >= 0.0,
              "Kernel1D::initGaussian(): Standard deviation must be >= 0.");
    vigra_precondition(window_ratio >= 0.0,
              "Kernel1D::initGaussian(): window_ratio must be >= 0.");

    if(std_dev > 0.0)
    {
        Gaussian<ARITHTYPE> gauss((ARITHTYPE)std_dev);

        // first calculate required kernel sizes
        int radius;
        if (window_ratio == 0.0)
            radius = (int)(3.0 * std_dev + 0.5);
        else
            radius = (int)(window_ratio * std_dev + 0.5);
        if(radius == 0)
            radius = 1;

        // allocate the kernel
        kernel_.erase(kernel_.begin(), kernel_.end());
        kernel_.reserve(radius*2+1);

        for(ARITHTYPE x = -(ARITHTYPE)radius; x <= (ARITHTYPE)radius; ++x)
        {
            kernel_.push_back(gauss(x));
        }
        left_ = -radius;
        right_ = radius;
    }
    else
    {
        kernel_.erase(kernel_.begin(), kernel_.end());
        kernel_.push_back(1.0);
        left_ = 0;
        right_ = 0;
    }

    if(norm != 0.0)
        normalize(norm);
    else
        norm_ = 1.0;

    // best border treatment for Gaussians is BORDER_TREATMENT_REFLECT
    border_treatment_ = BORDER_TREATMENT_REFLECT;
    symmetry_ = KernelEven;
}

/***********************************************************************/

template <class ARITHTYPE>
void
Kernel1D<ARITHTYPE>::initDiscreteGaussian(double std_dev,
                                          value_type norm)
{
    vigra_precondition(std_dev >= 0.0,
              "Kernel1D::initDiscreteGaussian(): Standard deviation must be >= 0.");

    if(std_dev > 0.0)
    {
        // first calculate required kernel sizes
        int radius = (int)(3.0*std_dev + 0.5);
        if(radius == 0)
            radius = 1;

        double f = 2.0 / std_dev / std_dev;

        // allocate the working array
        int maxIndex = (int)(2.0 * (radius + 5.0 * std::sqrt((double)radius)) + 0.5);
        InternalVector warray(maxIndex+1);
        warray[maxIndex] = 0.0;
        warray[maxIndex-1] = 1.0;

        for(int i = maxIndex-2; i >= radius; --i)
        {
            warray[i] = warray[i+2] + f * (i+1) * warray[i+1];
            if(warray[i] > 1.0e40)
            {
                warray[i+1] /= warray[i];
                warray[i] = 1.0;
            }
        }

        // the following rescaling ensures that the numbers stay in a sensible range
        // during the rest of the iteration, so no other rescaling is needed
        double er = std::exp(-radius*radius / (2.0*std_dev*std_dev));
        warray[radius+1] = er * warray[radius+1] / warray[radius];
        warray[radius] = er;

        for(int i = radius-1; i >= 0; --i)
        {
            warray[i] = warray[i+2] + f * (i+1) * warray[i+1];
            er += warray[i];
        }

        double scale = norm / (2*er - warray[0]);

        initExplicitly(-radius, radius);
        iterator c = center();

        for(int i=0; i<=radius; ++i)
        {
            c[i] = c[-i] = warray[i] * scale;
        }
    }
    else
    {
        kernel_.erase(kernel_.begin(), kernel_.end());
        kernel_.push_back(norm);
        left_ = 0;
        right_ = 0;
    }

    norm_ = norm;

    // best border treatment for Gaussians is BORDER_TREATMENT_REFLECT
    border_treatment_ = BORDER_TREATMENT_REFLECT;
    symmetry_ = KernelEven;
}

/***********************************************************************/

template <class ARITHTYPE>
void
Kernel1D<ARITHTYPE>::initGaussianDerivative(double std_dev,
                                            int order,
                                            value_type norm,
                                            double window_ratio)
{
    vigra_precondition(order >= 0,
              "Kernel1D::initGaussianDerivative(): Order must be >= 0.");

    if(order == 0)
    {
        initGaussian(std_dev, norm, window_ratio);
        return;
    }

    vigra_precondition(std_dev > 0.0,
              "Kernel1D::initGaussianDerivative(): "
              "Standard deviation must be > 0.");
    vigra_precondition(window_ratio >= 0.0,
              "Kernel1D::initGaussianDerivative(): window_ratio must be >= 0.");

    Gaussian<ARITHTYPE> gauss((ARITHTYPE)std_dev, order);

    // first calculate required kernel sizes
    int radius;
    if(window_ratio == 0.0)
        radius = (int)((3.0  + 0.5 * order) * std_dev + 0.5);
    else
        radius = (int)(window_ratio * std_dev + 0.5);
    if(radius == 0)
        radius = 1;

    // allocate the kernels
    kernel_.clear();
    kernel_.reserve(radius*2+1);

    // fill the kernel and calculate the DC component
    // introduced by truncation of the Gaussian
    ARITHTYPE dc = 0.0;
    for(ARITHTYPE x = -(ARITHTYPE)radius; x <= (ARITHTYPE)radius; ++x)
    {
        kernel_.push_back(gauss(x));
        dc += kernel_[kernel_.size()-1];
    }
    dc = ARITHTYPE(dc / (2.0*radius + 1.0));

    // remove DC, but only if kernel correction is permitted by a non-zero
    // value for norm
    if(norm != 0.0)
    {
        for(unsigned int i=0; i < kernel_.size(); ++i)
        {
            kernel_[i] -= dc;
        }
    }

    left_ = -radius;
    right_ = radius;

    if(norm != 0.0)
        normalize(norm, order);
    else
        norm_ = 1.0;

    // best border treatment for Gaussian derivatives is
    // BORDER_TREATMENT_REFLECT
    border_treatment_ = BORDER_TREATMENT_REFLECT;
    symmetry_ = (order % 2 == 0) ? KernelEven : KernelOdd;
}

/***********************************************************************/

template <class ARITHTYPE>
void
Kernel1D<ARITHTYPE>::initBinomial(int radius,
                                  value_type norm)
{
    vigra_precondition(radius > 0,
              "Kernel1D::initBinomial(): Radius must be > 0.");

    // allocate the kernel
    InternalVector(radius*2+1).swap(kernel_);
    typename InternalVector::iterator x = kernel_.begin() + radius;

    // fill kernel
    x[radius] = norm;
    for(int j=radius-1; j>=-radius; --j)
    {
        x[j] = 0.5 * x[j+1];
        for(int i=j+1; i<radius; ++i)
        {
            x[i] = 0.5 * (x[i] + x[i+1]);
        }
        x[radius] *= 0.5;
    }

    left_ = -radius;
    right_ = radius;
    norm_ = norm;

    // best border treatment for Binomial is BORDER_TREATMENT_REFLECT
    border_treatment_ = BORDER_TREATMENT_REFLECT;
    symmetry_ = KernelEven;
}

/***********************************************************************/

template <class ARITHTYPE>
void
Kernel1D<ARITHTYPE>::initAveraging(int radius,
                                   value_type norm)
{
    vigra_precondition(radius > 0,
              "Kernel1D::initAveraging(): Radius must be > 0.");

    // calculate scaling
    double scale = 1.0 / (radius * 2 + 1);

    // normalize
    kernel_.erase(kernel_.begin(), kernel_.end());
    kernel_.reserve(radius*2+1);

    for(int i=0; i<=radius*2+1; ++i)
    {
        kernel_.push_back(scale * norm);
    }

    left_ = -radius;
    right_ = radius;
    norm_ = norm;

    // best border treatment for Averaging is BORDER_TREATMENT_CLIP
    border_treatment_ = BORDER_TREATMENT_CLIP;
    symmetry_ = KernelEven;
}

//@}

} // namespace vigra

#endif /* VIGRA2_CONVOLUTION_KERNELS_HXX */
