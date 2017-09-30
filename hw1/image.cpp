#include "image.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

// Added by Pei Xu
#ifndef NDEBUG
#include <iostream>
#include <chrono>
#define TIC(id) \
    auto _tic_##id = std::chrono::system_clock::now();
#define TOC(id) \
    auto _toc_##id = std::chrono::system_clock::now(); \
    std::cout << "[Info] Process time: " \
              << std::chrono::duration_cast<std::chrono::milliseconds>(_toc_##id - _tic_##id).count() \
              << "ms" << std::endl;
#endif
#include <random>
#include <cstring>
#include <functional>
#include <algorithm>
#include "extra.hpp"

Pixel & operator*=(Pixel &lhs, double const &rhs)
{
    lhs.r = ComponentClamp(lhs.r * rhs);
    lhs.g = ComponentClamp(lhs.g * rhs);
    lhs.b = ComponentClamp(lhs.b * rhs);
    lhs.a = ComponentClamp(lhs.a * rhs);
    return lhs;
}
Pixel & operator+=(Pixel &lhs, double const &rhs)
{
    lhs.r = ComponentClamp(lhs.r + rhs);
    lhs.g = ComponentClamp(lhs.g + rhs);
    lhs.b = ComponentClamp(lhs.b + rhs);
    lhs.a = ComponentClamp(lhs.a + rhs);
    return lhs;
}
Pixel & operator+=(Pixel &lhs, Pixel const &rhs)
{
    lhs.r = ComponentClamp(lhs.r + rhs.r);
    lhs.g = ComponentClamp(lhs.g + rhs.g);
    lhs.b = ComponentClamp(lhs.b + rhs.b);
    lhs.a = ComponentClamp(lhs.a + rhs.a);
    return lhs;
}
Pixel & operator-=(Pixel &lhs, Pixel const &rhs)
{
    lhs.r = ComponentClamp(lhs.r - rhs.r);
    lhs.g = ComponentClamp(lhs.g - rhs.g);
    lhs.b = ComponentClamp(lhs.b - rhs.b);
    lhs.a = ComponentClamp(lhs.a - rhs.a);
    return lhs;
}
std::string getSamplingMethodName(const Image * const &im)
{
    switch (im->sampling_method)
    {
        case IMAGE_SAMPLING_POINT:
            return "Point Sampling";
        case IMAGE_SAMPLING_BILINEAR:
            return "Bilinear Sampling";
//        case IMAGE_SAMPLING_GAUSSIAN:
        default:
            return "Gaussian Sampling";
    }
}
// Nonphotorealism effect
void Image::Watercolor()
{
#ifndef NDEBUG
    TIC()
#endif
    auto im = new double[NumPixels()*3];
    pcv::filter::watercolor(data.raw, Height(), Width(), 4, im);
    for (decltype(NumPixels()) i = 0; i < NumPixels(); ++i)
    {
        data.pixels[i].r = ComponentClamp(im[i*3]);
        data.pixels[i].g = ComponentClamp(im[i*3+1]);
        data.pixels[i].b = ComponentClamp(im[i*3+2]);
        data.pixels[i].a = ComponentClamp(255);
    }
#ifndef NDEBUG
    TOC()
    std::cout << "[Info] Action: watercolor " << std::endl;
#endif
}
// End

/**
 * Image
 **/
Image::Image (int width_, int height_){

    assert(width_ > 0);
    assert(height_ > 0);

    width           = width_;
    height          = height_;
    num_pixels      = width * height;
    sampling_method = IMAGE_SAMPLING_POINT;

    data.raw = new uint8_t[num_pixels*4];
    int b = 0; //which byte to write to
    for (int j = 0; j < height; j++){
        for (int i = 0; i < width; i++){
            data.raw[b++] = 0;
            data.raw[b++] = 0;
            data.raw[b++] = 0;
            data.raw[b++] = 0;
        }
    }

    assert(data.raw != NULL);
}

Image::Image (const Image& src){

    width           = src.width;
    height          = src.height;
    num_pixels      = width * height;
    sampling_method = IMAGE_SAMPLING_POINT;

    data.raw = new uint8_t[num_pixels*4];

    //memcpy(data.raw, src.data.raw, num_pixels);
    *data.raw = *src.data.raw; // <---- BUG!(Add by Pei Xu)
}

Image::Image (char* fname){

    int numComponents; //(e.g., Y, YA, RGB, or RGBA)

    // Add by Pei Xu
    if (fname[strlen(fname)-1] == 'x')
        data.raw = pcv::load::px(fname, width, height, numComponents, 4);
    else
    // End add by Pei Xu
    data.raw = stbi_load(fname, &width, &height, &numComponents, 4);

    if (data.raw == NULL){
        printf("Error loading image: %s", fname);
        exit(-1);
    }


    num_pixels = width * height;
    sampling_method = IMAGE_SAMPLING_POINT;

}

Image::~Image (){
    delete data.raw;
    data.raw = NULL;
}

void Image::Write(char* fname){

    int lastc = strlen(fname);

    switch (fname[lastc-1]){
        case 'g': //jpeg (or jpg) or png
            if (fname[lastc-2] == 'p' || fname[lastc-2] == 'e') //jpeg or jpg
                stbi_write_jpg(fname, width, height, 4, data.raw, 95);  //95% jpeg quality
            else //png
                stbi_write_png(fname, width, height, 4, data.raw, width*4);
            break;
        case 'a': //tga (targa)
            stbi_write_tga(fname, width, height, 4, data.raw);
            break;
        case 'x': //px // add by Pei Xu
            pcv::write::px(fname, width, height, 4, data.raw);
            break;
        case 'p': //bmp
        default:
            stbi_write_bmp(fname, width, height, 4, data.raw);
    }
}

void Image::AddNoise (double factor)
{
    /* WORK HERE */
#ifndef NDEBUG
    TIC()
#endif

    constexpr double mu = 0;
    constexpr double sigma = 1;

    std::random_device rd;
    std::mt19937 sd(rd());
    std::normal_distribution<double> dist(mu, sigma);

    std::for_each(data.pixels, data.pixels + NumPixels(),
                  [&](Pixel &p)
                  {
                      p += factor * dist(sd);
                  });

#ifndef NDEBUG
    TOC()
    std::cout << "[Info] Action: AddNoise " << factor << std::endl;
#endif
}


void Image::Brighten (double factor)
{
#ifndef NDEBUG
    TIC()
#endif
    // int x,y;
    // for (x = 0 ; x < Width() ; x++)
    // {
    // 	for (y = 0 ; y < Height() ; y++)
    // 	{
    // 		Pixel p = GetPixel(x, y);
    // 		Pixel scaled_p = p*factor;
    // 		GetPixel(x,y) = scaled_p;
    // 	}
    // }

    // adjust saturation in HSL
    factor *= 255;
    auto trans = [factor](Pixel &p)
    {
        float h, s, l;
        pcv::conversion::rgb2hsl(p.r, p.g, p.b, h, s, l);
        l += factor;
        if (l < 0)
            l = 0;
        else if (l > 255)
            l = 255;
        pcv::conversion::hsl2rgb(h, s, l, h, s, l);
        p.r = ComponentClamp(h);
        p.g = ComponentClamp(s);
        p.b = ComponentClamp(l);
    };

    std::for_each(data.pixels, data.pixels + NumPixels(), trans);
#ifndef NDEBUG
    TOC()
    std::cout << "[Info] Action: brightness " << factor << std::endl;
#endif
}


void Image::ChangeContrast (double factor)
{
    /* WORK HERE */
#ifndef NDEBUG
    TIC()
#endif
    if (factor == 0)
        return;

    double avg_gray = 0;
    for(decltype(NumPixels()) i = 0; i < NumPixels(); ++i)
    {
        avg_gray += 0.299*data.pixels[i].r;
        avg_gray += 0.587*data.pixels[i].g;
        avg_gray += 0.114*data.pixels[i].b;
    }
    avg_gray /= Height() * Width();

    Component lookup[256];
    if (factor >= 1)
    {
        for (auto i = 0; i < 256; ++i)
            lookup[i] = ComponentClamp(i == avg_gray ? avg_gray : i > avg_gray ? 255 : 0);
    }
    else if (factor <= -1)
    {
        for (auto i = 0; i < 256; ++i)
            lookup[i] = ComponentClamp(avg_gray);
    }
    else if (factor < 0)
    {
        auto contrast = 1+factor;
        for (auto i = 0; i < 256; ++i)
            lookup[i] = ComponentClamp(avg_gray + (i - avg_gray)*contrast);
    }
    else
    {
        auto contrast = 1.0/(1-std::abs(factor));
        for (auto i = 0; i < 256; ++i)
            lookup[i] = ComponentClamp(avg_gray + (i - avg_gray)*contrast);
    }
    auto trans = [lookup](Pixel &p)
    {
        p.r = lookup[p.r];
        p.g = lookup[p.g];
        p.b = lookup[p.b];
    };

    std::for_each(data.pixels, data.pixels + NumPixels(), trans);
#ifndef NDEBUG
    TOC()
    std::cout << "[Info] Action: contrast " << factor << std::endl;
#endif
}


void Image::ChangeSaturation(double factor)
{
    /* WORK HERE */
#ifndef NDEBUG
    TIC()
#endif
    // adjust saturation in HSV
//    factor *= 255;
//    auto trans = [factor](Pixel &p)
//    {
//        double h, s, v;
//        pcv::conversion::rgb2hsv(p.r, p.g, p.b, h, s, v);
//        s += factor;
//        if (s < 0)
//            s = 0;
//        else if (s > 255)
//            s = 255;
//        pcv::conversion::hsv2rgb(h, s, v, h, s, v);
//        p.r = ComponentClamp(h);
//        p.g = ComponentClamp(s);
//        p.b = ComponentClamp(v);
//    };

    // adjust saturation in HSL
    factor *= 255;
    auto trans = [factor](Pixel &p)
    {
        float h, s, l;
        pcv::conversion::rgb2hsl(p.r, p.g, p.b, h, s, l);
        s += factor;
        if (s < 0)
            s = 0;
        else if (s > 255)
            s = 255;
        pcv::conversion::hsl2rgb(h, s, l, h, s, l);
        p.r = ComponentClamp(h);
        p.g = ComponentClamp(s);
        p.b = ComponentClamp(l);
    };

    // fast implementation
//    if (factor >= 1) factor = 0.99;
//    auto trans = [factor](Pixel &p)
//    {
//        auto min = p.r < p.g ? (p.r < p.b ? p.r : p.b) : (p.g > p.b ? p.b : p.g);
//        auto max = p.r > p.g ? (p.r > p.b ? p.r : p.b) : (p.g < p.b ? p.b : p.g);
//        if (max == 0 || max == min)
//            return;
//
//        auto l = 0.5 * (min + max);
//        auto s = (max - min) / (l > 255 ? 510-min-max : min+max);
//        auto alpha = factor >= 0 ? 1/(factor + s >= 1 ? s : 1-factor) - 1 : factor;
//        p.r = ComponentClamp(p.r + (p.r-l)*alpha);
//        p.g = ComponentClamp(p.g + (p.g-l)*alpha);
//        p.b = ComponentClamp(p.b + (p.b-l)*alpha);
//    };

    std::for_each(data.pixels, data.pixels + NumPixels(), trans);
#ifndef NDEBUG
    TOC()
    std::cout << "[Info] Action: saturation " << factor << std::endl;
#endif
}


Image* Image::Crop(int x, int y, int w, int h)
{
    /* WORK HERE */
#ifndef NDEBUG
    TIC()
#endif
    if (x < 0)
        x = Width() + x;
    if (y < 0)
        y = Height() + y;
    if (w < 0)
        x += w;
    if (h < 0)
        y += h;
    if (x < 0 || x >= Width())
        throw std::invalid_argument("Invalid horizontal start position for crop");
    if (y < 0 || y >= Height())
        throw std::invalid_argument("Invalid vertical start position for crop");

    auto abs_w = std::abs(w);
    if (x + abs_w >= Width())
        throw std::invalid_argument("Invalid width parameter for crop");

    auto abs_h = std::abs(h);
    if (y + abs_h >= Height())
        throw std::invalid_argument("Invalid height parameter for crop");

    auto im = new Image(abs_w, abs_h);

    for (auto i = 0; i < abs_w; ++i)
    {
        auto tar_x = w > 0 ? i : abs_w - i - 1;
        for (auto j = 0; j < abs_h; ++j)
        {
            auto tar_y = h > 0 ? j : abs_h - j - 1;
            im->GetPixel(tar_x, tar_y) = GetPixel(x+i, y+j);
        }
    }

#ifndef NDEBUG
    TOC()
    std::cout << "[Info] Action: crop " << x << ", " << y << ", " << w << ", " << h << std::endl;
#endif
    return im;
}


void Image::ExtractChannel(int channel)
{
    /* WORK HERE */
#ifndef NDEBUG
    TIC()
#endif
    std::for_each(data.pixels, data.pixels + NumPixels(),
                  [channel](Pixel &p)
                  {
                      if ((channel & 1) != 1)
                          p.r = 0;
                      if ((channel & 2) != 2)
                          p.g = 0;
                      if ((channel & 4) != 4)
                          p.b = 0;
                      if ((channel & 8) != 8)
                          p.a = 255;
                  });
#ifndef NDEBUG
    TOC()
    std::cout << "[Info] Action: extractChannel " << channel << std::endl;
#endif
}

void Image::Quantize (int nbits)
{
    /* WORK HERE */
#ifndef NDEBUG
    TIC()
#endif

    if (nbits > 7)
        return;

#define BUILD_NBIT_COLOR_LOOKUP_TABLE(nbits)                        \
    if (nbits < 1)                                                  \
        throw std::invalid_argument("Invalid (negative) value "     \
                                    "for color space bit length");  \
    if (nbits > 8)                                                  \
        nbits = 8;                                                  \
    auto __level = 2 << (nbits-1);                                  \
    auto __color_interval = 255.0 / (__level-1);                    \
    auto interval = 256/__level;                                  \
    Component lookup[256];                                          \
    for (auto i = 256; --i > 0;)                                    \
        lookup[i] = ComponentClamp(i/interval*__color_interval);  \
    lookup[0] = 0;

    BUILD_NBIT_COLOR_LOOKUP_TABLE(nbits)
    std::for_each(data.pixels, data.pixels + NumPixels(),
                  [&](Pixel &p)
                  {
                      p.r = lookup[p.r];
                      p.g = lookup[p.g];
                      p.b = lookup[p.b];
                  });
#ifndef NDEBUG
    TOC()
    std::cout << "[Info] Action: quantize " << nbits << std::endl;
#endif
}

void Image::RandomDither (int nbits)
{
    /* WORK HERE */
#ifndef NDEBUG
    TIC()
#endif
    BUILD_NBIT_COLOR_LOOKUP_TABLE(nbits)
    std::random_device rd;
    std::mt19937 sd(rd());
    std::uniform_int_distribution<int> dist(0, 1);
    auto patch = [&](Component &p)
    {
        if (p > 255 - interval || dist(sd) == 0)
            p = lookup[p];
        else
            p = lookup[p + interval];
    };
    for (auto x = 0; x < Width(); ++x)
    {
        for (auto y = 0; y < Height(); ++y)
        {
            auto & p = GetPixel(x, y);

            patch(p.r);
            patch(p.g);
            patch(p.b);
        }
    }

#ifndef NDEBUG
    TOC()
    std::cout << "[Info] Action: randomDither " << nbits << std::endl;
#endif
}


static int Bayer4[4][4] =
        {
                {15,  7, 13,  5},
                { 3, 11,  1,  9},
                {12,  4, 14,  6},
                { 0,  8,  2, 10}
        };


void Image::OrderedDither(int nbits)
{
    /* WORK HERE */
#ifndef NDEBUG
    TIC()
#endif
    // M_1 = [[0, 2], [3 1]];
    // M_{n+1} = [[4M_{n}, 4M_{n}+2U_{n}], [4M_{n}+3U_{n}, 4M_{n}+U_{n}]];
    // where U_{n} is a n-by-n matrix whose entries are all 1.
    // M_{n}[y][x] = Bayer_{n}[x][y]

    BUILD_NBIT_COLOR_LOOKUP_TABLE(nbits)

    auto patch = [&](Component &p, int const &row, int const &col)
    {
        if (p > 255 - interval)
            p = lookup[p];
        else
            p = lookup[p>>4 < Bayer4[row][col] ? p : p + interval];
    };
    for (auto x = 0; x < Width(); ++x)
    {
        auto row = x % 4;
        for (auto y = 0; y < Height(); ++y)
        {
            auto col = y % 4;
            auto & p = GetPixel(x, y);
            patch(p.r, row, col);
            patch(p.g, row, col);
            patch(p.b, row, col);
        }
    }
#ifndef NDEBUG
    TOC()
    std::cout << "[Info] Action: orderedDither " << nbits << std::endl;
#endif
}

/* Error-diffusion parameters */
const double
        ALPHA = 7.0 / 16.0,
        BETA  = 3.0 / 16.0,
        GAMMA = 5.0 / 16.0,
        DELTA = 1.0 / 16.0;

void Image::FloydSteinbergDither(int nbits)
{
    /* WORK HERE */
#ifndef NDEBUG
    TIC()
#endif
    BUILD_NBIT_COLOR_LOOKUP_TABLE(nbits)

    auto patch = [](Pixel &p,
                    float const &dr,
                    float const &dg,
                    float const &db)
    {
        p.r = ComponentClamp(p.r+dr);
        p.g = ComponentClamp(p.g+dg);
        p.b = ComponentClamp(p.b+db);
    };
    for (auto y = 0; y < Height(); ++y)
    {
        for (auto x = 0; x < Width(); ++x)
        {
            auto & p = GetPixel(x, y);
            auto er = static_cast<float>(p.r);
            auto eg = static_cast<float>(p.g);
            auto eb = static_cast<float>(p.b);
            p.r = lookup[p.r];
            p.g = lookup[p.g];
            p.b = lookup[p.b];

            er -= p.r;
            eg -= p.g;
            eb -= p.b;

            if (x < Width() - 1)
                patch(GetPixel(x+1, y), ALPHA*er, ALPHA*eg, ALPHA*eb);
            if (y < Height() - 1)
            {
                if (x > 0)
                    patch(GetPixel(x-1, y+1), BETA*er, BETA*eg, BETA*eb);

                patch(GetPixel(x, y+1), GAMMA*er, GAMMA*eg, GAMMA*eb);

                if (x < Width() - 1)
                    patch(GetPixel(x+1, y+1), DELTA*er, DELTA*eg, DELTA*eb);
            }
        }
    }
#ifndef NDEBUG
    TOC()
    std::cout << "[Info] Action: FloydSteinbergDither " << nbits << std::endl;
#endif
}

void Image::Blur(int n)
{
    /* WORK HERE */
#ifndef NDEBUG
    TIC()
#endif
    if (n < 1)
        throw std::invalid_argument("Invalid (non-positive integer) value "
                                            "for kernel size of blurring operation.");

    // image size not change
    if (n % 2 == 0)
        ++n;

    auto kernel = pcv::kernel::gaussian(n);
    auto new_im = pcv::math::conv2(data.raw, Height(), Width(), 3, 4, kernel, n, n);

    for (auto i = 0; i < NumPixels(); ++i)
    {
        data.pixels[i].r = ComponentClamp(new_im[i*3]);
        data.pixels[i].g = ComponentClamp(new_im[i*3+1]);
        data.pixels[i].b = ComponentClamp(new_im[i*3+2]);
    }

    delete [] kernel;
    delete [] new_im;

#ifndef NDEBUG
    TOC()
    std::cout << "[Info] Action: Blur " << n << std::endl;
#endif
}

void Image::Sharpen(int n)
{
    /* WORK HERE */
#ifndef NDEBUG
    TIC()
#endif
    if (n < 1)
        throw std::invalid_argument("Invalid (non-positive integer) value "
                                            "for kernel size of blurring operation.");

    // image size not change
    if (n % 2 == 0)
        ++n;
    else if (n == 1)
        n = 3;

    auto kernel = pcv::kernel::median(n);
    auto new_im = pcv::math::conv2(data.raw, Height(), Width(), 3, 4, kernel, n, n);

    double alpha = 1.5;
    for (auto i = 0; i < NumPixels(); ++i)
    {
        data.pixels[i].r = ComponentClamp((1+alpha)*data.pixels[i].r - alpha*new_im[i*3]);
        data.pixels[i].g = ComponentClamp((1+alpha)*data.pixels[i].g - alpha*new_im[i*3+1]);
        data.pixels[i].b = ComponentClamp((1+alpha)*data.pixels[i].b - alpha*new_im[i*3+2]);
    }

    delete [] kernel;
    delete [] new_im;

#ifndef NDEBUG
    TOC()
    std::cout << "[Info] Action: sharpen " << n << std::endl;
#endif
}


void Image::EdgeDetect()
{
    /* WORK HERE */
#ifndef NDEBUG
    TIC()
#endif
    // to grayscale
    auto gray = new double[NumPixels()];
    for(decltype(NumPixels()) i = 0; i < NumPixels(); ++i)
        pcv::conversion::rgb2gray(data.pixels[i].r, data.pixels[i].g, data.pixels[i].b,
                                  gray[i]);
    // Canny
    auto edge = pcv::filter::canny(gray, Height(), Width());

    //show binary result
    for (decltype(NumPixels()) i = 0; i < NumPixels(); ++i)
    {
        if (edge[i] != 0)
            data.pixels[i].r = ComponentClamp(255);
        else
            data.pixels[i].r = ComponentClamp(0);
        data.pixels[i].g = data.pixels[i].r;
        data.pixels[i].b = data.pixels[i].r;
        data.pixels[i].a = ComponentClamp(255);
    }
    delete [] edge;

#ifndef NDEBUG
    TOC()
    std::cout << "[Info] Action: edgeDetect " << std::endl;
#endif
}

Image* Image::Scale(double sx, double sy)
{
    /* WORK HERE */
#ifndef NDEBUG
    TIC()
#endif
    auto new_im = new Image(static_cast<int>(sx * Width()),
                            static_cast<int>(sy * Height()));
    sx = static_cast<double>(new_im->Width()-1) / (Width()-1);
    sy = static_cast<double>(new_im->Height()-1) / (Height()-1);
    for (decltype(Height()) h = 0; h < new_im->Height(); ++h)
    {
        for (decltype(Width()) w = 0; w < new_im->Width(); ++w)
            new_im->GetPixel(w, h) = Sample(w/sx, h/sy);
    }

#ifndef NDEBUG
    TOC()
    std::cout << "[Info] Action: scale "
              << static_cast<double>(new_im->Width()) / Width()  << ", "
              << static_cast<double>(new_im->Height())/ Height() << ", "
              << getSamplingMethodName(this) << std::endl;
#endif

    return new_im;
}

Image* Image::Rotate(double angle)
{
    /* WORK HERE */
#ifndef NDEBUG
    TIC()
#endif
    angle /= 180.0;
    angle *= pcv::math::PI;

    auto c = std::cos(angle);
    auto s = std::sin(angle);
    auto center_x = (Width()-1) * 0.5;
    auto center_y = (Height()-1) * 0.5;

    auto l = 0 - center_x;
    auto r = Width() - 1 - center_x;
    auto t = 0 - center_y;
    auto b = Height() - 1 - center_y;

    auto x1 = c * l - s * t;
    auto x2 = c * l - s * b;
    auto x3 = c * r - s * t;
    auto x4 = c * r - s * b;

    auto y1 = s * l + c * t;
    auto y2 = s * l + c * b;
    auto y3 = s * r + c * t;
    auto y4 = s * r + c * b;

    auto shift_x = std::min(std::min(x1, x2), std::min(x3, x4));
    auto shift_y = std::min(std::min(y1, y2), std::min(y3, y4));

    auto width  = std::max(std::max(x1, x2), std::max(x3, x4)) - shift_x;
    auto height = std::max(std::max(y1, y2), std::max(y3, y4)) - shift_y;

    auto new_im = new Image(width, height);
    for (decltype(Height()) h = 0; h < new_im->Height(); ++h)
    {
        for (decltype(Width()) w = 0; w < new_im->Width(); ++w)
        {
            new_im->GetPixel(w, h) = Sample( c * (w+shift_x) + s * (h+shift_y) + center_x,
                                            -s * (w+shift_x) + c * (h+shift_y) + center_y);
        }
    }

#ifndef NDEBUG
    TOC()
    std::cout << "[Info] Action: rotate " << angle << ", " << getSamplingMethodName(this) << std::endl;
#endif

    return new_im;
}

void Image::Fun()
{
    /* WORK HERE */
#ifndef NDEBUG
    TIC()
#endif
    Pixel * new_im = new Pixel[NumPixels()];

    auto center_x = (Width()-1) * 0.5;
    auto center_y = (Height()-1) * 0.5;

    auto radius = std::min(center_x, center_y)*0.9;
    auto alpha = 0.2;
    auto beta  = pcv::math::PI*15/radius;
    auto sr = radius * radius;

    for (decltype(Height()) h = 0, i=0; h < Height(); ++h)
    {
        for (decltype(Width()) w = 0; w < Width(); ++w)
        {
            auto x = w - center_x;
            auto y = h - center_y;
            auto ang = std::atan2(y, x);
            auto c = std::cos(ang);
            auto s = std::sin(ang);
            auto r = std::sqrt(x*x + y*y);
            r += alpha*r*std::sin(beta*r);
            if (x*x + y*y < sr && r < radius)
                new_im[i++] = Sample(c*r + center_x, s*r + center_y);
            else
                new_im[i++] = GetPixel(w, h);
        }
    }

    delete data.pixels;
    data.pixels = new_im;

#ifndef NDEBUG
    TOC()
    std::cout << "[Info] Action: fun (water ripple effect)"<< std::endl;
#endif
}

/**
 * Image Sample
 **/
void Image::SetSamplingMethod(int method)
{
    assert((method >= 0) && (method < IMAGE_N_SAMPLING_METHODS));
    sampling_method = method;
}

Pixel Image::Sample (double u, double v){
    /* WORK HERE */
    Pixel res;
    switch (sampling_method)
    {
        case IMAGE_SAMPLING_POINT:
        {
            auto x = static_cast<int>(std::round(u));
            auto y = static_cast<int>(std::round(v));
            if (std::abs(x-u) == 0.5 && u > 0 && x < Width()-1)
            {
                auto &p00 = GetPixel(x, y);
                auto &p10 = GetPixel(x+1, y);
                if (std::abs(y-v) == 0.5 && v > 0 && y < Height()-1)
                {
                    auto &p01 = GetPixel(x, y+1);
                    auto &p11 = GetPixel(x+1, y+1);
                    res.r = ComponentClamp((static_cast<int>(p00.r)+p01.r+p11.r+p10.r)>>2);
                    res.g = ComponentClamp((static_cast<int>(p00.g)+p01.g+p11.g+p10.g)>>2);
                    res.b = ComponentClamp((static_cast<int>(p00.b)+p01.b+p11.b+p10.b)>>2);
                    res.a = ComponentClamp((static_cast<int>(p00.a)+p01.a+p11.a+p10.a)>>2);
                }
                else
                {
                    res.r = ComponentClamp((static_cast<int>(p00.r)+p10.r)>>1);
                    res.g = ComponentClamp((static_cast<int>(p00.g)+p10.g)>>1);
                    res.b = ComponentClamp((static_cast<int>(p00.b)+p10.b)>>1);
                    res.a = ComponentClamp((static_cast<int>(p00.a)+p10.a)>>1);
                }
            }
            else if (x > -1 && x < Width() && y > -1 && y < Height())
            {
                if (std::abs(y-v) == 0.5 && v > 0 && y < Height()-1)
                {
                    auto &p01 = GetPixel(x, y+1);
                    auto &p11 = GetPixel(x+1, y+1);
                    res.r = ComponentClamp((static_cast<int>(p01.r)+p11.r)>>1);
                    res.g = ComponentClamp((static_cast<int>(p01.g)+p11.g)>>1);
                    res.b = ComponentClamp((static_cast<int>(p01.b)+p11.b)>>1);
                    res.a = ComponentClamp((static_cast<int>(p01.a)+p11.a)>>1);
                }
                else
                    res = GetPixel(x, y);
            }
        }
            break;

        case IMAGE_SAMPLING_GAUSSIAN:
        {
            if (u < 0 || u > Width()-1 || v < 0 || v > Height()-1)
                return res;

            double tmp_r = 0, tmp_g = 0, tmp_b = 0;
            auto sigma = 1.0;
            auto r = 3*sigma;
            if (r > 10)
                r = 10;
            auto sr = r*r;
            if (Height() < r || Width() < r)
                return res;

            auto den = -2 * sigma *sigma;

            auto sum = 0.0;
            for (auto i = static_cast<int>(u - r), i_end = static_cast<int>(u + r); i < i_end; ++i)
            {
                for (auto j = static_cast<int>(v - r), j_end = static_cast<int>(v + r); j < j_end; ++j)
                {
                    auto d = (i-u)*(i-u) + (j-v)*(j-v);

                    if (d > sr)
                        continue;

                    auto x = i < 0 ? -i : i > Width()-1  ?  2*Width()-i-2 : i; // prevent black border
                    auto y = j < 0 ? -j : j > Height()-1 ? 2*Height()-j-2 : j; // prevent black border

                    auto f = std::exp(d / den);
                    const auto &p = GetPixel(x, y);
                    tmp_r += f * p.r;
                    tmp_g += f * p.g;
                    tmp_b += f * p.b;
                    sum += f;
                }
            }

            res.r = ComponentClamp(tmp_r/sum);
            res.g = ComponentClamp(tmp_g/sum);
            res.b = ComponentClamp(tmp_b/sum);
        }
            break;
        case IMAGE_SAMPLING_BILINEAR:
        default:
        {
            auto l = std::floor(u);
            auto t = std::floor(v);
            if (l < 0 || l > Width()-1 || t < 0 || t > Height()-1)
                return res;
            auto r = l == Width()  - 1 ? l : l + 1;
            auto b = t == Height() - 1 ? t : t + 1;
            auto w_up = b - v;
            auto w_dn = v - t;
            auto w_lf = r - u;
            auto w_rt = u - l;
            if (w_up == 0 && w_dn == 0) // prevent black border
                w_up = 1;
            if (w_lf == 0 && w_rt == 0) // prevent black border
                w_lf = 1;

            const auto &p11 = GetPixel(l, t);
            const auto &p12 = GetPixel(r, t);
            const auto &p21 = GetPixel(l, b);
            const auto &p22 = GetPixel(r, b);

            res.r = ComponentClamp((p11.r*w_lf + p12.r*w_rt)*w_up +
                                   (p21.r*w_lf + p22.r*w_rt)*w_dn);
            res.g = ComponentClamp((p11.g*w_lf + p12.g*w_rt)*w_up +
                                   (p21.g*w_lf + p22.g*w_rt)*w_dn);
            res.b = ComponentClamp((p11.b*w_lf + p12.b*w_rt)*w_up +
                                   (p21.b*w_lf + p22.b*w_rt)*w_dn);
            res.a = ComponentClamp((p11.a*w_lf + p12.a*w_rt)*w_up +
                                   (p21.a*w_lf + p22.a*w_rt)*w_dn);
        }
            break;
    }
    return res;
}
