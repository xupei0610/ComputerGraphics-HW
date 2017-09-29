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
    *data.raw = *src.data.raw; // <---- BUG! Add by Pei Xu
}

Image::Image (char* fname){

	int numComponents; //(e.g., Y, YA, RGB, or RGBA)
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
	   case 'p': //bmp
	   default:
	     stbi_write_bmp(fname, width, height, 4, data.raw);
	}
}

// Add by Pei Xu
#define PI 3.141592653589793238462643383279502884L
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
//Pixel * log(Pixel * const &lhs, double const &base)
//{
//	double b = 1;//std::log1p(base);
//	lhs->r = ComponentClamp(std::log(1+static_cast<double>(lhs->r)/255)/b*255);
//	lhs->g = ComponentClamp(std::log(1+static_cast<double>(lhs->g)/255)/b*255);
//	lhs->b = ComponentClamp(std::log(1+static_cast<double>(lhs->b)/255)/b*255);
//	lhs->a = ComponentClamp(std::log(1+static_cast<double>(lhs->a)/255)/b*255);
//	return lhs;
//}
//Pixel * pow(Pixel * const &lhs, double const &gamma)
//{
//	lhs->r = ComponentClamp(std::pow(lhs->r, gamma));
//	lhs->g = ComponentClamp(std::pow(lhs->g, gamma));
//	lhs->b = ComponentClamp(std::pow(lhs->b, gamma));
//	lhs->a = ComponentClamp(std::pow(lhs->a, gamma));
//	return lhs;
//}
// End

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

	// adjust brightness in HSL

	// Linear transformation
	// same with ImageMagick but with range [-1, 1]
	// const double beta = factor >= 1 ? 100 :
	// 					(factor <= -1 ? -100 : factor * 100);
	// auto trans = [beta](Pixel &p)
	// {
	//   p += beta;
	// }

	// Use linear constrast transformation to replace brightness transformation
	// for better performance
	// same with ImageMagick, but with range [-1, 1] instead of [-100, 100]
	const double alpha = std::tan(PI/2*(factor <= -1 ? -1 : factor >= 1 ? 1 : factor));
	auto trans = [alpha](Pixel &p)
	{
		p *= alpha;
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
	// Log
	// auto trans = [factor](Pixel &p)
	// {
	// 	log(&p, factor);
	// };
	// Gamma
	// auto trans = [factor](Pixel &p)
	// {
	// 	pow(&p, factor);
	// };
	// Linear transformation
	// same with ImageMagick, but with range [-1, 1] instead of [-100, 100]
	// const double alpha = std::tan(PI/2*(factor <= -1 ? -1 : factor >= 1 ? 1 : factor));
	// auto trans = [alpha](Pixel &p)
	// {
	// 	p *= alpha;
	// };

	// Better
	if (factor ==0)
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
			lookup[i] = ComponentClamp(i > avg_gray ? 255 : 0);
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
		auto contrast = 1.0/(1-factor);
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
	// auto trans = [factor](Pixel &p)
	// {
	// 	auto min = p.r < p.g ? (p.r < p.b ? p.r : p.b) : (p.g > p.b ? p.b : p.g);
	// 	auto max = p.r > p.g ? (p.r > p.b ? p.r : p.b) : (p.g < p.b ? p.b : p.g);

	// 	if (max == 0 || max == min)
	// 		return;

	// 	auto v = static_cast<double>(max);
	// 	auto s = factor*(1 - static_cast<double>(min) / max);
	// 	if (s > 1)
	// 		s = 1;
	// 	else if (s < 0)
	// 		s = 0;

	// 	decltype(v) h;
	// 	if (max == p.r)
	// 		h = static_cast<double>(p.g-p.b)/(max - min);
	// 	else if (max == p.g)
	// 		h = 2 + static_cast<double>(p.b-p.r)/(max - min);
	// 	else // if (max == p.b)
	// 		h = 4 + static_cast<double>(p.r-p.g)/(max - min);
	// 	if (h < 0)
	// 		h += 6;

	// 	double i, f;
	// 	f = std::modf(h, &i);

	// 	switch (static_cast<int>(i))
	// 	{
	// 	case 0:
	// 		p.r = v;
	// 		p.g = ComponentClamp(v * (1 - s * (1- f)));
	// 		p.b = ComponentClamp(v * (1 - s));
	// 		break;
	// 	case 1:
	// 		p.r = ComponentClamp(v * (1 - s * f));
	// 		p.g = v;
	// 		p.b = ComponentClamp(v * (1 - s));
	// 		break;
	// 	case 2:
	// 		p.r = ComponentClamp(v * (1 - s));
	// 		p.g = v;
	// 		p.b = ComponentClamp(v * (1 - s * (1- f)));
	// 		break;
	// 	case 3:
	// 		p.r = ComponentClamp(v * (1 - s));
	// 		p.g = ComponentClamp(v * (1 - s * f));
	// 		p.b = v;
	// 		break;
	// 	case 4:
	// 		p.r = ComponentClamp(v * (1 - s * (1- f)));
	// 		p.g = ComponentClamp(v * (1 - s));
	// 		p.b = v;
	// 		break;
	// 	default: // case 5:
	// 		p.r = v;
	// 		p.g = ComponentClamp(v * (1 - s));
	// 		p.b = ComponentClamp(v * (1 - s * f));
	// 		break;
	// 	}
	// };

	// adjust saturation in HSL
	// auto trans = [factor](Pixel &p)
	// {
	// 	auto min = p.r < p.g ? (p.r < p.b ? p.r : p.b) : (p.g > p.b ? p.b : p.g);
	// 	auto max = p.r > p.g ? (p.r > p.b ? p.r : p.b) : (p.g < p.b ? p.b : p.g);

	// 	if (max == 0 || max == min)
	// 		return;

	// 	auto l = 0.5 * (min + max);
	// 	auto s = factor * (max - min) * (l > 255 ? 510-min-max : min+max);
	// 	double h;
	// 	if (max == p.r)
	// 		h = static_cast<double>(p.g-p.b)/(max - min);
	// 	else if (max == p.g)
	// 		h = 2 + static_cast<double>(p.b-p.r)/(max - min);
	// 	else // if (max == p.b)
	// 		h = 4 + static_cast<double>(p.r-p.g)/(max - min);
	// 	if (h < 0)
	// 		h += 6;
	// 	auto q = l < 0.5 ? l*(1+s) : l+s-l*s;
	// 	auto px = 2*l - q;
	// 	auto tg = h/6;
	// 	auto tr = tg + 1.0/3;
	// 	if (tr > 1) tr = tr - 1;
	// 	auto tb = tg - 1.0/3;
	// 	if (tb < 0) tb = tb + 1;

	// 	if (tr < 1.0/6)
	// 		p.r = ComponentClamp(px + ((q-px)*6*tr));
	// 	else if (tr >= 1.0/6 and tr < 0.5)
	// 		p.r = ComponentClamp(q);
	// 	else if (tr >= 0.5 and tr < 2.0/3)
	// 		p.r = ComponentClamp(px + ((q-px)*6*(2.0/3 - tr)));
	// 	else
	// 		p.r = ComponentClamp(px);
	// 	if (tg < 1.0/6)
	// 		p.g = ComponentClamp(px + ((q-px)*6*tg));
	// 	else if (tg >= 1.0/6 and tg < 0.5)
	// 		p.g = ComponentClamp(q);
	// 	else if (tg >= 0.5 and tg < 2.0/3)
	// 		p.g = ComponentClamp(px + ((q-px)*6*(2.0/3 - tg)));
	// 	else
	// 		p.g = ComponentClamp(px);
	// 	if (tb < 1.0/6)
	// 		p.b = ComponentClamp(px + ((q-px)*6*tb));
	// 	else if (tb >= 1.0/6 and tb < 0.5)
	// 		p.b = ComponentClamp(q);
	// 	else if (tb >= 0.5 and tb < 2.0/3)
	// 		p.b = ComponentClamp(px + ((q-px)*6*(2.0/3 - tb)));
	// 	else
	// 		p.b = ComponentClamp(px);
	// };

	// fast implementation
	auto trans = [factor](Pixel &p)
	{
		auto min = p.r < p.g ? (p.r < p.b ? p.r : p.b) : (p.g > p.b ? p.b : p.g);
		auto max = p.r > p.g ? (p.r > p.b ? p.r : p.b) : (p.g < p.b ? p.b : p.g);
		if (max == 0 || max == min)
			return;

		auto l = 0.5 * (min + max);
		auto s = (max - min) / (l > 255 ? 510-min-max : min+max);
		auto alpha = factor >= 0 ? 1/(factor + s >= 1 ? s : 1-factor) - 1 : factor;
        l *= alpha;
        alpha += 1;
		p.r = ComponentClamp(p.r*alpha - l);
		p.g = ComponentClamp(p.g*alpha - l);
		p.b = ComponentClamp(p.b*alpha - l);
	};
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
        x = Width() + x - w;
    if (y < 0)
        y = Height() + y - h;
    if (x < 0)
        throw std::invalid_argument("Invalid horizontal start position for crop");
    if (y < 0)
        throw std::invalid_argument("Invalid horizontal start position for crop");
    auto abs_w = std::abs(w);
    if (x + abs_w >= Width())
        throw std::invalid_argument("Invalid width parameter for crop");
    auto abs_h = std::abs(h);
    if (y + abs_h >= Height())
        throw std::invalid_argument("Invalid width parameter for crop");

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
                        p.a = 0;
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

#define BUILD_NBIT_COLOR_LOOKUP_TABLE(nbits)                        \
    if (nbits < 0)                                                  \
        throw std::invalid_argument("Invalid (negative) value "     \
                                    "for color space bit length");  \
    if (nbits > 7)                                                  \
        return;                                                     \
    auto __level = 2 << nbits;                                      \
    auto __color_interval = 255 / (__level-1);                      \
    auto interval = 256 / __level;                                  \
    Component lookup[__level];                                      \
    for (auto i = __level; --i > 0;)                                \
        lookup[i] = ComponentClamp(__color_interval*i);             \
    lookup[0] = 0;                                                  \
    lookup[__level] = 255;

    BUILD_NBIT_COLOR_LOOKUP_TABLE(nbits)
    std::for_each(data.pixels, data.pixels + NumPixels(),
                  [&](Pixel &p)
                  {
                      p.r = lookup[p.r / interval];
                      p.g = lookup[p.g / interval];
                      p.b = lookup[p.b / interval];
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
    std::uniform_int_distribution<Component> dist(0, 255);
    auto patch = [&](Component &p)
    {
        if (p > dist(sd))
            p = lookup[p/interval + 1];
        else
            p = lookup[p/interval];
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
        if (p>>4 > Bayer4[row][col])
            p = lookup[p/interval + 1];
        else
            p = lookup[p/interval];
    };
    for (auto x = 0; x < Width(); ++x)
    {
        auto row = x & 3;
        for (auto y = 0; y < Height(); ++y)
        {
            auto col = y & 3;
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
    for (auto y = 0; y < Height(); ++y)
    {
        for (auto x = 0; x < Width(); ++x)
        {
            auto e = GetPixel(x, y);
            auto & p = GetPixel(x, y);
            p.r = lookup[p.r/interval];
            p.g = lookup[p.g/interval];
            p.b = lookup[p.b/interval];

            e -= p;

            if (x < Width() - 1)
                GetPixel(x + 1, y) += e * ALPHA;
            if (y < Height() - 1)
            {
                if (x > 0)
                    GetPixel(x-1, y+1) += e * BETA;

                GetPixel(x,   y+1) += e * GAMMA;

                if (x < Width() - 1)
                    GetPixel(x+1, y+1) += e * DELTA;
            }
        }
    }
#ifndef NDEBUG
    TOC()
    std::cout << "[Info] Action: FloydSteinbergDither " << nbits << std::endl;
#endif
}

template<typename T>
T * im2col(const T * const &img,
               int const &height,
               int const &width,
               int const &kernel_h,
               int const &kernel_w,
               int const &pad_t,
               int const &pad_l)
{
    // convolution with legacy 'same'
    auto col_rows = kernel_h * kernel_w;
    auto im_size = height * width;
    auto col = new T[col_rows * im_size];

    for (auto r = 0, tar = 0; r < col_rows; ++r)
    {
        auto offset_h = pad_t - ((r / kernel_w) % kernel_h); // no dilation;
        auto offset_w = pad_l - (r % kernel_w); // no dilation

        auto y0 = std::max(0, offset_h); // stride = 1
        auto x0 = std::max(0, offset_w); // stride = 1
        auto y1 = std::min(height, height + offset_h); // stride = 1, legacy = 'same'
        auto x1 = std::min(width,  width + offset_w);  // stride = 1, legacy = 'same'

        for (auto h = 0; h < y0; ++h)
        {
            for (auto w = 0; w < width; ++w)
                col[tar++] = T();
        }
        for (auto h = y0; h < y1; ++h)
        {
            for (auto w = 0; w < x0; ++w)
                col[tar++] = T();
            auto src_h = h - offset_h;
            auto src_w = x0 - offset_w;
            auto src = src_h * width + src_w;
            for (auto w = x0; w < x1; ++w)
            {
                col[tar++] = img[src++];
            }
            for (auto w = x1; w < width; ++w)
                col[tar++] = T();
        }
        for (auto h = y1; h < height; ++h)
        {
            for (auto w = 0; w < width; ++w)
                col[tar++] = T();
        }
    }

    return col;
}
// legacy = 'same'
struct RGB
{
    double r;
    double g;
    double b;
};

template<typename T>
RGB * conv2(const T *const &im,
            int const &im_height,
            int const &im_width,
            const double * const &kernel,
            int const &kernel_h,
            int const &kernel_w)
{
    auto new_im = new RGB[im_height*im_width];
    auto im_size = im_height * im_width;
    auto n = kernel_h * kernel_w;
    auto pad_h = kernel_h / 2;
    auto pad_w = kernel_w / 2;
    // Normal implementation
//    for (auto w = 0; w < im_width; ++w)
//    {
//        for (auto h = 0; h < im_height; ++h)
//        {
//            double r = 0, g = 0, b = 0;
//            for (auto k = 1; k < n; ++k)
//            {
//                auto x = w + k/kernel_w-pad_w;
//                auto y = h + k%kernel_h-pad_h;
//                if (x >= 0 && x < im_width && y >= 0 && y < im_height)
//                {
//                    auto p = im+y*im_width+x;
//                    r +=  p->r * kernel[n-k];
//                    g +=  p->g * kernel[n-k];
//                    b +=  p->b * kernel[n-k];
//                }
//            }
//            new_im[h*im_width+w] = {r, g, b};
//        }
//    }

    // fast implementation
    auto col = im2col(im, im_height, im_width, kernel_h, kernel_w, pad_h, pad_w);
    for (auto w= 0; w < im_size; ++w)
    {
        double r = 0, g = 0, b = 0;
        for (auto h = 0; h < n; ++h)
        {
            r += col[h*im_size+w].r * kernel[h];
            g += col[h*im_size+w].g * kernel[h];
            b += col[h*im_size+w].b * kernel[h];
        }
        new_im[w] ={r, g, b};
    }
    delete [] col;
    return new_im;
}

double *conv2(const double *const &im,
              int const &im_height,
              int const &im_width,
              const double * const &kernel,
              int const &kernel_h,
              int const &kernel_w)
{
    auto new_im = new double[im_height*im_width];
    auto im_size = im_height * im_width;
    auto n = kernel_h * kernel_w;
    auto pad_h = kernel_h / 2;
    auto pad_w = kernel_w / 2;
    auto col = im2col(im, im_height, im_width, kernel_h, kernel_w, pad_h, pad_w);
    for (auto w= 0; w < im_size; ++w)
    {
        new_im[w] = 0;
        for (auto h = 0; h < n; ++h)
            new_im[w] += col[h*im_size+w] * kernel[h];
    }
    delete [] col;
    return new_im;
}

double * gaussian(std::size_t const &size,
                  double sigma = 0)
{
    auto k_size = size * size;
    auto kernel = new double[k_size];

    // @see http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#getgaussiankernel
    if (sigma == 0)
        sigma = 0.3 * ((size-1)*0.5 - 1) + 0.8;

    auto den = -2*sigma*sigma;
    auto pad = size/2;
    auto sum = 0.0;
    for (decltype(k_size) i = 0; i < k_size; ++i)
    {
        auto r = i/size;
        auto c = i%size;
        kernel[i] = std::exp(((r-pad)*(r-pad)+(c-pad)*(c-pad))/den);
        sum += kernel[i];
    }
    std::for_each(kernel, kernel+k_size,
                  [&sum](double &k)
                  {
                      k /= sum;
                  }
                 );
    return kernel;
}

double *laplacian(std::size_t const &size)
{
    auto kernel = new double[size*size];
    if (size == 1)
    {
        kernel[0] = 0;
        kernel[1] = 1;
        kernel[2] = 0;
        kernel[3] = 1;
        kernel[4] = -4;
        kernel[5] = 1;
        kernel[6] = 0;
        kernel[7] = 1;
        kernel[8] = 0;
    }
    else if (size == 3)
    {
        kernel[0] = 2;
        kernel[1] = 0;
        kernel[2] = 2;
        kernel[3] = 0;
        kernel[4] = -8;
        kernel[5] = 0;
        kernel[6] = 2;
        kernel[7] = 0;
        kernel[8] = 2;
    }
    else if (size == 5)
    {
        kernel[0] = 0;
        kernel[1] = 0;
        kernel[2] = 1;
        kernel[3] = 0;
        kernel[4] = 0;
        kernel[5] = 0;
        kernel[6] = 1;
        kernel[7] = 2;
        kernel[8] = 1;
        kernel[9] = 0;
        kernel[10] = 1;
        kernel[11] = 2;
        kernel[12] = -16;
        kernel[13] = 2;
        kernel[14] = 1;
        kernel[15] = 0;
        kernel[16] = 1;
        kernel[17] = 2;
        kernel[18] = 1;
        kernel[19] = 0;
        kernel[20] = 0;
        kernel[21] = 0;
        kernel[22] = 1;
        kernel[23] = 0;
        kernel[24] = 0;
    }
    else
    {
        auto k_size = size*size;
        auto sigma = 0.5;
        auto den = -2*sigma*sigma;
        auto pad = size/2;
        auto sum = 0.0;
        for (decltype(k_size) i = 0; i < k_size; ++i)
        {
            auto r = i/size;
            auto c = i%size;
            auto s = (r-pad)*(r-pad)+(c-pad)*(c-pad);
            kernel[i] = (s+den)*std::exp(s/den);
            sum += kernel[i];

        }
        std::for_each(kernel, kernel+k_size,
                      [&sum](double &k)
                      {
                          k /= sum;
                      }
        );
    }

    return kernel;
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

    auto kernel = gaussian(n);
    auto new_im = conv2(data.pixels, Height(), Width(), kernel, n, n);

    for (auto i = 0; i < NumPixels(); ++i)
    {
        data.pixels[i].r = ComponentClamp(new_im[i].r);
        data.pixels[i].g = ComponentClamp(new_im[i].g);
        data.pixels[i].b = ComponentClamp(new_im[i].b);
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

    auto kernel = laplacian(n); //gaussin(n)

    auto new_im = conv2(data.pixels, Height(), Width(), kernel, n, n);

    auto alpha = 0.1*n; // empirical
    for (auto i = 0; i < NumPixels(); ++i)
    {
        data.pixels[i].r = ComponentClamp((1+alpha)*data.pixels[i].r - alpha*new_im[i].r);
        data.pixels[i].g = ComponentClamp((1+alpha)*data.pixels[i].g - alpha*new_im[i].g);
        data.pixels[i].b = ComponentClamp((1+alpha)*data.pixels[i].b - alpha*new_im[i].b);
    }

    delete [] kernel;
    delete [] new_im;

#ifndef NDEBUG
    TOC()
    std::cout << "[Info] Action: sharpen " << n << std::endl;
#endif
}

static double PrewittX[] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
static double PrewittY[] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};

void Image::EdgeDetect()
{
    /* WORK HERE */
#ifndef NDEBUG
    TIC()
#endif
    // Canny

    auto kernel = gaussian(5);

    // Obtain gradient
    auto grad_mag = new double[NumPixels()];
    auto grad_dir = new double[NumPixels()];
    // in colorful space
//    auto blurred_im = conv2(data.pixels, Height(), Width(), kernel, 5, 5);
//    auto h_grad = conv2(blurred_im, Height(), Width(), PrewittX, 3, 3);
//    auto v_grad = conv2(blurred_im, Height(), Width(), PrewittY, 3, 3);
//    for (decltype(NumPixels()) i = 0; i < NumPixels(); ++i)
//    {
//        auto Jx  = h_grad[i].r*h_grad[i].r + h_grad[i].g*h_grad[i].g + h_grad[i].b*h_grad[i].b;
//        auto Jy  = v_grad[i].r*v_grad[i].r + v_grad[i].g*v_grad[i].g + v_grad[i].b*v_grad[i].b;
//        auto Jxy = h_grad[i].r*v_grad[i].r + h_grad[i].g*v_grad[i].g + h_grad[i].b*v_grad[i].b;
//
//        auto D = std::sqrt(std::abs(Jx*Jx - 2*Jx*Jy + Jy*Jy + 4*Jxy*Jxy));
//        grad_mag[i] = (Jx + Jy + D)/2;
//        grad_dir[i] = std::atan(-Jxy / (grad_mag[i] - Jy));
//
//        grad_mag[i] = std::sqrt(grad_mag[i]);
//    }

    // in gray or HSV space
    auto gray = new double[NumPixels()];
    for(decltype(NumPixels()) i = 0; i < NumPixels(); ++i)
    {
        // gray space
        gray[i] = 0.299*data.pixels[i].r + 0.587*data.pixels[i].g + 0.114*data.pixels[i].b;
        // HSV space
//        auto min = std::min(std::min(data.pixels[i].r, data.pixels[i].g), data.pixels[i].b);
//        auto max = std::max(std::max(data.pixels[i].r, data.pixels[i].g), data.pixels[i].b);
//        if (max == 0 || max == min)
//            gray[i] = 0;
//        else
//            gray[i] = 1 - static_cast<double>(min) / max;
    }
    auto blurred_im = conv2(gray, Height(), Width(), kernel, 5, 5);
    auto h_grad = conv2(blurred_im, Height(), Width(), PrewittX, 3, 3);
    auto v_grad = conv2(blurred_im, Height(), Width(), PrewittY, 3, 3);
    for (decltype(NumPixels()) i = 0; i < NumPixels(); ++i)
    {
        grad_mag[i] = std::sqrt(h_grad[i] * h_grad[i] + v_grad[i]*v_grad[i]);
    }
    delete [] gray;

    // Non-Maxima Suppression
    const double PI_by_4 = std::atan(1);
    const double PI_by_2 = std::asin(1);
    double theta1;
    for (decltype(Height()) h = 0, i = 0; h < Height(); ++h)
    {
        for (decltype(Width()) w = 0; w < Width(); ++w, ++i)
        {
            int p1 = -1, p2 = -1, p3 = -1, p4 = -1;
            if (grad_dir[i] >= -PI_by_2 && grad_dir[i] <= -PI_by_4)
            {
                if (h != Height()-1)
                {
                    if (w != Width() - 1)
                        p1 = i + Width() + 1;
                    p2 = i + Width();
                }
                if (h != 0)
                {
                    if (w != 0)
                        p3 = i-Width()-1;
                    p4 = i-Width();
                }
                theta1 = grad_dir[i] + PI_by_2;
            }
            else if (grad_dir[i] > -PI_by_4 && grad_dir[i] <= 0)
            {
                if (w != Width()-1)
                {
                    p1 = i+1;
                    if (h != Height()-1)
                        p2 = i+Width()+1;
                }
                if (w != 0)
                {
                    p3 = i-1;
                    if (h != 0)
                        p4 = i-Width()-1;
                }
                theta1 = grad_dir[i] + PI_by_4;
            }
            else if (grad_dir[i] > 0 && grad_dir[i] <= PI_by_4)
            {
                if (w != Width()-1)
                {
                    if (h != 0)
                        p1 = i-Width()+1;
                    p2 = i+1;
                }
                if (w != 0)
                {
                    if (h != Height()-1)
                        p3 = i+Width()-1;
                    p4 = i-1;
                }
                theta1 = grad_dir[i];
            }
            else //if (grad_dir[i] > PI_by_4 && grad_dir[i] <= PI_by_2)
            {
                if (h != 0)
                {
                    p1 = i-Width();
                    if (w != Width()-1)
                        p2 = i-Width()+1;
                }
                if (h != Height()-1)
                {
                    p3 = i+Width();
                    if (w != 0)
                        p4 = i+Width()-1;
                }
                theta1 = grad_dir[i] - PI_by_4;
            }
            theta1 = 1/std::tan(theta1);
            double max1 = 0, max2 = 0;
            if (p2 != -1)
                max1 = theta1 * grad_mag[p2];
            if (p1 != -1)
                max1 += (1-theta1)*grad_mag[p1];
            if (p4 != -1)
                max2 = theta1 * grad_mag[p4];
            if (p3 != -1)
                max2 += (1-theta1)*grad_mag[p3];
            if (grad_mag[i] < std::max(max1, max2))
                grad_mag[i] = 0;
        }
    }

    // Hysteresis
    double min = 1000000, max = 0;
    const double one_by_256 = 0.00390625;
    for (decltype(NumPixels()) i = 0; i < NumPixels(); ++i)
    {
        if (grad_mag[i] < min and grad_mag[i] > one_by_256)
            min = grad_mag[i];
        if (grad_mag[i] > max)
            max = grad_mag[i];
    }
    auto gap = max - min;
    auto upper_threshold = min + gap * 0.15;
    auto lower_threshold = min + gap * 0.03;
    max = 0;
    for (decltype(NumPixels()) i = 0; i < NumPixels(); ++i)
    {
        auto flag = false;
        if (grad_mag[i] >= lower_threshold)
            flag = true;
        else
        {
            auto r = i/Width();
            auto c = i%Width();
            for (auto i = r-1; i < r+1; ++i)
            {
                for (auto j = c-1; j < c+1; ++j)
                {
                    if (grad_mag[i*Width()+j] > upper_threshold)
                    {
                        flag = true;
                        break;
                    }
                }
                if (flag)
                    break;
            }
        }
        if (flag == false)
        {
            data.pixels[i].r = 0;
            data.pixels[i].g = 0;
            data.pixels[i].b = 0;
        }
        else
        {
            auto tmp = (std::max(data.pixels[i].r, data.pixels[i].g), data.pixels[i].b);
            if (tmp > max)
                max = tmp;
        }
    }
    // show colorful result
    if (max != 255)
    {
        for (decltype(NumPixels()) i = 0; i < NumPixels(); ++i)
        {
            data.pixels[i].r = ComponentClamp(data.pixels[i].r / max * 255);
            data.pixels[i].g = ComponentClamp(data.pixels[i].g / max * 255);
            data.pixels[i].b = ComponentClamp(data.pixels[i].b / max * 255);
            data.pixels[i].a = ComponentClamp(255);
        }
    }
    // show gray result
    //    for (decltype(NumPixels()) i = 0; i < NumPixels(); ++i)
    //    {
    //        if (data.pixels[i].r != 0 || data.pixels[i].g != 0 || data.pixels[i].b != 0)
    //        data.pixels[i].r = ComponentClamp(255);
    //        data.pixels[i].g = data.pixels[i].r;
    //        data.pixels[i].b = data.pixels[i].r;
    //        data.pixels[i].a = data.pixels[i].r;
    //    }

    delete [] kernel;
    delete [] blurred_im;
    delete [] h_grad;
    delete [] v_grad;
    delete [] grad_mag;
    delete [] grad_dir;

#ifndef NDEBUG
    TOC()
    std::cout << "[Info] Action: edgeDetect " << std::endl;
#endif
}

// Add by Pei Xu
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
    return "";
}
// end

static const double GaussianKernel[] = {
        1.0/256,  4.0/256,  6.0/256,  4.0/256, 1.0/256,
        4.0/256, 16.0/256, 24.0/256, 16.0/256, 4.0/256,
        6.0/256, 24.0/256, 36.0/256, 24.0/256, 6.0/256,
        4.0/256, 16.0/256, 24.0/256, 16.0/256, 4.0/256,
        1.0/256,  4.0/256,  6.0/256,  4.0/256, 1.0/256,
};

Image* Image::Scale(double sx, double sy)
{
	/* WORK HERE */
#ifndef NDEBUG
    TIC()
#endif
    auto new_im = new Image(static_cast<int>(sx * Width()),
                            static_cast<int>(sy * Height()));
    sx = static_cast<double>(new_im->Width()) / Width();
    sy = static_cast<double>(new_im->Height()) / Height();
    for (decltype(Height()) h = 0; h < new_im->Height(); ++h)
    {
        for (decltype(Width()) w = 0; w < new_im->Width(); ++w)
        {
            new_im->GetPixel(w, h) = Sample(w / sx, h / sy);
        }
    }

#ifndef NDEBUG
    TOC()
    std::cout << "[Info] Action: scale " << sx << ", " << sy << ", " << getSamplingMethodName(this) << std::endl;
#endif

    return new_im;
}

Image* Image::Rotate(double angle)
{
	/* WORK HERE */
#ifndef NDEBUG
    TIC()
#endif

    auto c = std::cos(angle);
    auto s = std::sin(angle);
    auto center_x = Width()  / 2.0 - 0.5;
    auto center_y = Height() / 2.0 - 0.5;

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
            if (x > -1 && x < Width() && y > -1 && y < Height())
            {
                res = GetPixel(x, y);
            }
        }
            break;

        case IMAGE_SAMPLING_GAUSSIAN: // need blurring outside
        {
            if (u < 0 || u > Width()-1 || v < 0 || v > Height()-1)
                return res;

            RGB tmp_res{0, 0, 0};
            auto sigma = 1.0;
            auto d = 3*sigma;
            if (d > 10)
                d = 10;
            if (Height() < d || Width() < d)
                return res;

            auto den = -2 * sigma *sigma;

            auto sum = 0.0;
            for (auto i = static_cast<int>(u - d), i_end = static_cast<int>(u + d); i < i_end; ++i)
            {
                for (auto j = static_cast<int>(v - d), j_end = static_cast<int>(v + d); j < j_end; ++j)
                {
                    auto x = i < 0 ? -i : i > Width()-1  ? 2*Width()-i-2 : i;
                    auto y = j < 0 ? -j : j > Height()-1 ? 2*Height()-j-2 : j;

                    auto f = std::exp(((i-u)*(i-u) + (j-v)*(j-v)) / den);
                    const auto &p = GetPixel(x, y);
                    tmp_res.r += f * p.r;
                    tmp_res.g += f * p.g;
                    tmp_res.b += f * p.b;
                    sum += f;
                }
            }

            res.r = ComponentClamp(tmp_res.r/sum);
            res.g = ComponentClamp(tmp_res.g/sum);
            res.b = ComponentClamp(tmp_res.b/sum);
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
