#pragma once
#include <limits>
#include <algorithm>
#include <cstdio>
#include <string>



inline void minmax(float* min, float* max, float* values, int w, int h)
{
	float mmin = (std::numeric_limits<float>::max)();
	float mmax = (std::numeric_limits<float>::min)();
	for (int y = 0; y < h; y++)
		for (int x = 0; x < w; x++)
		{
			float v = values[y*w + x];
			mmin = std::min<float>(mmin, v);
			mmax = std::max<float>(mmax, v);
		}
	*min = mmin;
	*max = mmax;
}

inline void write(const std::string& file, int dimx, int dimy, int channels, float* data, bool normalize = false)
{
	int i, j;
	FILE *fp;
	fopen_s(&fp, file.c_str(), "wb"); /* b - binary mode */
	(void)fprintf(fp, "P6\n%d %d\n255\n", dimx, dimy);

	float min = 0;
	float max = 1;

	if(normalize)
	{
		minmax(&min, &max, data, dimx * channels, dimy);
	}
	
	for (j = 0; j < dimy; ++j)
	{
		for (i = 0; i < dimx; ++i)
		{
			static unsigned char color[3] = {255, 255, 255};
			for (int channel = 0; channel < 3; channel++)
			{
				float v = 1.0;
				if(channel < channels)
				{
					v = data[channels * (dimx * j + i) + channel];
				}
				color[channel] = (unsigned char)(std::clamp<float>((v - min) / (max - min), 0.0f, 1.0f) * 255.0f);
			}
			
			
			(void)fwrite(color, 1, 3, fp);
		}
	}
	(void)fclose(fp);
}

inline void printMinmax(float* values, int w, int h)
{
	float min, max;
	minmax(&min, &max, values, w, h);
	printf("min: %f, max: %f\n", min, max);
}

