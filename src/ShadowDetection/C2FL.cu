/*
* Copyright (c) 2017, JUN CHEN <1986ytuak@gmail.com>
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*     * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*     notice, this list of conditions and the following disclaimer in the
*     documentation and/or other materials provided with the distribution.
*     * Neither the name of the <organization> nor the
*     names of its contributors may be used to endorse or promote products
*     derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY JUN CHEN <email> ''AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL JUN CHEN <email> BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
*/

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2\opencv.hpp>
#include <fstream>
#include <algorithm>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include "ShadowDetection\GPUConnectedComponents.h"

	__device__ int find(int * localLabel, int p)
	{
		if (localLabel[p] != -1)
		{
			while (p != localLabel[p])
			{
				p = localLabel[p];
			}
			return p;
		}
		else
			return -1;

	}

	__device__ int labelUnion(int* buf, int g1, int g2)
	{ 
		
		g1 = buf[g1];
		g2 = buf[g2];

		if (g1 < g2) {
			int old = atomicMin(&buf[g2], g1);
			//done = (old == g2);
			//g2 = old;
		}
		else if (g2 < g1) {
			int old = atomicMin(&buf[g1], g2);
			//done = (old == g1);
			//g1 = old;
		}
	}

	__device__ void findAndUnion(int* buf, int g1, int g2) {
		bool done;
		do {

			g1 = find(buf, g1);
			g2 = find(buf, g2);

			// it should hold that g1 == buf[g1] and g2 == buf[g2] now

			if (g1 < g2) {
				int old = atomicMin(&buf[g2], g1);
				done = (old == g2);
				g2 = old;
			}
			else if (g2 < g1) {
				int old = atomicMin(&buf[g1], g2);
				done = (old == g1);
				g1 = old;
			}
			else {
				done = true;
			}

		} while (!done);
	}


	__global__ void gpuBufferLineLocalArrangeFirst(unsigned char *devSrcData, int * devLabelMap, int2 imgDimension, size_t binaryPitch)
	{

		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;

		int tid = threadIdx.x + threadIdx.y * blockDim.x;

		__shared__ int localLabel[32 * 16];
		__shared__ int dataBuff[32 * 16];

		localLabel[tid] = tid;
		unsigned char *devSrcDataRow = devSrcData + y * binaryPitch;
		int centerPosition = x + y * imgDimension.x;
		dataBuff[tid] = devSrcDataRow[x];
		__syncthreads();

		bool limits = x < imgDimension.x && y < imgDimension.y;

		if (limits)
		{
			uchar focusP = dataBuff[tid];
			if (focusP == 255)
			{
				// arrange			
				if (threadIdx.x > 0 && focusP == dataBuff[tid - 1])	localLabel[tid] = localLabel[tid - 1];
				__syncthreads();

				// arrange			
				if (threadIdx.y > 0 && focusP == dataBuff[tid - blockDim.x])	localLabel[tid] = localLabel[tid - blockDim.x];
				__syncthreads();

				int buf = tid;
				while (buf != localLabel[buf])
				{
					buf = localLabel[buf];
					localLabel[tid] = buf;
				}

				// UF
				// search neighbour, left and up
				if (threadIdx.x > 0 && focusP == dataBuff[tid - 1])		findAndUnion(localLabel, tid, tid - 1); // left
				__syncthreads();


				// link
				int l = find(localLabel, tid);

				// set global label map
				int lx = l % blockDim.x;
				int ly = l / blockDim.x;

				int globalL = (blockIdx.x * blockDim.x + lx) + (blockIdx.y * blockDim.y + ly) * imgDimension.x;
				devLabelMap[centerPosition] = globalL;
			}
			else
				devLabelMap[centerPosition] = -1;
		}
	}


	__global__ void gpuBufferLineUfGlobalArrangeCombine(unsigned char * devSrcData, int * devLabelMap, int2 imgDimension, size_t binaryPitch)
	{
		int id = threadIdx.x + threadIdx.y * blockDim.x + (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x * blockDim.y;

		// x direction
		int xy = id % imgDimension.x;
		int yy = id /imgDimension.x * blockDim.y;

		// y direction
		int xInLine = imgDimension.x / blockDim.x;
		int xx = id % xInLine * blockDim.x;
		int yx = id / xInLine;


		bool in_limitsx = xx < imgDimension.x && yx < imgDimension.y;
		bool in_limitsy = xy < imgDimension.x && yy < imgDimension.y;

		int centerx = xx + yx * imgDimension.x;
		int centery = xy + yy * imgDimension.x;

		unsigned char *devSrcDataX = devSrcData + yx * binaryPitch;
		unsigned char *devSrcDataY = devSrcData + yy * binaryPitch;
		unsigned char *devSrcDataY1 = devSrcData + (yy - 1) * binaryPitch;
		

		// search neighbour, left and up
		if (in_limitsx && xx > 0 && devSrcDataX[xx] == devSrcDataX[xx - 1] && devSrcDataX[xx] == 255)
			findAndUnion(devLabelMap, centerx, centerx - 1); // left

		if (in_limitsy && yy > 0 && devSrcDataY[xy] == devSrcDataY1[xy] && devSrcDataY[xy] == 255)
			findAndUnion(devLabelMap, centery, centery - imgDimension.x); // up

	}


	//__global__ void gpuBufferLineUfGlobalArrange(unsigned char * devSrcData, int * devLabelMap, int2 imgDimension)
	//{
	//	int id = threadIdx.x + threadIdx.y * blockDim.x + (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x * blockDim.y;
	//	if (id >= imgDimension.x * imgDimension.y)	return;;

	//	int x = blockIdx.x * blockDim.x + threadIdx.x;
	//	int y = blockIdx.y * blockDim.y + threadIdx.y;

	//	bool in_limits = x < imgDimension.x && y < imgDimension.y;

	//	if (in_limits)
	//	{
	//		unsigned char center = devSrcData[x + y * imgDimension.x];

	//		// search neighbour, left and up
	//		if (in_limits && x > 0 && threadIdx.x == 0 && center == devSrcData[x - 1 + y * imgDimension.x])
	//			devLabelMap[x + y * imgDimension.x] = devLabelMap[x + y * imgDimension.x - 1];
	//			//findAndUnion(devLabelMap, x + y * imgDimension.x, x - 1 + y * imgDimension.x); // left
	//		if (in_limits && y > 0 && threadIdx.y == 0 && center == devSrcData[x + (y - 1) * imgDimension.x])
	//			devLabelMap[x + y * imgDimension.x] = devLabelMap[x + (y - 1) * imgDimension.x];
	//			//findAndUnion(devLabelMap, x + y * imgDimension.x, x + (y - 1) * imgDimension.x); // up
	//	}
	//}



	__global__ void gpuBufferLineFinal(int * labelMap, int2 imgDimension)
	{
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;

		bool limits = x < imgDimension.x && y < imgDimension.y;

		int gid = x + y * imgDimension.x;
		if (limits)		labelMap[gid] = find(labelMap, gid);

	}


	void GPUCheckError3(char *methodName) {

	cudaThreadSynchronize();
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess) printf("%s: %s\n", methodName, cudaGetErrorString(error));
	
	}		

	void Coarse2FineCCL(unsigned char *srcImg, int *labelMap, size_t binaryPitch, int rows, int cols)
	{
		
		int2 blockSizes; 	blockSizes.x = 32;	blockSizes.y = 16;
		dim3 blockDim(blockSizes.x, blockSizes.y, 1);
		dim3 gridDim((cols + blockSizes.x - 1) / blockSizes.x, (rows + blockSizes.y - 1) / blockSizes.y, 1);

		int2 imgSize; imgSize.x = cols;  imgSize.y = rows;
		
		// reconfiguration
		int totalPixel = max(imgSize.x / blockDim.x * imgSize.y, imgSize.x * imgSize.y / blockDim.y);
		int sq = (int)sqrtf((float)totalPixel);
		dim3 blockSizeBuffLineCombine = blockDim;
		dim3 gridSizeBuffLineCombine((sq + blockSizeBuffLineCombine.x - 1) / blockSizeBuffLineCombine.x, (sq + blockSizeBuffLineCombine.y - 1) / blockSizeBuffLineCombine.y, 1);

		gpuBufferLineLocalArrangeFirst << < gridDim, blockDim >> > (srcImg, labelMap, imgSize, binaryPitch);
		gpuBufferLineUfGlobalArrangeCombine << < gridSizeBuffLineCombine, blockSizeBuffLineCombine >> > (srcImg, labelMap, imgSize, binaryPitch);
		gpuBufferLineFinal << < gridDim, blockDim >> > (labelMap, imgSize);
		
	}
