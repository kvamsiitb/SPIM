/************************ \cond COPYRIGHT *****************************
 *                                                                    *
 * Copyright (C) 2020 HOLOEYE Photonics AG. All rights reserved.      *
 * Contact: https://holoeye.com/contact/                              *
 *                                                                    *
 * This file is part of HOLOEYE SLM Display SDK.                      *
 *                                                                    *
 * You may use this file under the terms and conditions of the        *
 * "HOLOEYE SLM Display SDK Standard License v1.0" license agreement. *
 *                                                                    *
 **************************** \endcond ********************************/


 // Calculates an axicon and shows it on the SLM.

 // Please see readme.txt file located in the same folder of this example on how to compile this code.

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include <algorithm>

#include <cpp/holoeye_slmdisplaysdk.hpp>
#include <pylon/PylonIncludes.h>
#ifdef PYLON_WIN_BUILD
//    include <pylon/PylonGUI.h>
#endif

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching.hpp"
#include "opencv2/core/core.hpp"
using namespace holoeye;


// Namespace for using pylon objects.
using namespace Pylon;

// Namespace for using cout.
using namespace std;
using namespace cv;
// Number of images to be grabbed.
static const uint32_t c_countOfImagesToGrab = 1;
//#include "show_slm_preview.h"

#include <thread>
#include <chrono>
#include <slm_target\utils.hpp>

vector<pair< unsigned int, unsigned int> > spin_tuple(pair<unsigned int, unsigned int> shape
	, unsigned int bin_size)
{
	vector<pair< unsigned int, unsigned int> > lattice_val;
	for (unsigned int i = 0; i < shape.first / bin_size; i++) 
	{
		for (unsigned int j = 0; j < shape.second / bin_size; j++) 
		{
			lattice_val.push_back(make_pair(i, j));
		}
	}
}

vector<double> create_beta_schedule_linear(uint32_t num_sweeps, double beta_start, double beta_end)
{
	vector<double> beta_schedule;
	double beta_max;
	if (beta_end == -1)
		beta_max = (1 / 1000) * beta_start;//  here temperature will be zero when beta_max is 1000.f
	else
		beta_max = beta_end;
	double diff = (beta_start - beta_max) / (num_sweeps - 1);// A.P 3.28 - 0.01 inverse value increa finnal decrease
	for (int i = 0; i < num_sweeps; i++)
	{
		double val = beta_start - (i)*diff;
		beta_schedule.push_back((1.f / val));
	}

	return beta_schedule;
}

/*
1 buffer : If fails revert the data by remembering the index
*/
void InitialSLMLattice(shared_ptr<field<float>>& phaseData, int dataWidth, int dataHeight, int size_outer_bins, int size_bins
	, std::vector<float> numbVec)
{
	float maxNum = *std::max_element(numbVec.begin(), numbVec.end());

	for (int i = 0; i < numbVec.size(); i++)
		numbVec[i] = acos(numbVec.at(i) / maxNum);

	
	// Initial it to zero
	for (int y = 0; y < dataHeight; ++y)
	{
		float* row = phaseData->row(y);
		for (int x = 0; x < dataWidth; ++x)
		{
			row[x] = (float)0.0f;
		}
	}

	// Checkboard pattern of 16 bins
	int outer_bins = pow(2, size_outer_bins);
	pair< int, int> area = { 1024, 1024 };
	int sideHeight = (dataHeight - area.first) / 2;
	int sideWidth = (dataWidth - area.second) / 2;
	// number of boxes 
	for (int y = 0; y < area.first / outer_bins; ++y)
	{
		for (int x = 0; x < area.second / outer_bins; ++x)
		{
			for (int k = 0; k < outer_bins; ++k)
			{
				float* row = phaseData->row(y * outer_bins + k + sideHeight);
				for (int l = 0; l < outer_bins; ++l)
					row[x * outer_bins + l + sideWidth] = HOLOEYE_PIF * ((x + y) % 2);
			}
		}
	}


	// Checkboard pattern of 8 bins
	int bins = pow(2, size_bins);
	pair<int, int> active_area = { 512, 512 };
	sideHeight = (dataHeight - active_area.first) / 2;
	sideWidth = (dataWidth - active_area.second) / 2;

	vector< vector< float > > check;
	check.resize(active_area.first);
	for (int ii = 0; ii < active_area.first; ii++)
		check[ii].resize(active_area.second);
	
	int checkBin = 2;
	for (int y = 0; y < active_area.first / checkBin; ++y)
	{
		for (int x = 0; x < active_area.second / checkBin; ++x)
		{
			for (int k = 0; k < checkBin; ++k)
			{
				auto row = check.at(y * checkBin + k);
				for (int l = 0; l < checkBin; ++l)
					row[x * checkBin + l] = pow(-1.f, (x + y));
			}
		}
	}

	for (int y = 0; y < active_area.first / bins; ++y)
	{
		for (int x = 0; x < active_area.second / bins; ++x)
		{
			for (int k = 0; k < bins; ++k)
			{
				float* row = phaseData->row(y * bins + k + sideHeight);
				for (int l = 0; l < bins; ++l)
					row[x * bins + l + sideWidth] = numbVec[y * (active_area.first / bins) + x] * check[y * bins + k][x * bins + l];// HOLOEYE_PIF* ((x + y) % 2);
			}
		}
	}
}


int main(int argc, char* argv[])
{
		HOLOEYE_UNUSED(argc);
		HOLOEYE_UNUSED(argv);

		// Check if the installed SDK supports the required API version
		if( !heds_requires_version(3, false) )
			return 1;

		// Detect SLMs and open a window on the selected SLM:
		heds_instance slm;
		heds_errorcode error = slm.open();
		if (error != HEDSERR_NoError)
		{
			std::cerr << "ERROR: " << heds_error_string_ascii(error) << std::endl;
			return error;
		}

		// Configure the axicon properties:
		const int innerRadius = heds_slm_height_px() / 3;
		const int centerX = 0;
		const int centerY = 0;

		// Calculate the phase values of an axicon in a pixel-wise matrix:

		// pre-calc. helper variables:
		const float phaseModulation = 2.0f * HOLOEYE_PIF;
		const int dataWidth = heds_slm_width_px();
		const int dataHeight = heds_slm_height_px();

		// Reserve memory for the phase data matrix.
		// Use data type single to optimize performance:
		shared_ptr<field<float>> phaseData = field<float>::create(dataWidth, dataHeight);
		
		// phaseData.refreshrate()
		std::cout << "dataWidth  = " << dataWidth << std::endl;
		std::cout << "dataHeight = " << dataHeight << std::endl;

		ParseData parseObj = ParseData();

		std::vector<float> numberVect;
		parseObj.readNumberCSV("numbers.csv", numberVect);
		InitialSLMLattice(phaseData, dataWidth, dataHeight, 4, 3, numberVect);




		float** temp;
		float** check;

		// Show phase data on SLM:
		error = heds_show_phasevalues(phaseData, HEDSSHF_PresentAutomatic, phaseModulation);
		
		if (error != HEDSERR_NoError)
		{
			std::cerr << "ERROR: " << heds_error_string_ascii(error) << std::endl;

			return error;
		}

		// You may insert further code here.

		// Wait until the SLM process was closed
		std::cout << "Waiting for SDK process to close. Please close the tray icon to continue ..." << std::endl << std::flush;
		error = heds_utils_wait_until_closed();

		if (error != HEDSERR_NoError)
		{
			std::cerr << "ERROR: " << heds_error_string_ascii(error) << std::endl;

			return error;
		}


			// The exit code of the sample application.
		int exitCode = 0;

		// Before using any pylon methods, the pylon runtime must be initialized.
		PylonInitialize();

		try
		{
			// Create an instant camera object with the camera device found first.
			CInstantCamera camera( CTlFactory::GetInstance().CreateFirstDevice() );

			// Print the model name of the camera.
			cout << "Using device " << camera.GetDeviceInfo().GetModelName() << endl;

			// The parameter MaxNumBuffer can be used to control the count of buffers
			// allocated for grabbing. The default value of this parameter is 10.
			camera.MaxNumBuffer = 5;

			// Start the grabbing of c_countOfImagesToGrab images.
			// The camera device is parameterized with a default configuration which
			// sets up free-running continuous acquisition.
			camera.StartGrabbing( c_countOfImagesToGrab );

			// This smart pointer will receive the grab result data.
			CGrabResultPtr ptrGrabResult;

			// Camera.StopGrabbing() is called automatically by the RetrieveResult() method
			// when c_countOfImagesToGrab images have been retrieved.
			while (camera.IsGrabbing())
			{
				// Wait for an image and then retrieve it. A timeout of 5000 ms is used.
				camera.RetrieveResult( 5000, ptrGrabResult, TimeoutHandling_ThrowException );

				// Image grabbed successfully?
				if (ptrGrabResult->GrabSucceeded())
				{
					// Access the image data.
					cout << "SizeX: " << ptrGrabResult->GetWidth() << endl;
					cout << "SizeY: " << ptrGrabResult->GetHeight() << endl;
					const uint8_t* pImageBuffer = (uint8_t*) ptrGrabResult->GetBuffer();
					cout << "Gray value of first pixel: " << (uint32_t) pImageBuffer[0] << endl << endl;

	#ifdef PYLON_WIN_BUILD
					// Display the grabbed image.
					//Pylon::DisplayImage( 1, ptrGrabResult );
	#endif
				//CPylonImage pylonImage;

				//formatConverter.Convert(pylonImage, ptrGrabResult);//me
				// Create an OpenCV image out of pylon image
				Mat openCvImage = Mat(ptrGrabResult->GetHeight(), ptrGrabResult->GetWidth(), CV_8UC1, (void*)ptrGrabResult->GetBuffer());

				imshow("my camera", openCvImage);
				waitKey(0);
				}
				else
				{
					cout << "Error: " << std::hex << ptrGrabResult->GetErrorCode() << std::dec << " " << ptrGrabResult->GetErrorDescription() << endl;
				}
			}
		}
		catch (const GenericException& e)
		{
			// Error handling.
			cerr << "An exception occurred." << endl
				<< e.GetDescription() << endl;
			exitCode = 1;
		}

			// Wait until the SLM process was closed
		std::cout << "Waiting for SDK process to close. Please close the tray icon to continue ..." << std::endl << std::flush;
		error = heds_utils_wait_until_closed();

		if (error != HEDSERR_NoError)
		{
			std::cerr << "ERROR: " << heds_error_string_ascii(error) << std::endl;

			return error;
		}




		// Comment the following two lines to disable waiting on exit.
		cerr << endl << "Press enter to exit." << endl;
		while (cin.get() != '\n');

		// Releases all pylon resources.
		PylonTerminate();

		return exitCode;
	
}

#define CPU_GPU_COMPARISON 0

#if CPU_GPU_COMPARISON
// https://holoeye.com/wp-content/uploads/Application_Note_SLM-V.59.pdf

// https://sodocumentation.net/cuda/topic/6566/parallel-reduction--e-g--how-to-sum-an-array-#:~:text=The%20simplest%20approach%20to%20parallel,)%20%7B%20int%20idx%20%3D%20threadIdx.

static const int wholeArraySize = 65536000;
static const int blockSize = 1024;
static const int gridSize = 12; //this number is hardware-dependent; usually #SM*2 is a good number.

__global__ void sumCommMultiBlock(const int* gArr, int arraySize, int* gOut) {
	int thIdx = threadIdx.x;
	int gthIdx = thIdx + blockIdx.x * blockSize;
	const int gridSize = blockSize * gridDim.x;
	int sum = 0;
	for (int i = gthIdx; i < arraySize; i += gridSize)
		sum += gArr[i];
	__shared__ int shArr[blockSize];
	shArr[thIdx] = sum;
	__syncthreads();
	for (int size = blockSize / 2; size > 0; size /= 2) { //uniform
		if (thIdx < size)
			shArr[thIdx] += shArr[thIdx + size];
		__syncthreads();
	}
	if (thIdx == 0)
		gOut[blockIdx.x] = shArr[0];
}

__host__ int sumArray(vector<int> arr) {
	int* dev_arr;
	cudaMalloc((void**)&dev_arr, wholeArraySize * sizeof(int));
	cudaMemcpy(dev_arr, arr.data(), wholeArraySize * sizeof(int), cudaMemcpyHostToDevice);

	int out;
	int* dev_out;
	cudaMalloc((void**)&dev_out, sizeof(int) * gridSize);
	auto t0 = std::chrono::high_resolution_clock::now();
	sumCommMultiBlock << <gridSize, blockSize >> > (dev_arr, wholeArraySize, dev_out);
	//dev_out now holds the partial result
	sumCommMultiBlock << < 1, blockSize >> > (dev_out, gridSize, dev_out);
	//dev_out[0] now holds the final result
	cudaDeviceSynchronize();
	auto t1 = std::chrono::high_resolution_clock::now();
	float duration = (float)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

	printf("duration for GPU %.6f \n", (duration * 1e-6));//28e-6 // 203e-6

	cudaMemcpy(&out, dev_out, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(dev_arr);
	cudaFree(dev_out);
	return out;
}

void main() {
	int sum = 0;
	vector<int> vect(wholeArraySize, 1);
	cout << vect.size() << endl;
	auto t0 = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < wholeArraySize; i++)
	{
		sum += vect[i];
	}
	//this_thread::sleep_for(chrono::milliseconds(20000));

	auto t1 = std::chrono::high_resolution_clock::now();
	float duration = (float)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

	printf("duration for CPU %.6f, sum %d \n", (duration * 1e-6), sum);//28e-6 // 203e-6

	int sum1 = sumArray(vect);

	printf("Sum %d\n", sum1);
}

#endif