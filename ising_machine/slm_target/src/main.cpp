/************************ \cond COPYRIGHT *****************************
 *                                                                    *
 * Developed by N Krishna Vamsi for Spatial Photonic Ising Machine    *
 *                                                                    *
 **************************** \endcond ********************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include <algorithm>

#include <cpp/holoeye_slmdisplaysdk.hpp>
#include <pylon/PylonIncludes.h>
#include <pylon/BaslerUniversalInstantCamera.h>
#include <slm_target/getopt.h>

#ifdef PYLON_WIN_BUILD
//    include <pylon/PylonGUI.h>
#endif

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching.hpp"
#include "opencv2/core/core.hpp"
using namespace holoeye;

using HoloeyeType = field<float>;
#define ASSERT(condition) { if(!(condition)){ std::cerr << "ASSERT FAILED: " << #condition << " @ " << __FILE__ << " (" << __LINE__ << ")" << std::endl; } }

// Namespace for using pylon objects.
using namespace Pylon;

// Namespace for using cout.
using namespace std;
using namespace cv;
// Number of images to be grabbed.
static const uint32_t c_countOfImagesToGrab = 1;
#include "cpp/show_slm_preview.h"

#include <thread>
#include <chrono>
#include <slm_target\utils.hpp>

#define NUM_SPINS_FLIP 4

vector<pair< unsigned int, unsigned int> > SpinTuple(pair<unsigned int, unsigned int> shape
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
	double diff = (beta_start - beta_max) / (num_sweeps - 1);// A.P 3.28 - 0.01 inverse value increa final decrease
	for (int i = 0; i < num_sweeps; i++)
	{
		double val = beta_start - (i)*diff;
		beta_schedule.push_back( val );
	}

	return beta_schedule;
}

void DisplayCheckerBoardPattern(shared_ptr<HoloeyeType>& phaseData, int dataWidth, int dataHeight, pair< int, int> area, int outer_bins)
{
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
}
/*
1 buffer : If fails revert the data by remembering the index
*/
void InitialSLMLattice(shared_ptr<HoloeyeType>& phaseData, int dataWidth, int dataHeight, 
	std::vector<float>& numbVec, std::vector<float>& isingSpins, pair<int, int> active_area, int bins)
{
	// Checkboard pattern of 8 bins
	//pair<int, int> active_area = { 512, 512 };
	int sideHeight = (dataHeight - active_area.first) / 2;
	int sideWidth = (dataWidth - active_area.second) / 2;

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
				auto row = check[y * checkBin + k];
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
				// (-1)^j cos-1 Em
				for (int l = 0; l < bins; ++l)
					row[x * bins + l + sideWidth] =
							(isingSpins[y * (active_area.first / bins) + x] + 1) * HOLOEYE_PIF / 2 + HOLOEYE_PIF / 2 +
												numbVec[y * (active_area.first / bins) + x] * check[y * bins + k][x * bins + l];
					
			}
		}
	}
}

void FLipLattice(shared_ptr<HoloeyeType> phaseData, int dataWidth, int dataHeight, std::vector<float>& isingSpins,
	pair<int, int> active_area, int bins, vector<pair< unsigned int, unsigned int> >  spinLatticePts, vector<unsigned int> selLatticeIndex)
{
	cout << "-2. FLipLattice " << endl;
	int sideHeight = (dataHeight - active_area.first) / 2;
	int sideWidth = (dataWidth - active_area.second) / 2;
	cout << "-1. FLipLattice " << endl;
	for (unsigned int sel_spin = 0; sel_spin < selLatticeIndex.size(); ++sel_spin)
	{
		int y = spinLatticePts.at(sel_spin).first;
		int x = spinLatticePts.at(sel_spin).second;
		cout << "0. FLipLattice" << endl;
		for (int k = 0; k < bins; ++k)
		{
			cout << "0. FLipLattice " << k << endl;
			float* row = phaseData->row(y * bins + k + sideHeight);
			// (-1)^j cos-1 Em
			for (int l = 0; l < bins; ++l)
			{
				if (isingSpins[y * (active_area.first / bins) + x] == 1.f)
				{
					row[x * bins + l + sideWidth] -= HOLOEYE_PIF;
				}
				else if (isingSpins[y * (active_area.first / bins) + x] == -1.f)
				{
					row[x * bins + l + sideWidth] += HOLOEYE_PIF;
				}
				else {
					std::cout << "Rama rama" << std::endl;
				}
				cout << "1. FLipLattice " << endl;
			}
			
		}
		isingSpins[y * (active_area.first / bins) + x] *= -1.f;
	}
}


void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
	}
}
class BaslerCamera {
public:
	~BaslerCamera()
	{
		 delete _camera;
	}
	BaslerCamera(int _timeoutMs = 5000, int width = 1920, int height= 1080, int offsetx = 64, int offsety = 4, float exposure_value = 1100.0)
	{
		// Create the device and attach it to CInstantCamera.
		// Let CInstantCamera take care of destroying the device.
		//_camera->Attach(CTlFactory::GetInstance().CreateFirstDevice(), Pylon::Cleanup_Delete);
		//cout << "Using device " << _camera->GetDeviceInfo().GetModelName() << endl;
		cout << "Device initialized" << endl;

		_camera = new CBaslerUniversalInstantCamera(CTlFactory::GetInstance().CreateFirstDevice(), Pylon::Cleanup_Delete);
		// Open camera.
		_camera->Open();
		_camera->DeviceLinkThroughputLimitMode.SetValue("Off");

		//Set the camera Region of Interest (ROI) to maximize allowed frame rate
		int64_t maxWidth = _camera->Width.GetMax();      // Maximum width of camera in pixels
		int64_t maxHeight = _camera->Height.GetMax();    // Maximum height of camera in pixels
		

		_camera->Width.SetValue(width);              // Set pixel width as lower than full allowed width
		_camera->Height.SetValue(height);            // Set pixel height as lower than full allowed height

		// https://docs.baslerweb.com/image-roi
		_camera->OffsetX.SetValue(offsetx);          // Setting origin of the sensor array from the top left corner
		_camera->OffsetY.SetValue(offsety);

		//camera.GetNodeMap();
		_camera->AcquisitionFrameRateEnable.SetValue(true);
		// https://docs.baslerweb.com/resulting-frame-rate#factors-limiting-the-frame-rate
		// Decrease the exposure time, Image ROI, Sensor Bit Depth
		_camera->AcquisitionFrameRate.SetValue(150.0);
		//cout << "frame rate set" << endl;
		double rate = _camera->ResultingFrameRate.GetValue();
		cout << "frame rate is:" << rate << endl;

		//// Set the pixel data format.
		
		//_camera->PixelFormat.SetValue(Basler_UniversalCameraParams::PixelFormat_Mono8);

		// Get the ExposureTime feature.
		// On GigE cameras, the feature is called 'ExposureTimeRaw'.
		// On USB cameras, it is called 'ExposureTime'.
		if (_camera->ExposureTime.IsValid())
		{
			// We need the integer representation because the GUI controls can only use integer values.
			// If it doesn't exist, return an empty parameter.
			_camera->ExposureTime.SetValue(exposure_value);
		}
		else // if (_camera->ExposureTimeRaw.IsValid())
		{
			cout << "How dark" << endl;
			// m_exposureTime.Attach(_camera->ExposureTimeRaw.GetNode());
		}
		
	}

	void checkImage()
	{
		cout << "here i am" << endl;
		// This smart pointer will receive the grab result data.
		
		cout << "0. here i am" << endl;
		// GrabOne calls StartGrabbing and StopGrabbing internally.
		// As seen above Open() is called by StartGrabbing and
		// the OnOpened() method of the CAcquireSingleFrameConfiguration handler is called.
		//_camera->GrabOne(_timeoutMs, _ptrGrabResult);

		//grab one image
		_camera->StartGrabbing(1, GrabStrategy_OneByOne, GrabLoop_ProvidedByUser);
		//grab is stopped automatically due to maxImages = 1
		_camera->RetrieveResult(_timeoutMs, _ptrGrabResult, TimeoutHandling_ThrowException) && _ptrGrabResult->GrabSucceeded();

		cout << "1. here i am" << endl;
		cout << "SizeX: " << _ptrGrabResult->GetWidth() << endl;
		cout << "SizeY: " << _ptrGrabResult->GetHeight() << endl;
		// Image grabbed successfully?
		cout << "2. here i am" << endl;
		if (_ptrGrabResult->GrabSucceeded())
		{
			Mat openCvImage = Mat(_ptrGrabResult->GetHeight(), _ptrGrabResult->GetWidth(), CV_8UC1, (void*)_ptrGrabResult->GetBuffer());
			//Create a window
			namedWindow("My Window", 1);
			//set the callback function for any mouse event
			setMouseCallback("My Window", CallBackFunc, NULL);
			//show the image
			imshow("My Window", openCvImage);
			waitKey(0);
		}
		cout << "Provide _xin, _yin, _xout, _yout in this order" << endl;
		std::cin >> _xin;
		std::cin >> _yin;
		std::cin >> _xout;
		std::cin >> _yout;
		std::cout << " "<< _xin << " " << _yin << " " << _xout << " " << _yout << endl;
		
		_heightFinImg = _xout - _xin;
		_widthFinImg = _yout - _yin;
		_areaFinImg = _heightFinImg * _widthFinImg;
	}

	void openBaslerCamera()
	{
		_camera->Open();
	}

	float collectSingleImageNEnergy()
	{
		// https://docs.baslerweb.com/pylonapi/cpp/pylon_advanced_topics#one-by-one-grab-strategy
		
		//_camera->GrabOne(5000, _ptrGrabResult);

		//grab one image
		_camera->StartGrabbing(1, GrabStrategy_OneByOne, GrabLoop_ProvidedByUser);

		//grab is stopped automatically due to maxImages = 1
		_camera->RetrieveResult(_timeoutMs, _ptrGrabResult, TimeoutHandling_ThrowException) && _ptrGrabResult->GrabSucceeded();

		float energy;
		
		//if (_ptrGrabResult->GrabSucceeded())
		//{
			const uint8_t* pImageBuffer = (uint8_t*)_ptrGrabResult->GetBuffer();

			int sum = 0;
			for (int y = _yin; y < _yout; ++y)
				for (int x = _xin; x < _xout; x++)
					sum += (int)pImageBuffer[y * _heightFinImg + x];

			energy = sum / _areaFinImg;
		//}
		//else
		//{
			//cout << "Error: " << std::hex << _ptrGrabResult->GetErrorCode() << std::dec << " " << _ptrGrabResult->GetErrorDescription() << endl;
		//}
		return energy;
	}
	void closeBaslerCamera()
	{
		_camera->Close();
	}
private:
	// C:\Program Files\Basler\pylon 6\Development\Samples\C++\GUI_MFCMultiCam\GuiCamera.cpp
	CBaslerUniversalInstantCamera* _camera;
	// https://docs.baslerweb.com/auto-function-roi#sample-code
	// CIntegerParameter m_exposureTime;
	CGrabResultPtr _ptrGrabResult;
	int _timeoutMs;
	int _xin, _yin, _xout, _yout, _areaFinImg, _widthFinImg, _heightFinImg;
};

static void usage(const char* pname) {

	const char* bname = nullptr;//@R = rindex(pname, '/');

	fprintf(stdout,
		"Usage: %s [options]\n"
		"options:\n"
		"\t-a|--filename  <String>\n"
		"\t\tfilename containing numbers\n"
		"\n"
		"\t-x|--start temperature <FLOAT>\n"
		"\t\t \n"
		"\n"
		"\t-y|--stop temperature <FLOAT>\n"
		"\t\tnumber of lattice columns\n"
		"\n"
		"\t-n|--niters <INT>\n"
		"\t\tnumber of iterations\n"
		"\n"
		"\t-n|--sweeps_per_beta <INT>\n"
		"\t\tnumber of sweep per temperature\n"
		"\n"
		"\t-s|--seed <SEED>\n"
		"\t\tfix the starting point\n"
		"\n"
		"\t-s|--debug \n"
		"\t\t Print the final lattice value and shows avg magnetization at every temperature\n"
		"\n"
		"\t-o|--write-lattice\n"
		"\t\twrite final lattice configuration to file\n\n",
		bname);
	exit(EXIT_SUCCESS);
}

#define MYDEBU 1


// ./SPIM.exe -a numbers64.csv -x 6.4 -y 0.01 -n 3 -m 1 -1 1290 -2 508 -w 72 -h 92 -e 5000 -5 8 -6 512
int main(int argc, char* argv[])
{
	vector<float> energies;
	energies.push_back(255.0);
	std::string filename = "";//argv[1]
	std::string linear_file;

	float start_temp = 20.f;
	float stop_temp = 0.01f;
	unsigned long long seed = ((GetCurrentProcessId() * rand()) & 0x7FFFFFFFF);

	unsigned int num_temps = 1000; //atoi(argv[2]);
	unsigned int num_sweeps_per_beta = 1;//atoi(argv[3]);


	bool write = false;
	bool debug = false;

	int CamtimeoutMs = 5000;
	int width = 1920;
	int height = 1080;
	int offsetx = 64;
	int offsety = 4;
	float exposure_value = 1100.0;// reduce for faster data transfer
	int out_bi = 16;
	int are = 1024;
	int bi = 8;
	int act_are = 512;

	std::cout << "Start parsing the file containing numbers" << std::endl;

	while (1) {
		static struct option long_options[] = {
			{     "Number_filename", required_argument, 0, 'a'},
			{     "start_temp", required_argument, 0, 'x'},
			{     "stop_temp", required_argument, 0, 'y'},
			{          "seed", required_argument, 0, 's'},
			{        "niters", required_argument, 0, 'n'},
			{ "sweeps_per_beta", required_argument, 0, 'm'},
			{  "CamTimeOut",       required_argument, 0, 't'},
			{  "width",       required_argument, 0, 'w'},
			{  "height",       required_argument, 0, 'h'},
			{  "offsetx",       required_argument, 0, '1'},
			{  "offsety",       required_argument, 0, '2'},
			{  "exposure_value",       required_argument, 0, 'e'},
			{  "outer_bin",       required_argument, 0, '3'},
			{  "area",       required_argument, 0, '4'},
			{  "bin",       required_argument, 0, '5'},
			{  "active_area",       required_argument, 0, '6'},
			{ "write-lattice",       no_argument, 0, 'o'},
			{          "debug",       no_argument, 0, 'd'},
			{          "help",       no_argument, 0, 'z'},
			{               0,                 0, 0,   0}
		};

		int option_index = 0;
		int ch = getopt_long(argc, argv, "a:x:y:s:n:m:t:w:h:1:2:e:3:4:5:6:odz", long_options, &option_index);
		if (ch == -1) break;

		switch (ch) {
		case 0:
			break;
		case 'a':
			filename = (optarg); break;
		case 'x':
			start_temp = atof(optarg); break;
		case 'y':
			stop_temp = atof(optarg); break;
		case 's':
			seed = atoll(optarg);
			break;
		case 'n':
			num_temps = atoi(optarg); break;
		case 'm':
			num_sweeps_per_beta = atoi(optarg); break;
		case 'o':
			write = true; break;
		case 'd':
			debug = true; break;
		case 'z':
			usage(argv[0]); break;
		case '?':
			exit(EXIT_FAILURE);
		case 't':
			CamtimeoutMs = atoi(optarg); break;
		case 'w':
			width = atoi(optarg); break;
		case 'h':
			height = atoi(optarg); break;
		case '1':
			offsetx = atoi(optarg); break;
		case '2':
			offsety = atoi(optarg); break;
		case 'e':
			exposure_value = atof(optarg); break;
		case '3':
			out_bi = atoi(optarg); break;
		case '4':
			are = atoi(optarg); break;
		case '5':
			bi = atoi(optarg); break;
		case '6':
			act_are = atoi(optarg); break;
		default:
			fprintf(stderr, "unknown option: %c\n", ch);
			exit(EXIT_FAILURE);
		}
	}

	HOLOEYE_UNUSED(argc);
	HOLOEYE_UNUSED(argv);


		// Check if the installed SDK supports the required API version
		if (!heds_requires_version(3, false))
			return 1;
		// Before using any pylon methods, the pylon runtime must be initialized.
		PylonInitialize();
		// The exit code of the sample application.

		BaslerCamera baslerCamera = BaslerCamera(CamtimeoutMs, width, height, offsetx, offsety, exposure_value);
		int exitCode = 0;
		//baslerCamera.checkImage();
#if MYDEBU
		// Detect SLMs and open a window on the selected SLM:
		heds_instance slm;
		heds_errorcode error = slm.open();
		if (error != HEDSERR_NoError)
		{
			std::cerr << "ERROR: " << heds_error_string_ascii(error) << std::endl;
			return error;
		}

		// Open the SLM preview window in "Fit" mode:
		// Please adapt the file show_slm_preview.h if preview window
		// is not at the right position or even not visible.
		// The additional flag HEDSSLMPF_ShowZernikeRadius presses the button to
		// show the Zernike radius visualization in preview window from code.
		
		error = show_slm_preview(0.0);
		if (error != HEDSERR_NoError)
		{
			std::cerr << "ERROR: " << heds_error_string_ascii(error) << std::endl;
			return error;
		}

		// pre-calc. helper variables:
		const float phaseModulation = 2.0f * HOLOEYE_PIF;
		const int dataWidth = heds_slm_width_px();
		const int dataHeight = heds_slm_height_px();

		// Reserve memory for the phase data matrix.
		// Use data type single to optimize performance:
		auto phaseData = HoloeyeType::create(dataWidth, dataHeight);
		
		// phaseData.refreshrate()
		std::cout << "dataWidth  = " << dataWidth << std::endl;
		std::cout << "dataHeight = " << dataHeight << std::endl;

		// Display checkerboard of size {1024, 1024}  and bins {16, 16}
		int outer_bin = out_bi;//pow(2, 4); 
		pair< int, int> area = { are, are };
		DisplayCheckerBoardPattern(phaseData, dataWidth, dataHeight, area, outer_bin);
		// Show phase data on SLM:
		error = heds_show_phasevalues(phaseData, HEDSSHF_PresentAutomatic, phaseModulation);
		if (error != HEDSERR_NoError)
		{
			std::cerr << "ERROR: " << heds_error_string_ascii(error) << std::endl;
			return error;
		}
		

		std::this_thread::sleep_for(std::chrono::milliseconds(150));
		std::cout << "Do Ur calibration Using Pylon GUI and then press enter" << std::endl;
		char ch = getchar();
		//
		ParseData parseObj = ParseData();

		std::vector<float> numberVect;
		parseObj.readNumberCSV(filename, numberVect);
		float maxNum = *std::max_element(numberVect.begin(), numberVect.end());

		for (int i = 0; i < numberVect.size(); i++)
			numberVect[i] = acos(numberVect.at(i) / maxNum);

		std::vector<float> isingSpins;
		srand(time(0));
		for (int i = 0; i < numberVect.size(); ++i)
			isingSpins.push_back( float( 2.f * (rand() % 2) - 1.f) );

		//for (int i = 0; i < isingSpins.size(); ++i)
		//	isingSpins[i] = (isingSpins[i] + 1) * HOLOEYE_PIF / 2 + HOLOEYE_PIF / 2; //@R remove PI from Ising Spins

 		ASSERT(numberVect.size() == isingSpins.size());
		int bin = bi;// pow(2, 3);// pow(2, 7);
		pair<int, int> active_area = { act_are, act_are};
		cout << "1. InitialSLMLattice" << endl;
		InitialSLMLattice(phaseData, dataWidth, dataHeight, numberVect, isingSpins, active_area, bin);

		// Show phase data on SLM:
		error = heds_show_phasevalues(phaseData, HEDSSHF_PresentAutomatic, phaseModulation);
		if (error != HEDSERR_NoError)
		{
			std::cerr << "ERROR: " << heds_error_string_ascii(error) << std::endl;
			return error;
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(150));
		std::cout << "See the expected pattern on camera" << std::endl;
		//baslerCamera.checkImage();// @R
		

		vector<double> vecBetas = create_beta_schedule_linear(num_temps, 1.f / start_temp, 1.f / stop_temp);
		cout << "2. baslerCamera.openBaslerCamera()" << endl;
		baslerCamera.openBaslerCamera();

		// Lattice creation
		vector<pair< unsigned int, unsigned int> >  spinLatticePts = SpinTuple( active_area, bin);
		vector<unsigned int> selLatticeIndex; 
		selLatticeIndex.resize(NUM_SPINS_FLIP, 0);


		auto start = std::chrono::high_resolution_clock::now();
		cout << "2.5. MH Loop " << endl;
		// Flip lattice
		for (int count = 0; count < vecBetas.size(); )
		{
			for (int ii = 0; ii < num_sweeps_per_beta; ++ii)
			{	
				for (int i = 0; i < NUM_SPINS_FLIP; i++)
					selLatticeIndex[i] = rand() % spinLatticePts.size();

				cout << "3. FLipLattice()" << endl;
				FLipLattice(phaseData, dataWidth, dataHeight, isingSpins, active_area, bin, spinLatticePts, selLatticeIndex);
				cout << "3.1 FLipLattice()" << endl;
				// Show phase data on SLM:
				error = heds_show_phasevalues(phaseData, HEDSSHF_PresentAutomatic, phaseModulation);
				if (error != HEDSERR_NoError)
				{
					std::cerr << "ERROR: " << heds_error_string_ascii(error) << std::endl;
					return error;
				}
				cout << "3.2 FLipLattice()" << endl;
				std::this_thread::sleep_for(std::chrono::milliseconds(150));
				try
				{
					float energy = 0.f; // baslerCamera.collectSingleImageNEnergy(); // @R
					// MH algo
					float delE = energy - energies[energies.size() - 1];
					// sum of Ising spins with numbers for fidelity
					float prob = exp(-1.f * vecBetas[count] * delE);
					float acceptance_probability = min((float)1.f, prob);
					// Flip back if not selected in MH iter
					double gen_pro = ((double)rand() / (RAND_MAX));
					if (delE <= 0)
					{
						energies.push_back(energy);
					}
					else if (gen_pro < acceptance_probability)
					{
						energies.push_back(energy);
					}
					else
					{
						FLipLattice(phaseData, dataWidth, dataHeight, isingSpins, active_area, bin, spinLatticePts, selLatticeIndex);
					}
					cout << "3.4 FLipLattice()" << endl;
					count++;

				}
				catch (const GenericException & e)
				{
					// Error handling.
					std::cerr << "An exception occurred." << endl
						<< e.GetDescription() << endl;
					exitCode = 1;
				}
			}
		}
		auto end = std::chrono::high_resolution_clock::now();

		double duration = (double)std::chrono::duration_cast<std::chrono::microseconds>(start - end).count();

		std::cout << "Number of count: " << vecBetas.size() << " elapsed time: " << duration * 1e-6 << " ms\n";
	
		// Wait until the SLM process was closed
		std::cout << "Waiting for SDK process to close. Please close the tray icon to continue ..." << std::endl << std::flush;
		error = heds_utils_wait_until_closed();

		if (error != HEDSERR_NoError)
		{
			std::cerr << "ERROR: " << heds_error_string_ascii(error) << std::endl;

			return error;
		}

#endif
		baslerCamera.closeBaslerCamera();
		// Releases all pylon resources.
		PylonTerminate();		
		// Comment the following two lines to disable waiting on exit.
		std::cerr << endl << "Press enter to exit." << endl;
		while (std::cin.get() != '\n');

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