#include <cstdlib>
#include <iostream>
#include <map>
#include <vector>
#include <ctime>
#include <iostream>
#include <string>
#include <algorithm>
#include <conio.h>
#include <numeric>
#include <chrono>

#include "Types.h"
#include "FrequentPatternSearcher.h"
#include "HelpTools.h"

using namespace std;

constexpr unsigned int MIN_SUPPORT = 3;  /**< Минимальная поддержка. */
constexpr unsigned int GROUP_SIZE = 32; /**< Размер представления списка транзакций. */
constexpr unsigned int TESTS_COUNT = 1;  /**< Число тестов. */


class Kernel
{
public:
	using Ptr = std::shared_ptr<Kernel>;
	void printLog(std::string& msg)
	{
		std::cout << msg << std::endl;
	}

	static Ptr get()
	{
		static Kernel* kernel = new Kernel;
		return Ptr(kernel);
	}
};

class Backend
{
public:
	Backend()
	{
		m_kernel = Kernel::get();
		m_kernel->printLog(std::string("Ok"));
	}

private:
	Kernel::Ptr m_kernel;
};



int main(int argc, char* argv[])
{
	const std::string transacBaseFile = "base_for_debug.txt";
	FrequentPatternSearcher freqPatternTreeSearcher(MIN_SUPPORT);

	NodePtr freqPatternTreeRootsHost = std::make_shared<Node>();
    assert(HelpTools::readTransBase(transacBaseFile, MIN_SUPPORT, freqPatternTreeRootsHost));

    //----------------------------------------------------------------------------------------------------------------------
	cout << "Device: Host" << endl << endl;
    freqPatternTreeSearcher.performanceOnHost(freqPatternTreeRootsHost);
    //----------------------------------------------------------------------------------------------------------------------

	vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	
    vector<cl::Platform> platfroms2_0;
	for (auto &p : platforms) {
		std::string platver = p.getInfo<CL_PLATFORM_VERSION>();
		if (platver.find("OpenCL 2.") != std::string::npos) 
        {
            platfroms2_0.emplace_back(p);
		}
	}
	if (platfroms2_0.size() == 0) {
		std::cout << "No OpenCL 2.0 platform found.";
		return -1;
	}

	NodePtr freqPatternTreeRootsDevice[enDeviceCalcTypesCount] = { std::make_shared<Node>(), std::make_shared<Node>() };
    assert(HelpTools::readTransBase(transacBaseFile, MIN_SUPPORT, GROUP_SIZE, freqPatternTreeRootsDevice[enDeviceLocalMemory]));
    assert(HelpTools::readTransBase(transacBaseFile, MIN_SUPPORT, GROUP_SIZE, freqPatternTreeRootsDevice[enDeviceGlobalMemory]));

	//----------------------------------------------------------------------------------------------------------------------
	std::vector<cl::Device> devices;
    platfroms2_0[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

	try
	{
		cout << "Device: " << devices[0].getInfo<CL_DEVICE_NAME>() << endl << endl;
        freqPatternTreeSearcher.performanceOnDeviceNaivePrefixSum(freqPatternTreeRootsDevice[enDeviceGlobalMemory], devices[0]);
	}
	catch (cl::Error error)
	{
		std::cout << error.what() << "(" << error.err() << ")" << std::endl;
	}

	//----------------------------------------------------------------------------------------------------------------------

	try
	{
		cout << "Device: " << devices[0].getInfo<CL_DEVICE_NAME>() << endl << endl;
        freqPatternTreeSearcher.performanceOnDeviceOptimizedPrefixSum(freqPatternTreeRootsDevice[enDeviceLocalMemory], devices[0]);
	}
	catch (cl::Error error)
	{
		std::cout << error.what() << "(" << error.err() << ")" << std::endl;
	}

	_getch();
	return 0;
}
