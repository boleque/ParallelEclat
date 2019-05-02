#include "FrequentPatternSearcher.h"

#include <memory>
#include <vector>
#include "FrequentPatternTree.h"
#include "HelpTools.h"

using namespace std;
constexpr unsigned int TESTS_COUNT = 1;  /**< Число тестов. */

FrequentPatternSearcher::FrequentPatternSearcher(const unsigned int min_supp) :
   m_minSupport(min_supp)
{}

FrequentPatternSearcher::~FrequentPatternSearcher()
{}

void FrequentPatternSearcher::retrieveFreqItemSetsOnHost(const NodePtr& parent)
{
    vector<NodePtr> currentNodeChildren = parent->children;
    vector<int> joinedTidList;
    for (int i = 0; i < currentNodeChildren.size(); ++i)
    {
        int isNewPattern = 0;
        for (int j = (i + 1); j < currentNodeChildren.size(); ++j)
        {

            __int64 start_count;
            __int64 end_count;
            __int64 freq;
            QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
            QueryPerformanceCounter((LARGE_INTEGER*)&start_count);

            currentNodeChildren[i]->joinTIDListsOnHost(currentNodeChildren[j], joinedTidList);

            QueryPerformanceCounter((LARGE_INTEGER*)&end_count);
            double time = 1000 * (double)(end_count - start_count) / (double)freq;
            m_hostCalcTimes.push_back(time);

            if (joinedTidList.size() >= m_minSupport)
            {
                parent->addNewNodeOnHost(currentNodeChildren[i], currentNodeChildren[j]->item, joinedTidList);
                ++isNewPattern;
            }
            joinedTidList.clear();
        }
        if (isNewPattern > 0)
            retrieveFreqItemSetsOnHost(currentNodeChildren[i]);
    }
}

void FrequentPatternSearcher::retrieveFreqItemSetsOnDeviceNaivePrefixSum(cl::Kernel& kernel, cl::CommandQueue& queue, cl::Context& context, const NodePtr& parent)
{
    vector<NodePtr> currentNodeChildren = parent->children;
    int buffersSize = currentNodeChildren.size();
    int tidlistSize = currentNodeChildren[0]->tidList.size();
    int matches = 0;

    vector<int> joinedTidList(tidlistSize, 0);
	vector<int> prefixSumList(tidlistSize, 0);
    for (int i = 0; i < buffersSize; ++i)
    {
        int isNewPattern = 0;
        for (int j = (i + 1); j < buffersSize; ++j)
        {
			// Создаем буферы данных
            cl::Buffer downTIDListBuff   = cl::Buffer(context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, tidlistSize * sizeof(int), currentNodeChildren[j]->tidList.data());
            cl::Buffer upTIDListBuff	 = cl::Buffer(context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, tidlistSize * sizeof(int), currentNodeChildren[i]->tidList.data());
            cl::Buffer joinedTIDListBuff = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, tidlistSize * sizeof(int), joinedTidList.data());
            cl::Buffer prefixSumListBuff = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, tidlistSize * sizeof(int), prefixSumList.data());

            //Установка аргументов ядра
            int kernelArg = 0;
			kernel.setArg(kernelArg++, sizeof(int), &tidlistSize);
            kernel.setArg(kernelArg++, prefixSumListBuff);
            kernel.setArg(kernelArg++, upTIDListBuff);
            kernel.setArg(kernelArg++, downTIDListBuff);
            kernel.setArg(kernelArg++, joinedTIDListBuff);
            kernel.setArg(kernelArg++, tidlistSize * sizeof(int), nullptr);

            __int64 start_count;
            __int64 end_count;
            __int64 freq;
            QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
            QueryPerformanceCounter((LARGE_INTEGER*)&start_count);

            // Запуск ядра на выполнение
            queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(tidlistSize), cl::NDRange(tidlistSize));
            queue.finish();

            QueryPerformanceCounter((LARGE_INTEGER*)&end_count);
            double time = 1000 * (double)(end_count - start_count) / (double)freq;
            m_deviceCalcTimesWithoutOptim.push_back(time);

            // считываем данные
            queue.enqueueReadBuffer(joinedTIDListBuff, CL_TRUE, 0, tidlistSize * sizeof(int), joinedTidList.data());
            queue.enqueueReadBuffer(prefixSumListBuff, CL_TRUE, 0, tidlistSize * sizeof(int), prefixSumList.data());

            if (prefixSumList.back() >= m_minSupport)
            {
                parent->addNewNodeOnDevice(currentNodeChildren[i], currentNodeChildren[j]->item, joinedTidList);
                ++isNewPattern;
            }

            joinedTidList.clear();
            joinedTidList.resize(tidlistSize, 0);

			prefixSumList.clear();
			prefixSumList.resize(tidlistSize, 0);
        }
        if (isNewPattern > 0)
           retrieveFreqItemSetsOnDeviceNaivePrefixSum(kernel, queue, context, currentNodeChildren[i]);
    }
}

void FrequentPatternSearcher::retrieveFreqItemSetsOnDeviceOptimizedPrefix(cl::Kernel& kernel, cl::CommandQueue& queue, cl::Context& context, const NodePtr& parent)
{
    vector<NodePtr> currentNodeChildren = parent->children;
    int buffersSize = currentNodeChildren.size();
    int tidlistSize = currentNodeChildren[0]->tidList.size();
    int matches = 0;

    vector<int> joinedTidList(tidlistSize, 0);
    vector<int> prefixSumList(tidlistSize, 0);
    for (int i = 0; i < buffersSize; ++i)
    {
        int isNewPattern = 0;
        for (int j = (i + 1); j < buffersSize; ++j)
        {
            // Создаем буферы данных
            cl::Buffer downTIDListBuff   = cl::Buffer(context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, tidlistSize * sizeof(int), currentNodeChildren[j]->tidList.data());
            cl::Buffer upTIDListBuff     = cl::Buffer(context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, tidlistSize * sizeof(int), currentNodeChildren[i]->tidList.data());          
            cl::Buffer joinedTIDListBuff = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, tidlistSize * sizeof(int), joinedTidList.data());
            cl::Buffer prefixSumListBuff = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, tidlistSize * sizeof(int), prefixSumList.data());

            //Установка аргументов ядра
            int kernelArg = 0;
            kernel.setArg(kernelArg++, sizeof(int), &tidlistSize);
            kernel.setArg(kernelArg++, prefixSumListBuff);
            kernel.setArg(kernelArg++, upTIDListBuff);
            kernel.setArg(kernelArg++, downTIDListBuff);
            kernel.setArg(kernelArg++, joinedTIDListBuff);
            kernel.setArg(kernelArg++, tidlistSize * sizeof(int), nullptr);

            __int64 start_count;
            __int64 end_count;
            __int64 freq;
            QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
            QueryPerformanceCounter((LARGE_INTEGER*)&start_count);

            // Запуск ядра на выполнение
            queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(tidlistSize), cl::NDRange(tidlistSize));
            queue.finish();

            QueryPerformanceCounter((LARGE_INTEGER*)&end_count);
            double time = 1000 * (double)(end_count - start_count) / (double)freq;
            m_deviceCalcTimesOptim.push_back(time);

            // считываем данные
            queue.enqueueReadBuffer(joinedTIDListBuff, CL_TRUE, 0, tidlistSize * sizeof(int), joinedTidList.data());
            queue.enqueueReadBuffer(prefixSumListBuff, CL_TRUE, 0, tidlistSize * sizeof(int), prefixSumList.data());

            if (prefixSumList.back() >= m_minSupport)
            {
                parent->addNewNodeOnDevice(currentNodeChildren[i], currentNodeChildren[j]->item, joinedTidList);
                ++isNewPattern;
            }

            joinedTidList.clear();
            joinedTidList.resize(tidlistSize, 0);

            prefixSumList.clear();
            prefixSumList.resize(tidlistSize, 0);
        }
        if (isNewPattern > 0)
            retrieveFreqItemSetsOnDeviceOptimizedPrefix(kernel, queue, context, currentNodeChildren[i]);
    }
}

void FrequentPatternSearcher::performanceOnHost(NodePtr root)
{
    retrieveFreqItemSetsOnHost(root);
    HelpTools::printTimeStatistic(m_hostCalcTimes, m_hostPerformanceTimeMS);
    Node::printTree(root);
}

void FrequentPatternSearcher::performanceOnDeviceNaivePrefixSum(NodePtr root, const cl::Device& device)
{
    cout << "Join TID Lists using GPU ( with naive prefix sum )" << endl << endl;

    //Загрузка исходного кода, для выполнения на GPU
    std::ifstream sourceFile("joinTIDListsNaivePrefixSum.cl");
    std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));

    //сборка OpenCL программы и ядра
    cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));

    //Создаем контекст для выбранного устройства
    vector<cl::Device> contextDevices;
    contextDevices.push_back(device);
    cl::Context context(contextDevices);
    cl::CommandQueue queue(context, device);

    cl::Program program = cl::Program(context, source);
    program.build(contextDevices);
    cl::Kernel kernel(program, "joinTIDListsNaivePrefixSum");

    retrieveFreqItemSetsOnDeviceNaivePrefixSum(kernel, queue, context, root);
    HelpTools::printTimeStatistic(m_deviceCalcTimesWithoutOptim, m_hostPerformanceTimeMS);
    Node::printTree(root);
}

void FrequentPatternSearcher::performanceOnDeviceOptimizedPrefixSum(NodePtr root, const cl::Device& device)
{
    cout << "Join TID Lists using GPU ( with optimized sum )" << endl << endl;

    //Загрузка исходного кода, для выполнения на GPU
    std::ifstream sourceFile("joinTIDListsOptimizedSum.cl");
    std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));

    //сборка OpenCL программы и ядра
    cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));

    //Создаем контекст для выбранного устройства
    vector<cl::Device> contextDevices;
    contextDevices.push_back(device);
    cl::Context context(contextDevices);
    cl::CommandQueue queue(context, device);

    cl::Program program = cl::Program(context, source);
    program.build(contextDevices, "-cl-std=CL2.0");
    cl::Kernel kernel(program, "joinTIDListsOptimizedSum");


    retrieveFreqItemSetsOnDeviceOptimizedPrefix(kernel, queue, context, root);
    HelpTools::printTimeStatistic(m_deviceCalcTimesOptim, m_hostPerformanceTimeMS);
    Node::printTree(root);
}