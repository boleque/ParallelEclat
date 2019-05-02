#pragma once
#define __CL_ENABLE_EXCEPTIONS
#include "cl.hpp"
#include "CL/cl2.hpp"
#include "Types.h"

class FrequentPatternSearcher
{

public:
    /** \brief ����������� ������
     *  \param[in] min_supp - ����������� ���������.
     */
    FrequentPatternSearcher(const unsigned int min_supp);

    /** \brief ���������� ������
     */
    ~FrequentPatternSearcher();

    /** \brief ����� ������������ ��� ���������� � ������� ���������� �� DEVICE � �������������� ����������������� ��������� ������������ ���������.
     *  \param[in] root - ������ ������ ������ �������.
     *  \param[in] device - GPU ����������.
     */
    void performanceOnDeviceOptimizedPrefixSum(NodePtr root, const cl::Device& device);
    
    /** \brief ����� ������������ ��� ���������� � ������� ���������� �� DEVICE � �������������� �������� ��������� ������������ ���������.
     *  \param[in] root - ������ ������ ������ �������.
     *  \param[in] device - GPU ����������.
     */
    void performanceOnDeviceNaivePrefixSum(NodePtr root, const cl::Device& device);

    /** \brief ����� ������������ ��� ���������� � ������� ���������� �� HOST.
     *  \param[in] root - ������ ������ ������ �������.
     */
    void performanceOnHost(NodePtr root);

private:
    /** \brief ����� ������������ ��� ���������� ������ ������� �� ������� HOST
     *  \param[in] parent  - ���� ������
     */
    void retrieveFreqItemSetsOnHost(const NodePtr& parent);

    /** \brief ����� ������������ ��� ���������� ������ ������� �� ������� DEVICE c �������������� �������� ��������� ������������ ���������
     *  \param[in] kernel  - OpenCL ����.
     *  \param[in] queue   - ������� ������.
     *  \param[in] context - ��������.
     *  \param[in] parent  - ���� ������
     */
    void retrieveFreqItemSetsOnDeviceNaivePrefixSum(cl::Kernel& kernel, cl::CommandQueue& queue, cl::Context& context, const NodePtr& parent);
   
    /** \brief ����� ������������ ��� ���������� ������ ������� �� ������� DEVICE � �������������� ����������������� ���������� OpenCL 2.0 ������������ ���������
     *  \param[in] kernel  - OpenCL ����.
     *  \param[in] queue   - ������� ������.
     *  \param[in] context - ��������.
     *  \param[in] parent  - ��������� �� ���� ������
     */
    void retrieveFreqItemSetsOnDeviceOptimizedPrefix(cl::Kernel& kernel, cl::CommandQueue& queue, cl::Context& context, const NodePtr& parent);

private:
    unsigned int m_minSupport;    /** ����������� ���������. */
    TimeValues   m_hostCalcTimes; /** ����� ���������� �������� ������� �������. */
    TimeValues   m_deviceCalcTimesWithoutOptim; /** ����� ���������� �������� ������� �������. */
    TimeValues   m_deviceCalcTimesOptim;        /** ����� ���������� �������� ������� �������. */

    double       m_hostPerformanceTimeMS; /** ����� HOST �����. */
};
