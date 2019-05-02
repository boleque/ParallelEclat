#pragma once
#define __CL_ENABLE_EXCEPTIONS
#include "cl.hpp"
#include "CL/cl2.hpp"
#include "Types.h"

class FrequentPatternSearcher
{

public:
    /** \brief Конструктор класса
     *  \param[in] min_supp - минимальная поддержка.
     */
    FrequentPatternSearcher(const unsigned int min_supp);

    /** \brief Деструктор класса
     */
    ~FrequentPatternSearcher();

    /** \brief Метод предназначен для подготовки и запуска вычисления на DEVICE с использованием оптимизированного алгоритма суммирования префиксов.
     *  \param[in] root - корень дерева частых наборов.
     *  \param[in] device - GPU устройство.
     */
    void performanceOnDeviceOptimizedPrefixSum(NodePtr root, const cl::Device& device);
    
    /** \brief Метод предназначен для подготовки и запуска вычисления на DEVICE с использованием наивного алгоритма суммирования префиксов.
     *  \param[in] root - корень дерева частых наборов.
     *  \param[in] device - GPU устройство.
     */
    void performanceOnDeviceNaivePrefixSum(NodePtr root, const cl::Device& device);

    /** \brief Метод предназначен для подготовки и запуска вычисления на HOST.
     *  \param[in] root - корень дерева частых наборов.
     */
    void performanceOnHost(NodePtr root);

private:
    /** \brief Метод предназначен для извлечения частых наборов на стороне HOST
     *  \param[in] parent  - узел предка
     */
    void retrieveFreqItemSetsOnHost(const NodePtr& parent);

    /** \brief Метод предназначен для извлечения частых наборов на стороне DEVICE c использованием наивного алгоритма суммирования префиксов
     *  \param[in] kernel  - OpenCL ядро.
     *  \param[in] queue   - очередь команд.
     *  \param[in] context - контекст.
     *  \param[in] parent  - узел предка
     */
    void retrieveFreqItemSetsOnDeviceNaivePrefixSum(cl::Kernel& kernel, cl::CommandQueue& queue, cl::Context& context, const NodePtr& parent);
   
    /** \brief Метод предназначен для извлечения частых наборов на стороне DEVICE с использованием оптимизированного средствами OpenCL 2.0 суммирования префиксов
     *  \param[in] kernel  - OpenCL ядро.
     *  \param[in] queue   - очередь команд.
     *  \param[in] context - контекст.
     *  \param[in] parent  - указатель на узел предка
     */
    void retrieveFreqItemSetsOnDeviceOptimizedPrefix(cl::Kernel& kernel, cl::CommandQueue& queue, cl::Context& context, const NodePtr& parent);

private:
    unsigned int m_minSupport;    /** Минимальная поддержка. */
    TimeValues   m_hostCalcTimes; /** Время выполнения операции слияния списков. */
    TimeValues   m_deviceCalcTimesWithoutOptim; /** Время выполнения операции слияния списков. */
    TimeValues   m_deviceCalcTimesOptim;        /** Время выполнения операции слияния списков. */

    double       m_hostPerformanceTimeMS; /** Время HOST части. */
};
