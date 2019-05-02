#pragma once

#include <memory>
#include <vector>

enum EnCalcType
{
    enDeviceGlobalMemory,
    enDeviceLocalMemory,
    enDeviceCalcTypesCount,
};

using NodePtr    = std::shared_ptr<struct Node>;
using TID_list   = std::vector<int>;
using const_iter = std::vector<int>::const_iterator;
using TimeValues = std::vector<double>;