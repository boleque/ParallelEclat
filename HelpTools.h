#pragma once

#include <memory>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <string>
#include <sstream>
#include <map>
#include <assert.h>

#include "Types.h"
#include "FrequentPatternTree.h"

struct HelpTools
{
    /** \brief Метод предназначен для вывода статистики времени на экран
     */
    static void printTimeStatistic(TimeValues times, double hostPerformanceTimeMS)
    {
        std::sort(times.begin(), times.end());
        double totalTime = accumulate(times.begin(), times.end(), 0.0);
        double averageTime = totalTime / times.size();
        double minTime = times[0];
        double maxTime = times[times.size() - 1];
        double medianTime = times[times.size() / 2];
        cout << "Calculation time statistic: (" << times.size() << " runs)" << endl;
        cout << "Med: " << medianTime << " ms (" << hostPerformanceTimeMS / medianTime << "X faster then host)" << endl;
        cout << "Avg: " << averageTime << " ms" << endl;
        cout << "Min: " << minTime << " ms" << endl;
        cout << "Max: " << maxTime << " ms" << endl << endl;
    }

    /** \brief Метод предназначен для конвертации битового представления числа в int .
     *  \param[in] begin - указатель на начало битового представления.
     *  \param[in] size - размер битового представления.
     */
    static int convertBitVecToInt(const_iter begin, int size)
    {
        int res = 0x00;
        int shift = size - 1;
        for (int indx = 0; indx < size; ++indx)
        {
            if (*(begin + indx))
                res |= 0x01 << shift;

            --shift;
        }

        return res;
    }

    /** \brief Метод предназначен для чтения базы транзакций и предобработки (для GPU).
     *  \param[in] filename - имя файла.
     *  \param[in] minsup - минимальная поддержка.
     *  \param[in] root - корень дерева.
     */
    static bool readTransBase(const std::string& fileName, const unsigned int minsup, const unsigned int groupsize, NodePtr root)
    {
	    std::ifstream file;
	    file.open(fileName.c_str());
	    if (!file)
	    {
		    assert("Can not open file!");
		    return false;
	    }

	    string line;
	    stringstream ss;

	    int TID = 1;

	    map<string, TID_list > tmp;
	    map<string, TID_list > vertTransacView;

	    while (getline(file, line))
	    {
		    if (line.empty())
			    break;

		    ss << line;
			string item;

		    while (ss >> item)
		    {
			    if (tmp.find(item) == tmp.end())
			    {
				    vector<int> tids;
				    tids.push_back(TID);
				    tmp.insert(std::pair<string, vector<int> >(item, tids));
			    }
			    else
				    tmp[item].push_back(TID);
		    }

		    ss.clear();
		    ++TID;
	    }

	    while (TID % groupsize != 0)
		    ++TID;
	
        for (const auto& TransacRow : tmp)
        {
            if (TransacRow.second.size() < minsup)
                continue;

            TID_list bittidList(TID, 0);
            for (const auto& tid : TransacRow.second)
			    bittidList[tid] = 1;

		    int tidlistNum = 0;
		    const_iter iterator = bittidList.cbegin();
		    TID_list dectidList;
		    for (int iShift = 0; iShift < bittidList.size(); iShift += groupsize)
			    dectidList.emplace_back(convertBitVecToInt(iterator + iShift, groupsize));
			
            vertTransacView.insert(std::pair<string, vector<int> >(TransacRow.first, dectidList));
        }

        for (const auto& TransacRow : vertTransacView)
            root->addNewNodeOnDevice(root, TransacRow.first, TransacRow.second);

        file.close();
    }

    /** \brief Метод предназначен для чтения базы транзакций и предобработки.
     *  \param[in] filename - имя файла.
     *  \param[in] minsup - минимальная поддержка.
     *  \param[in] root - корень дерева.
     */
    static bool readTransBase(const std::string& fileName, const unsigned int minsup, NodePtr root)
    {
	    std::ifstream file;
	    file.open(fileName.c_str());
	    if (!file)
	    {
		    assert("Can not open file!");
		    return false;
	    }

	    string line;
	    stringstream ss;

	    int TID = 1;
	    map<int, TID_list > vertTransacView;
	    int colCount = 0;
	    while (getline(file, line))
	    {
		    if (line.empty())
			    break;

		    ++colCount;

		    ss << line;
		    int item;

		    while (ss >> item)
		    {
			    if (vertTransacView.find(item) == vertTransacView.end())
			    {
				    vector<int> tids;
				    tids.push_back(TID);
				    vertTransacView.insert(std::pair<int, vector<int> >(item, tids));
			    }
			    else
				    vertTransacView[item].push_back(TID);
		    }

		    ss.clear();
		    ++TID;
	    }

	    for (auto& TransacRow : vertTransacView)
	    {
		    if (TransacRow.second.size() < minsup)
			    continue;

		    root->addNewNodeOnHost(root, TransacRow.first, TransacRow.second);
	    }

	    file.close();
    }
};


