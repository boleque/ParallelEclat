#pragma once

#include <memory>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>

#include <fstream>
#include <string>
#include <sstream>
#include <map>
#include <assert.h>
#include "Types.h"

using namespace std;

/** \class Node
*  \brief ��������� ������������ ����� ���� ������ ������ �������
*/
struct Node
{
    /** \brief ����� ������������ ��� ����������� TID ������� �����.
    *  \param[in] node - ����.
    *  \return ��������� ����������� TID ������� - ����� ������
    */
    void joinTIDListsOnHost(const NodePtr _node, vector<int>& out);

    /** \brief ����� ������������ ��� ������ �� ������ ������ �������.
    *  \param[in] root - ������ ������.
    */
    static void printTree(NodePtr root);

    /** \brief ����� ������������ ��� ������ �� ������ ������� ����.
    *  \param[in] parent - ��������.
    *  \param[in] head - ���������.
    */
    static void printChild(const NodePtr& parent, const string& head);

    /** \brief ����� ������������ ��� ���������� � ������ ������ ����.
    *  \param[in] parent - ��������.
    *  \param[in] item - �������.
    *  \param[in] itemTIDList - ������ TID.
    */
    void addNewNodeOnHost(NodePtr& parent, string item, vector<int>& itemTIDList);

    /** \brief ����� ������������ ��� ���������� � ������ ������ ����.
    *  \param[in] parent - ��������.
    *  \param[in] item - �������.
    *  \param[in] itemTIDList - ������ TID.
    */
    void addNewNodeOnDevice(NodePtr& parent, string item, vector<int> itemTIDList);

	string          item;     /** ������� (item) ����. */
    vector<int>     tidList;  /** ������ ���������� (TID list), � ������� ������ ������� (item). */
    vector<NodePtr> children; /** ������ �������� ����. */
};