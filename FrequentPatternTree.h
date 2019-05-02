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
*  \brief Структура представляет собой Узел дерева частых наборов
*/
struct Node
{
    /** \brief Метод предназначен для объединения TID списков узлов.
    *  \param[in] node - узел.
    *  \return результат объединения TID списков - новый список
    */
    void joinTIDListsOnHost(const NodePtr _node, vector<int>& out);

    /** \brief Метод предназначен для вывода на печать дерева наборов.
    *  \param[in] root - корень дерева.
    */
    static void printTree(NodePtr root);

    /** \brief Метод предназначен для вывода на печать потомка узла.
    *  \param[in] parent - родитель.
    *  \param[in] head - заголовок.
    */
    static void printChild(const NodePtr& parent, const string& head);

    /** \brief Метод предназначен для добавления в дерево нового узла.
    *  \param[in] parent - родитель.
    *  \param[in] item - элемент.
    *  \param[in] itemTIDList - список TID.
    */
    void addNewNodeOnHost(NodePtr& parent, string item, vector<int>& itemTIDList);

    /** \brief Метод предназначен для добавления в дерево нового узла.
    *  \param[in] parent - родитель.
    *  \param[in] item - элемент.
    *  \param[in] itemTIDList - список TID.
    */
    void addNewNodeOnDevice(NodePtr& parent, string item, vector<int> itemTIDList);

	string          item;     /** Элемент (item) узла. */
    vector<int>     tidList;  /** список транзакций (TID list), в которые входит элемент (item). */
    vector<NodePtr> children; /** Список потомков узла. */
};