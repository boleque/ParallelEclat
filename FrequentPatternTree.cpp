#include "FrequentPatternTree.h"

void Node::joinTIDListsOnHost(const NodePtr _node, vector<int>& out)
{
    int firstTIDListIndx = 0;
    int secondTIDListIndx = 0;

    while (firstTIDListIndx < _node->tidList.size())
    {
        while (secondTIDListIndx < tidList.size())
        {
            if (_node->tidList[firstTIDListIndx] == tidList[secondTIDListIndx])
            {
                out.push_back(_node->tidList[firstTIDListIndx]);
                ++secondTIDListIndx;
                break;
            }

            if (_node->tidList[firstTIDListIndx] < tidList[secondTIDListIndx])
                break;

            ++secondTIDListIndx;
        }
        ++firstTIDListIndx;
    }
}

void Node::printTree(NodePtr root)
{
    printChild(root, "itemset");
}

void Node::printChild(const NodePtr& parent, const string& head)
{
    stringstream ss;
    for (const auto& child : parent->children)
    {
        cout << head << "-" << child->item << "\n\n";
        string newHead;
        ss << head << "-" << child->item;
        ss >> newHead;
        ss.clear();
        printChild(child, newHead);
    }
}

void Node::addNewNodeOnHost(NodePtr& parent, string item, vector<int>& itemTIDList)
{
    NodePtr new_node = make_shared<Node>();
    new_node->item = item;
    new_node->tidList.swap(itemTIDList);
    parent->children.emplace_back(new_node);
}

void Node::addNewNodeOnDevice(NodePtr& parent, string item, vector<int> itemTIDList)
{
    NodePtr new_node = make_shared<Node>();
    new_node->item = item;
    new_node->tidList = itemTIDList;
    parent->children.emplace_back(new_node);
}