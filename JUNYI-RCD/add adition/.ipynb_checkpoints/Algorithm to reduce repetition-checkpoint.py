# -*- coding: utf-8 -*-
"""
Created on Sat May 13 20:07:10 2023

@author: 24807
"""
# 用python语言编写一个程序：实现以下步骤：
#①在一个有向图中，为各条边进行编号，编号名称为edge_id。找到该有向图中的所有环，为各环进行编号，编号名称为circuit_id

#②计算各条边分别经过几个环，统计该数据（称该数据为该边的重复值），用duplicate_values表示

#③找到duplicate_values最大的那条边，删除该边及该边所在的环。然后再次重复②、③步骤。

#④若在进行上述操作后，图中仍存在环，且环中各边的重复值相同。则选取父结点编号减去子结点编号的值最大的那条边进行删除。若父结点编号减去子结点编号差值最大的边不止一条，则在这些符合最大值的边中任选一条进行删除。

#重复上述操作，直至该图无环为止，逐次输出所删除的边的edge_id，并画出该有向图在处理之前和之后的图像。


#由于这是一道较为复杂的问题，需要考虑图的数据结构以及如何实现环和边的编号等操作。
# 以下是一个可行的Python程序，供您参考。

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

def calculate_duplicate_values(G, circuits):
    # 初始化每条边的重复值为0
    duplicate_values = {edge: 0 for edge in G.edges()}
    # 遍历每个环
    for circuit in circuits:
        # 计算该环中每条边的重复值
        for i in range(len(circuit)):
            # 获取当前边和下一条边
            curr_edge = circuit[i]
            next_edge = circuit[(i+1) % len(circuit)]
            # 对当前边和下一条边的重复值进行累加
            duplicate_values[curr_edge] += 1
            duplicate_values[next_edge] += 1
    return duplicate_values

def get_max_duplicate_value_edge(G, duplicate_values):
    # 找到重复值最大的边
    max_duplicate_value = max(duplicate_values.values())
    print(max_duplicate_value)
    max_duplicate_value_edges = [edge for edge in duplicate_values if duplicate_values[edge] == max_duplicate_value]
    print(max_duplicate_value_edges)
    # 如果存在多条重复值相等的边，则找到父结点编号减去子结点编号差值最大的那条边
    if len(max_duplicate_value_edges) > 1:
        max_edge = None
        max_difference = -1
        for edge in max_duplicate_value_edges:
            parent = max(edge)
            child = min(edge)
            difference = parent - child
            if difference > max_difference:
                max_difference = difference
                max_edge = edge
        return max_edge
    else:
        return max_duplicate_value_edges[0]

def draw_graph(G, circuits):
    # 绘制原始图像
    plt.subplot(1, 2, 1)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='black', arrows=True)
    nx.draw_networkx_edges(G, pos, edgelist=circuits, edge_color='red', arrows=True)
    plt.title("Original Graph")

    # 绘制处理后的图像
    plt.subplot(1, 2, 2)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='black', arrows=True)
    plt.title("Processed Graph")
    plt.show()

def process_graph(G):
    # 初始化边和环的编号
    edge_id = {edge: i for i, edge in enumerate(G.edges())}
    # circuit_id = {circuit: i for i, circuit in enumerate(nx.simple_cycles(G))}
    # 初始化循环计数器
    count = 1
    # 当图中存在环时，进行处理
    while nx.is_directed_acyclic_graph(G) == False:
        # 找到所有环，并为它们编号
        circuits = [circuit for circuit in nx.simple_cycles(G)]
        # circuit_id = {circuit: i for i, circuit in enumerate(circuits)}
        # 计算每条边的重复值
        duplicate_values = calculate_duplicate_values(G, circuits)
        # 找到重复值最大的边，并删除该边以及该边所在的环
        max_duplicate_value_edge = get_max_duplicate_value_edge(G, duplicate_values)
        G.remove_edge(*max_duplicate_value_edge)
        for circuit in circuits:
            if max_duplicate_value_edge in circuit:
                G.remove_edges_from(circuit)
        # 输出删除的边的edge_id
        print("Count {}: Edge {} deleted".format(count, edge_id[max_duplicate_value_edge]))
        count += 1

    # 输出处理后的图像和边的编号
    print("Final Edge IDs:")
    for edge in edge_id:
        print("{}: {}".format(edge_id[edge], edge))
    return G
        
def drawGraph(G):
    # 绘制有向图
    plt.figure(figsize=(5,5))
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos)
    plt.show()


if __name__ == '__main__':
    # 创建有向图
    df = pd.read_csv('../data/ASSIST/graph/K_Directed.txt', sep='\t',header=None, skiprows=1)
    G = nx.DiGraph()
    for i, row in df.iterrows():
        G.add_edge(row[1], row[0])

    # # 要的图
    # drawGraph(G)
    ## 测试数据
    # G = nx.DiGraph()
    # edges = [(1, 2), (2, 3), (2, 4), (3, 5), (4, 5), (5, 6), (6, 4)]
    # G.add_edges_from(edges)


    # 方法1：贪心算法
    newG1 = process_graph(G)
    # 写入文件
    count = 0
    data = {'from': [], 'to': []}
    for node in newG1:
        for child in newG1[node]:
            count += 1
            data['from'].append(node)
            data['to'].append(child)
    info = pd.DataFrame(data)
    print(count)
    csv_data = info.to_csv("../data/ASSIST/graph/K_Directed-liu.txt",sep='\t',index=None)
    

# 以上程序中，我们首先定义了一个函数`calculate_duplicate_values`，用于计算每条边的重复值。
# 具体来说，我们遍历所有环，对于每个环中的每条边，都将它的重复值加1。
# 最后，我们得到了每条边的重复值，存储在一个字典`duplicate_values`中。

# 接着，我们定义了一个函数`get_max_duplicate_value_edge`，用于找到重复值最大的边。
# 如果存在多条重复值相等的边，则找到父结点编号减去子结点编号差值最大的那条边。

#在`process_graph`函数中，我们首先为每条边和每个环分别编号。
# 然后，我们使用一个循环计数器，不断找到重复值最大的边，删除该边及该边所在的环，并输出删除的边的编号。
# 当图中不存在环时，程序结束，并输出处理后的图像和边的编号。

#最后，我们通过创建一个有向图来测试程序。运行程序后，程序会输出删除的边的edge_id，
# 并画出该有向图在处理之前和之后的图像。
