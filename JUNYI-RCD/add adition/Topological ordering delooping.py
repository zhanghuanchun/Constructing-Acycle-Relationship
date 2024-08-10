# -*- coding: utf-8 -*-
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def drawGraph(G):
    # 绘制有向图
    plt.figure(figsize=(5,5))
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos)
    plt.show()

def find_cycles(G):
    """
    在给定的有向图中查找指定数量的环结构，并返回一个包含这些环的列表。
    G 是一个 NetworkX 的 DiGraph 对象。
    num_cycles 是要找到的环的数量。
    """
    cycles = []
    cycle_count = 0

    def dfs(node, visited, path):
        """
        DFS 遍历图，查找所有环结构。
        node 是当前正在访问的节点。
        visited 是一个集合，包含所有已经访问过的节点。
        path 是一个列表，包含了当前的路径。
        """
        nonlocal cycle_count

        visited.add(node)
        path.append(node)

        for neighbor in G.successors(node):
            if neighbor in path:
                # 找到了一个环，把它加入到结果中。
                start = path.index(neighbor)
                cycle = path[start:]
                # cycle.append(neighbor)
                cycles.append(cycle)
                cycle_count += 1
                print("=========cycle===========")
                print(cycle)
            elif neighbor not in visited:
                #递归地访问邻居节点。
                dfs(neighbor, visited, path)
        # path.pop()
        # visited.remove(node)

    # 对于每个未访问的节点，都进行一次 DFS 遍历。
    visited = set()
    for node in G.nodes:
        if node not in visited:
            dfs(node, visited, [])
    
    print(cycle_count) 
    return cycles


# 读取csv文件，构建图
# 创建有向图
df = pd.read_csv('../data/junyi/graph/K_Directed_raw.txt', sep='\t',header=None, skiprows=1)
G = nx.DiGraph()
for i, row in df.iterrows():
    G.add_edge(row[1], row[0])


# 要的图
# drawGraph(G)

# 生成了一个有向图，DiGraph with 93 nodes and 913 edges
# print(G)


# 后面的参数小于零表示找出所有，但是凭这数据量，电脑跑不完，耗时非常长
circles = find_cycles(G)

#number of cyclic 
#assist 1048
#junyi 113

