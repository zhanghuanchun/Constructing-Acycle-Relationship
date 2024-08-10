import collections
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

def remove_cycles(graph):
    """
    
    通过以最低准确度迭代删除边来删除有向图中的环，直到删除所有环。

    参数：
        图（字典）：有向图的字典表示形式：字典的键是结点，值是它们连接到的结点的列表。
        返回：无环图的字典表示形式：格式与输入图形相同。
    """
    # Calculate the in-degree and out-degree of each node
    # 计算每个结点的入度和出度
    outdegrees = dict((u, 0) for u in graph)
    indegrees = dict((u, 0) for u in graph)
    for u in graph:
        outdegrees[u] = len(graph[u])
    for u in graph:
        for v in graph[u]:
            indegrees[v] += 1


    # Find all cycles using Tarjan's algorithm
    # 使用 Tarjan 算法查找所有环
    stack = []
    on_stack = set()
    index = {node: None for node in graph}
    lowlink = {node: None for node in graph}
    components = []
    i = 0
    def strongconnect(node):
        nonlocal i
        index[node] = i
        lowlink[node] = i
        i += 1
        stack.append(node)
        on_stack.add(node)

        for child in graph[node]:
            if index[child] is None:
                strongconnect(child)
                lowlink[node] = min(lowlink[node], lowlink[child])
            elif child in on_stack:
                lowlink[node] = min(lowlink[node], index[child])

        if lowlink[node] == index[node]:
            component = []
            while True:
                child = stack.pop()
                on_stack.remove(child)
                component.append(child)
                if child == node:
                    break
            components.append(component)

    for node in graph:
        if index[node] is None:
            strongconnect(node)

    # Find the lowest-accuracy edge in each cycle and remove it
    # 找到每个环中准确度最低的边并将其移除
    new_graph = collections.defaultdict(list)
    for component in components:
        accuracies = []
        for node in component:
            accuracy = outdegrees[node] - indegrees[node]
            accuracies.append((accuracy, node))
        accuracies.sort(reverse=True)
        lowest_accuracy, lowest_node = accuracies[-1]
        for node in component:
            for child in graph[node]:
                if child != lowest_node:
                    new_graph[node].append(child)

    # Add back any edges that were not part of a cycle
    # # 重新添加不属于环的边
    # count1 = 0
    # count2 = 0
    for node in graph:
    #     count1+=1
        if node in new_graph:
    #         count2+=1
    #         print("node:",node)
    #         print("graph[node]:",graph[node])
    #         print("new_graph[node]:",new_graph[node])
            for child in graph[node]:
                if child not in new_graph[node]:
    #                 print("child:",child)
                    new_graph[node].append(child)

    # print("count1:",count1)
    # print("count2:",count2)
                    
    return new_graph



   

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
    df = pd.read_csv('../data/ASSIST/graph/K_Directed-raw.txt', sep='\t',header=None, skiprows=1)
    G = nx.DiGraph()
    for i, row in df.iterrows():
        G.add_edge(row[1], row[0])

    # 要的图
    # drawGraph(G)

    # 生成了一个有向图，DiGraph with 93 nodes and 913 edges
    # print(G)
        
    # 测试数据
    # G = nx.DiGraph()
    # edges = [(1, 2), (2, 3), (2, 4), (3, 5), (4, 5), (5, 6), (6, 4)]
    # G.add_edges_from(edges)
        
    # 方法2：准确度算法
    new_graph = remove_cycles(G)

    # 写入文件
    data = {'from': [], 'to': []}
    count = 0
    for from_node,to_nodes in new_graph.items():
        for to_node in to_nodes:
            count += 1
            data['from'].append(from_node)
            data['to'].append(to_node)
    print(count)
    info = pd.DataFrame(data)
    csv_data = info.to_csv("../data/ASSIST/graph/K_Directed-liu2.txt",sep='\t',index=None)
    
"""    
    该代码使用Tarjan算法查找图中的所有环，然后对每个环进行处理以删除一个边，直到没有环为止。
    Tarjan算法的复杂度为O(V+E)，其中V是节点数，E是边数。
    在每个环上，该代码计算每个节点的基础性并确定哪个边的准确性最低。
    基础性表示该结点对整个图的贡献。
    对于每个结点，基础性等于其出度减去其入度。
    对于每条边，准确性等于其父结点的基础性减去其子结点的基础性。
    代码删除具有最低准确性的边，并在最后返回不含有环的图是指存在一条路径，
    从某个节点出发可以回到该节点，而这个路径中的节点不重复。
    环可以使图中的一些算法失效或者产生意外的结果。因此，去除图中的环是一个很重要的任务。

# 算法的步骤如下：

    遍历图中的每个节点，计算其出度和入度。出度表示从该节点出发指向其他节点的边的数量，
    入度表示指向该节点的边的数量。将这些值保存在字典变量indegrees和outdegrees中。
    使用Tarjan算法查找图中的所有环。该算法基于深度优先搜索，并可以线性地找到所有强连通分量，
    其中每个分量都包含至少一个环。
    对于每个强连通分量（也就是每个环），计算其每个节点的基础性。基础性表示该节点对整个图的贡献。
    对于每个节点，基础性等于其出度减去其入度。然后，确定具有最低准确性的边并将其删除。
    准确性表示该边在环中的位置，越靠后准确性越低。
    对于每条边，准确性等于其父节点的基础性减去其子节点的基础性。
    因此，对于每个节点，需要先计算其基础性，然后根据该节点的基础性对其出边进行排序，从而确定最低准确性的边。
    删除环中的边后，将剩余的边添加到新的图中。最后返回这个新图。
    这个算法的时间复杂度取决于Tarjan算法的复杂度，其复杂度为O(V+E)，其中V是节点数，E是边数。
"""