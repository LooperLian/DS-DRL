from collections import defaultdict

import networkx as nx
import copy as cp


def k_shortest_paths(G, source, target, k=1, weight='weight'):
    # 有向图
    # G is a networkx graph.
    # source and target are the labels for the source and target of the path.
    # k is the amount of desired paths.
    # weight = 'weight' assumes a weighed graph. If this is undesired, use weight = None.


    A = [nx.dijkstra_path(G, source, target, weight='weight')]
    A_len = [sum([G[A[0][l]][A[0][l + 1]]['weight'] for l in range(len(A[0]) - 1)])]
    B = []

    for i in range(1, k):
        for j in range(0, len(A[-1]) - 1):
            Gcopy = cp.deepcopy(G)
            spurnode = A[-1][j]
            rootpath = A[-1][:j + 1]
            for path in A:
                if rootpath == path[0:j + 1]:  # and len(path) > j?
                    if Gcopy.has_edge(path[j], path[j + 1]):
                        Gcopy.remove_edge(path[j], path[j + 1])
                    if Gcopy.has_edge(path[j + 1], path[j]):
                        Gcopy.remove_edge(path[j + 1], path[j])
            for n in rootpath:
                if n != spurnode:
                    Gcopy.remove_node(n)
            try:
                spurpath = nx.dijkstra_path(Gcopy, spurnode, target, weight='weight')
                totalpath = rootpath + spurpath[1:]
                if totalpath not in B:
                    B += [totalpath]
            except nx.NetworkXNoPath:
                continue
        if len(B) == 0:
            break
        lenB = [sum([G[path[l]][path[l + 1]]['weight'] for l in range(len(path) - 1)]) for path in B]
        B = [p for _, p in sorted(zip(lenB, B))]
        A.append(B[0])
        A_len.append(sorted(lenB)[0])
        B.remove(B[0])

    return A, A_len

def calcualte_Candidate_Paths(linkmap, nodeNum, k): #计算K条路径
    G = nx.DiGraph()
    Candidate_Paths = defaultdict(
        lambda: defaultdict(lambda: defaultdict(
            lambda: None)))  # Candidate_Paths[i][j][k]:the k-th path from i to j
    for i in range(nodeNum):
        for j in range(nodeNum):
            if i != j:
                if linkmap[i+1][j+1] != None:
                    G.add_edge(i+1, j+1, length = 1, weight = linkmap[i+1][j+1][1])
    for i in range(nodeNum):
        for j in range(nodeNum):
            if i != j:
                paths = k_shortest_paths(G, i+1, j+1, k, "weight")
                for p in range(len(paths[0])):
                    Candidate_Paths[i+1][j+1][p] = paths[0][p]
    return Candidate_Paths

if __name__ == "__main__":
    # G = nx.DiGraph()
    # G.add_edge(1, 2, length=3, weight=1)
    # G.add_edge(2, 1, length=3, weight=1)
    #
    # G.add_edge(1, 6, length=2, weight=2)
    # G.add_edge(6, 1, length=2, weight=2)
    #
    # G.add_edge(2, 3, length=4, weight=3)
    # G.add_edge(3, 2, length=4, weight=3)
    #
    # G.add_edge(2, 6, length=1, weight=4)
    # G.add_edge(6, 2, length=1, weight=4)
    #
    # G.add_edge(3, 4, length=2, weight=5)
    # G.add_edge(4, 3, length=2, weight=5)
    #
    # G.add_edge(3,5, length=3, weight=6)
    # G.add_edge(5, 3, length=3, weight=6)
    #
    # G.add_edge(4, 5, length=2, weight=7)
    # G.add_edge(5, 4, length=2, weight=7)
    #
    # G.add_edge(5, 6, length=1, weight=8)
    # G.add_edge(6, 5, length=1, weight=8)
    #
    #
    # print(k_shortest_paths(G, 1, 5, 5, "weight")[0][1])
    # # ([[1, 6, 5], [1, 2, 3, 5], [1, 2, 6, 5], [1, 6, 2, 3, 5], [1, 2, 3, 4, 5]], [10, 10, 13, 15, 16])

    linkmap = defaultdict(lambda: defaultdict(lambda: None))  # Topology: NSFNet
    linkmap[1][2] = (0, 1050)  # 编号，长度
    linkmap[2][1] = (3, 1050)
    linkmap[1][3] = (1, 1500)
    linkmap[3][1] = (6, 1500)
    linkmap[1][8] = (2, 2400)
    linkmap[8][1] = (22, 2400)

    linkmap[2][3] = (4, 600)
    linkmap[3][2] = (7, 600)
    linkmap[2][4] = (5, 750)
    linkmap[4][2] = (9, 750)
    linkmap[3][6] = (8, 1800)
    linkmap[6][3] = (15, 1800)

    linkmap[4][5] = (10, 600)
    linkmap[5][4] = (12, 600)
    linkmap[4][11] = (11, 1950)
    linkmap[11][4] = (32, 1950)
    linkmap[5][6] = (13, 1200)
    linkmap[6][5] = (16, 1200)
    linkmap[5][7] = (14, 600)
    linkmap[7][5] = (19, 600)

    linkmap[6][10] = (17, 1050)
    linkmap[10][6] = (29, 1050)
    linkmap[6][14] = (18, 1800)
    linkmap[14][6] = (41, 1800)
    linkmap[7][8] = (20, 750)
    linkmap[8][7] = (23, 750)
    linkmap[7][10] = (21, 1350)
    linkmap[10][7] = (30, 1350)

    linkmap[8][9] = (24, 750)
    linkmap[9][8] = (25, 750)
    linkmap[9][10] = (26, 750)
    linkmap[10][9] = (31, 750)
    linkmap[9][12] = (27, 300)
    linkmap[12][9] = (35, 300)
    linkmap[9][13] = (28, 300)
    linkmap[13][9] = (38, 300)

    linkmap[11][12] = (33, 600)
    linkmap[12][11] = (36, 600)
    linkmap[11][13] = (34, 750)
    linkmap[13][11] = (39, 750)
    linkmap[12][14] = (37, 300)
    linkmap[14][12] = (42, 300)
    linkmap[13][14] = (40, 150)
    linkmap[14][13] = (43, 150)

    path = calcualte_Candidate_Paths(linkmap, 14, 10)
    p = path[5][7]
    print(p)
    for m in p.items():
        print(m)
    print(path[5][7][3])

