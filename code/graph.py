from collections import deque

#Undirected graph using an adjacency list
class Graph:

    def __init__(self, n):
        self.adj = {}
        for i in range(1,n+1):
            self.adj[i] = []

    def are_connected(self, node1, node2):
        return node2 in self.adj[node1]

    def adjacent_nodes(self, node):
        return self.adj[node]

    def add_node(self):
        self.adj[len(self.adj)] = []

    def add_edge(self, node1, node2):
        if node1 not in self.adj[node2]:
            self.adj[node1].append(node2)
            self.adj[node2].append(node1)

    def number_of_nodes():
        return len()


#Breadth First Search
def BFS(G, node1, node2):
    Q = deque([node1])
    marked = {node1 : True}
    for node in G.adj:
        if node != node1:
            marked[node] = False
    while len(Q) != 0:
        current_node = Q.popleft()
        for node in G.adj[current_node]:
            if node == node2:
                return True
            if not marked[node]:
                Q.append(node)
                marked[node] = True
    return False


#Depth First Search
def DFS(G, node1, node2):
    S = [node1]
    marked = {}
    for node in G.adj:
        marked[node] = False
    while len(S) != 0:
        current_node = S.pop()
        if not marked[current_node]:
            marked[current_node] = True
            for node in G.adj[current_node]:
                if node == node2:
                    return True
                S.append(node)
    return False

#*************** DFS2 (our code) *********************

def DFS2(G, node1, node2):
    P = {}               
    S = [node1]           
    marked = {}
    for node in G.adj:
        marked[node] = False
    marked[node1] = True
    P[node1] = None
    while len(S) != 0:
        current_node = S.pop()

        if current_node == node2:
            path = []
            cur = node2
            while cur is not None:
                path.append(cur)
                cur = P[cur]
            path.reverse()
            return path

        for node in G.adj[current_node]:
            if not marked[node]:
                marked[node] = True
                P[node] = current_node
                S.append(node)

    return None


#*************** DFS3 (our code) *********************

def DFS3(G, node1):
    P = {}               
    S = [node1]           
    marked = {}
    for node in G.adj:
        marked[node] = False
    marked[node1] = True
    while len(S) != 0:
        current_node = S.pop()
        for node in G.adj[current_node]:
            if not marked[node]:
                marked[node] = True
                P[node] = current_node
                S.append(node)

    return P

#************ has cycle (our code) **********************
def has_cycle(G):
    marked = {}
    parent = {}
    for node in G.adj:
        marked[node] = False
        parent[node] = None
    for node in G.adj:
        if not marked[node]:
            S = [node]
            marked[node] = True
            parent[node] = None
            while len(S) != 0:
                current_node = S.pop()
                for node in G.adj[current_node]:
                    if not marked[node]:
                        marked[node] = True
                        parent[node] = current_node
                        S.append(node)
                    elif parent[current_node] != node:
                        return True

    return False


#Use the methods below to determine minimum vertex covers

def add_to_each(sets, element):
    copy = sets.copy()
    for set in copy:
        set.append(element)
    return copy

def power_set(set):
    if set == []:
        return [[]]
    return power_set(set[1:]) + add_to_each(power_set(set[1:]), set[0])

def is_vertex_cover(G, C):
    for start in G.adj:
        for end in G.adj[start]:
            if not(start in C or end in C):
                return False
    return True

def MVC(G):
    nodes = [i for i in range(G.get_size())]
    subsets = power_set(nodes)
    min_cover = nodes
    for subset in subsets:
        if is_vertex_cover(G, subset):
            if len(subset) < len(min_cover):
                min_cover = subset
    return min_cover

#********** Experiments (our code) **************

g = Graph(6)

g.add_edge(1,2)
g.add_edge(1,3)
g.add_edge(2,4)
g.add_edge(3,4)
g.add_edge(3,5)
g.add_edge(5,4)
g.add_edge(4,6)

print(DFS2(g,1,6))
print(DFS3(g,1))

print(has_cycle(g))
