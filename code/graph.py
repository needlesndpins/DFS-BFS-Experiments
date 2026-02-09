from collections import deque

#Undirected graph using an adjacency list
class Graph:

    def __init__(self, n):
        self.adj = {}
        for i in range(0,n):
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

    def number_of_nodes(self):
        return len(self.adj)


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



# * Our Code *

# BFS2/DFS2 


# Do we assume node1 != node2 ??
def BFS2(G,node1, node2):

    # Initialize parent array with -1 for every entry
    p = [-1 for i in range(len(G.adj))] # Parent array, p[i] = cj -> vertex i is parent of vertex j
    path = []

    Q = deque([node1]) # stores vertices to visit
    marked = {node1 : True} # Dictionary to store if vertex visited


    # Fill up the marked dictionary with False for every vertex except node1
    for node in G.adj:
        if node != node1:
            marked[node] = False



    while len(Q) != 0:
        current_node = Q.popleft() # Pop next vertex to check

        for node in G.adj[current_node]: # Go through this vertice's adjacency list

            if node == node2: # If connection with node2, you're done 
                p[node] = current_node
                n = node2 

                # Go back through parent array to find path
                while n != node1: 
                    print(n)
                    print(p[n])
                    path.append(n)
                    n = p[n]
                path.append(node1)

                path.reverse()
                #print(f"path:{path}")
                return path
                
            

            if not marked[node]: # If vertex isn't marked
                p[node] = current_node # parent of adjacent node (node) is node currently visiting (current_node)
                Q.append(node)   # Add to deque 
                marked[node] = True # Mark it 

    return path


# Return predecessor dictionary of form: {2 : 1, 3 : 1, 4 : 2, 5 : 3, 6 : 4}
def BFS3(G,node1): 
    Q = deque([node1])
    marked = {node1 : True}

    dic = {} 

    for node in G.adj:
        if node != node1:
            marked[node] = False

    while len(Q) != 0:

        current_node = Q.popleft()

        for node in G.adj[current_node]:
            

            if not marked[node]:
                Q.append(node)
                marked[node] = True 

                # This check is probably not needed
                if node not in dic: 
                    dic[node] = current_node


    return dic


# Choose a vertex, and check if you visit every vertex
# Assume all Graphs have vertex 0 (Should cause of how Graphs are defined in this file)
def is_connected(G): 
    num_vertices = len(G.adj)

    discovered = []

    Q = deque([0]) # Start at 0
    discovered.append(0)

    marked = {0 : True}

    for node in G.adj:
        if node != 0:
            marked[node] = False
    
    while len(Q) != 0:
        current_node = Q.popleft()

        for node in G.adj[current_node]:

            if not marked[node]:
                discovered.append(node)
                Q.append(node)
                marked[node] = True


    return len(discovered) == num_vertices
    




g = Graph(6)


g.add_edge(0,1)
g.add_edge(0,2)
g.add_edge(1,3)
g.add_edge(2,3)
g.add_edge(2,4)
g.add_edge(3,5)



#BFS2(g,0,4)


# 0 - 1 - 2    3 
# |-------|

g2 = Graph(4) 
g2.add_edge(0,1)
g2.add_edge(1,2)
g2.add_edge(2,0)

print(BFS2(g,0,5))
print(DFS2(g,0,5))
print(BFS3(g,0))
print(DFS3(g,0))
print(is_connected(g))
print(has_cycle(g))
