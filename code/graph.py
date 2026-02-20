from collections import deque
from itertools import combinations 
import random
import copy

import matplotlib.pyplot as plt

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
    nodes = [i for i in range(G.number_of_nodes())]
    subsets = power_set(nodes)
    min_cover = nodes
    for subset in subsets:
        if is_vertex_cover(G, subset):
            if len(subset) < len(min_cover):
                min_cover = subset
    return min_cover

#********** Experiments (our code) **************

#*************** BFS2 (our code) *********************
# BFS2

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


#*************** BFS3 (our code) *********************
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


#*************** is_connected (our code) *********************
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



def is_independent_set(G, I):
    for start in G.adj:
        for end in G.adj[start]:
            if start in I and end in I:
                return False
    return True


def MIS(G):
    nodes = [i for i in range(G.number_of_nodes())]
    subsets = power_set(nodes)

    max_independent = []

    for subset in subsets:
        if is_independent_set(G, subset):
            if len(subset) > len(max_independent):
                max_independent = subset

    return max_independent

    

#*************** create_random_graph (our code) *********************
# Return a graph with i nodes and j edges
def create_random_graph(i,j): 
    G = Graph(i) # Creates graph with i nodes


    # Calculate all possible pairs of nodes, ignore duplicates (u,v) , (v,u) 
    #  Will only include one of those pairs
    # Also ignores self loops 
    subset = list(combinations([k for k in range(i)],2)) 
                                    
    s = set() # Store pairs of edges (Used to avoid duplicates)

    k = 0 

    # Choose j random edges
    while k < j: 
        choice = random.choice(subset) # Choose random pair of edges 

        # Check if choice in s
        if choice not in s: 
            s.add(choice)
            G.add_edge(choice[0],choice[1])
            k += 1
    
    return G
    
def listOfRandomGraphs(m, e):
    graphs = []
    for i in range(m):
        graphs.append(create_random_graph(100,e))
    return graphs

def experiment1():
    m = 100 #number of graphs in each list
    maxEdges = 100
    graphs = [listOfRandomGraphs(m,x) for x in range(1,maxEdges)]
    probabilties = []
    tempSum = 0

    for i in graphs:
        for j in i:
            if has_cycle(j):
                tempSum += 1
        probabilties.append(tempSum/m)
        tempSum = 0

    plt.plot([x for x in range(1,maxEdges)], probabilties)
    plt.title("Probabiltity of Cycle vs Number of Edges")
    plt.xlabel("Number of edges")
    plt.ylabel("Probability of cycle")
    plt.show()
    return 0

def experiment2():
    m = 100 #number of graphs in each list
    minEdges = 125
    maxEdges = 500
    graphs = [listOfRandomGraphs(m,x) for x in range(minEdges,maxEdges)]
    probabilties = []
    tempSum = 0

    for i in graphs:
        for j in i:
            if is_connected(j):
                tempSum += 1
        probabilties.append(tempSum/m)
        tempSum = 0

    plt.plot([x for x in range(minEdges,maxEdges)], probabilties)
    plt.title("Probabiltity of Connection vs Number of Edges")
    plt.xlabel("Number of edges")
    plt.ylabel("Probability of Connection")
    plt.show()
    return 0

# Create m graphs with n nodes (For each varying edge size)
def experiment2(n,m): 

    #graphs = [create_random_graph(n,x) for x in range(n)]
    graphs = [[] for _ in range(m)]


    for i in range(m): 
        # Create m graphs 
        j = 0 
        num_edges = 0
        while j < m:
             graphs[i].append(create_random_graph(n,num_edges))
             num_edges = (num_edges + 10) % n 
             j += 1

    print(graphs)




#*************** approx1 (our code) *********************

# Find vertex with highest degree in G
# Defaults to 0 if all vertices have same number of edges
def highest_degree(G): 
    maxDegree = 0
    maxVertex = 0

    for vertex in G.adj:
        currentDegree = len(G.adj[vertex])

        if currentDegree > maxDegree: 
            maxDegree = currentDegree
            maxVertex = vertex
        
    return maxVertex

# Approximates the vertex cover using highest degree vertex
def approx1(G): 

    # Copy the graph
    graph = copy.deepcopy(G)

    # print(graph.adj)

    # Create empty set 
    C = set()

    # While C is not a vertex cover, repeat steps
    while not is_vertex_cover(graph,C): 
        largestVertex = highest_degree(graph) # Return vertex with largest degree in graph
        C.add(largestVertex)                  # Add largest vertex to set

        # print(C)

        graph.adj[largestVertex] = []        # Remove all adjacent edges to largest Vertex

    return C




# Approximates the vertex cover using random vertex
def approx2(G): 
    graph = copy.deepcopy(G) 
    C = set() 

    while not is_vertex_cover(graph,C):

        # Choose random vertex from graphs adjacency list
        rVertex = random.choice(list(graph.adj.keys())) 


        # Keep choosing new rVertex 
        while rVertex in C: 
            rVertex = random.choice(list(graph.adj.keys())) 

        C.add(rVertex)
    return C
    

# Approximates the vertex cover using random edge (u,v)
def approx3(G): 
    graph = copy.deepcopy(G) 
    C = set() 

    while not is_vertex_cover(graph,C):

        # Choose random vertex from graphs adjacency list 
        u = random.choice(list(graph.adj.keys()))   

        # Has at least one edge
        if len(graph.adjacent_nodes(u)) > 0: 
            # Choose random vertex from u's adjacency list
            v = random.choice(graph.adjacent_nodes(u))

        # No adjacent edges, so skip to next iteration
        else: 
            continue 

        # keep choosing new pairs while both are in C
        while u in C and v in C: 
            u = random.choice(list(graph.adj.keys()))

            if len(graph.adjacent_nodes(u)) > 0: 
                # Choose random vertex from u's adjacency list
                v = random.choice(graph.adjacent_nodes(u))

        graph.adj[u] = [] # Reset u's adjacency list 
        C.add(u)
        C.add(v)

    return C
        

#experiment1()
#experiment2()

# g = Graph(7)

# g.add_edge(0,1)
# g.add_edge(0,2)
# g.add_edge(1,3)
# g.add_edge(2,3)
# g.add_edge(2,4)
# g.add_edge(3,5)

g = create_random_graph(15,15)
a = MIS(g)
b = MVC(g)

print(a)
print(b)
print(len(a))
print(len(b))


        




G = create_random_graph(4,2) 

print(MVC(G))
print(approx1(G))
print(approx2(G))
print(approx3(G))


# u = random.choice(list(G.adj.keys()))

# print(G.adj)
# print(u)
# print(random.choice(G.adjacent_nodes(u)))
