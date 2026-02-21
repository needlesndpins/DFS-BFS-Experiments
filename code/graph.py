from collections import deque
from itertools import combinations 
import random
import copy
import math
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
    nodes = [i for i in range(len(G.adj))]
    subsets = power_set(nodes)
    min_cover = nodes
    for subset in subsets:
        if is_vertex_cover(G, subset):
            if len(subset) < len(min_cover):
                min_cover = subset
    return min_cover

#********** Experiments (our code) **************

#*************** BFS2 (our code) *********************
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
    

#*************** create_random_graph (our code) *********************
# Return a graph with i nodes and j edges
# Limited to i CHOOSE 2 edges; since unique edges
def create_random_graph(i,j): 
    G = Graph(i) # Creates graph with i nodes

    # More edges than possible connections in graph
    # Return empty graph
    if j > math.comb(i,2): 
        return G


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
# Defaults to 0 if all vertices have no edges
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
        



# Generate 1000 graphs with 7 different edge sizes 
# Compute MVC for each graph 
# 2 things to check
#   1) Check if Approx returns MVC (Count how many are minimum)
#   2) Calculate size of approximation, and compare to MVC 
#      - Sum # all edges in MVC to get total_sum
#      - Sum # all edges in approx to get aprox_sum
#   3) Calculate expected performance: approx_sum / total_sum
#   Do this for each different number of edges, then graph the results
def part2_edge():
        
    num_edges = [1,5,10,15,20,25,28] # 28 is max amount of unique edges for graph with 8 nodes
    num_graphs = 1000         # Num graphs for each edge count

    # Dictionary of graphs, where graphs[i] stores array of graphs with 'i' edges
    graphs = {1: [], 5: [], 10: [], 15: [], 20: [], 25:[], 28:[]} 

    # For each edge ampunt
    for edge in num_edges: 
        # Create num_graphs 
        for _ in range(num_graphs):
            temp = create_random_graph(8,edge)
            # print(f"{j}, {edge}: hit here")

            graphs[edge].append(temp) # append to array


    mvcSum = {1: 0, 5: 0, 10: 0, 15: 0, 20: 0, 25: 0, 28: 0} # Store sum of MVC for each num of edges

    # Keys represent number of edges
    # Values are arrays: 
    #  [Sum of all approx vertex cover, Count number of minimum vertex covers]
    a1Sum = {1: [0,0], 5: [0,0], 10: [0,0], 15: [0,0], 
             20: [0,0], 25: [0,0], 28: [0,0]}
    a2Sum = {1: [0,0], 5: [0,0], 10: [0,0], 15: [0,0], 
             20: [0,0], 25: [0,0], 28: [0,0]}
    a3Sum = {1: [0,0], 5: [0,0], 10: [0,0], 15: [0,0], 
             20: [0,0], 25: [0,0], 28: [0,0]}
    
    
    # Store expected performances
    a1Perf = []
    a2Perf = []
    a3Perf = []
    

    # Compute MVC, approx1, approx2, approx3 

    # For each number of edges
    for edge in graphs.keys():

        # Compute MCV, approx1, approx2, approx3 for each of the graphs 
        for i in range(num_graphs):
            g = copy.deepcopy(graphs[edge][i])



            # Feed copy of graph, avoid issues
            minCoverSize = len(MVC(g)) # Store size of minimum vertex cover

            # Don't need to feed copy since approx copies it 
            approx1Size = len(approx1(graphs[edge][i]))  # Store size of approximation vertex cover
            approx2Size = len(approx2(graphs[edge][i]))
            approx3Size = len(approx3(graphs[edge][i]))


            # Add to size 
            mvcSum[edge] += minCoverSize 
            a1Sum[edge][0] += approx1Size # [key][0] stores sum of all approx vertex cover sizes
            a2Sum[edge][0] += approx2Size
            a3Sum[edge][0] += approx3Size

            # Check if approximations are minimum vertex covers
            if(approx1Size == minCoverSize): 
                a1Sum[edge][1] += 1           # [key][1] stores number of approximations are minimum

            if(approx2Size == minCoverSize): 
                a2Sum[edge][1] += 1

            if(approx3Size == minCoverSize): 
                a3Sum[edge][1] += 1

        # Compute expected performanve for each edge count
        a1Perf.append(a1Sum[edge][0] / mvcSum[edge])
        a2Perf.append(a2Sum[edge][0] / mvcSum[edge])
        a3Perf.append(a3Sum[edge][0] / mvcSum[edge])

    print(f"approx1: {a1Sum}")
    print(f"approx2: {a2Sum}")
    print(f"approx3: {a3Sum}")



    plt.plot(num_edges,a1Perf, label = "Approximation 1")
    plt.plot(num_edges,a2Perf, label = "Approximation 2")
    plt.plot(num_edges,a3Perf, label = "Approximation 3")

    plt.xlabel("Number of edges")
    plt.ylabel("Expected performance (Approx sum / Min cover sum)")
    plt.title("Expected performance for 8 nodes and varied edges")
    plt.legend()

    plt.show()

# Compute expected performance of approximations for differen number of nodes
#   approx_sum / total_sum
# Takes a boolean argument
#  True -> Fix number of edges (6)
#  False -> Proportional number of edges (node CHOOSE 2 / 2, half max amount of edges)
def part2_node(fixed): 

    # Use a dictionary for index mapping 
    graphs = {}
    num_nodes = [4,5,6,7,8,9,10,11]
    num_graphs = 1000

    num_edges = 6

    # Create 1000 graphs with different number of nodes (5 - 10) 
    for i in num_nodes:
        
        if not fixed: 
            num_edges = math.comb(i,2) // 2  # Half max amount of edges

        graphs[i] = [] # Initialize spot i as an array

        for j in range(num_graphs):
            temp = create_random_graph(i,num_edges)
            graphs[i].append(temp)  # Append to position i
    

    a1Perf = []
    a2Perf = []
    a3Perf = []


    for node in graphs.keys(): 

        mvcSum = 0
        a1Sum = 0
        a2Sum = 0
        a3Sum = 0
        
        for i in range(num_graphs): 
            g = copy.deepcopy(graphs[node][i]) # Copy graph to feed to MVC (Avoid issues)



            # Feed copy of graph, avoid issues
            minCoverSize = len(MVC(g)) # Store size of minimum vertex cover

            # Don't need to feed copy since approx copies it 
            approx1Size = len(approx1(graphs[node][i]))  # Store size of approximation vertex cover
            approx2Size = len(approx2(graphs[node][i]))
            approx3Size = len(approx3(graphs[node][i]))


            # Add to size 
            mvcSum += minCoverSize 
            a1Sum += approx1Size # [key][0] stores sum of all approx vertex cover sizes
            a2Sum += approx2Size
            a3Sum += approx3Size

        # Compute expected performance
        a1Perf.append(a1Sum/mvcSum)
        a2Perf.append(a2Sum/mvcSum)
        a3Perf.append(a3Sum/mvcSum)


    plt.plot(num_nodes,a1Perf, label = "Approximation 1")
    plt.plot(num_nodes,a2Perf, label = "Approximation 2")
    plt.plot(num_nodes,a3Perf, label = "Approximation 3")

    plt.xlabel("Number of nodes")
    plt.ylabel("Expected performance (Approx sum / Min cover sum)")
    plt.title("Expected performance for varied number of nodes")
    plt.legend()

    plt.show()



    return


# Receives a set of tuples of edges 
# edges = ( (n1,n2), (n3,n4), ...)
def create_graph(edges): 
    g = Graph(5)

    if len(edges) <= 0:
        return g 
    

    for pair in edges:

        # Invalid size
        if len(pair) != 2: 
            continue

        g.add_edge(pair[0],pair[1])

    return g


# Generate all graphs of size 5
# Max edges: (5 CHOOSE 2) = 10 
# 2^10 = 1024 possible graphs 
# Consider all possible edges
#  (0,1), 
def part2_worst_case():

    # Get all subsets of this for possible edge combinations
    pEdges = list(combinations([k for k in range(5)],2))  # Store all possible edges

    graphs = []
    
    subsets = []

    # Generate all possible combinatiosn of pEdges for sizes 0 .. len(pEdges
    # Computes all possible subsets of possible edges, which computes all possible graphs
    for i in range(len(pEdges) + 1):
        subsets.extend(list(combinations(pEdges,i)))
    

    a1Perf = []
    # For every set in subsets
    for set in subsets: 
        g = create_graph(set) # Create a graph for each subset of edges
        g2 = copy.deepcopy(g)  # Create copy
        # graphs.append(create_graph(set)) # Create a graph for each subset 

        mvcSum = len(MVC(g2))
        a1Sum = len(approx1(g))

        if mvcSum == 0: 
            a1Perf.append(1)
        else:
            if(a1Sum/mvcSum > 1.8):
                print(set)
            a1Perf.append(a1Sum/mvcSum)
            

    print(len(a1Perf))
    
    plt.plot([i for i in range(1,len(subsets) + 1)],a1Perf, label = "Approximation 1")

    plt.xlabel("Graph ID")
    plt.ylabel("Expected performance (Approx sum / Min cover sum)")
    plt.title("Expected performance for all graphs with 5 nodes")
    plt.legend()

    plt.show()






    
    return


# Run experiments 
# part2_edge()
# part2_node(True)  # True -> Fixed number of edges
# part2_node(False) # False -> Not fixed, proportional number of edges


part2_worst_case()



# pEdges = list(combinations([k for k in range(5)],2))  # Store all possible edges

# subsets = []
#     # Generate all possible combinatiosn of pEdges for sizes 0 .. len(pEdges
#     # Computes all possible subsets of possible edges, which computes all possible graphs
# for i in range(len(pEdges) + 1):
#     subsets.extend(list(combinations(pEdges,i)))

# g = create_graph(subsets[1023])
# print(g.adj)
# print(len(subsets[1023][0]))
# # Example
# lst = [1, 2, 3]
# print(all_subsets(lst))


# print(list(combinations([k for k in range(10)],2)))