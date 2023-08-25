from math import factorial, cos, sin, tan, ceil, floor
from statistics import mean, median, mode
from matplotlib import pyplot as plt
from sys import stdin; read = lambda : stdin.readline()
from sys import setrecursionlimit; setrecursionlimit(10**6)
from heapq import heapify, heappop, heappush, heappushpop
import fileinput
from collections import defaultdict, deque

# Contribution to Sum Technique - To calculate the cumulative sum of all subarrays of a given array in O(n)
arr = [1,3,4,9,6,2,8,56,,6,5,12,5,6,25]; n = len(arr)
ans = 0
for i in range(n):
    ans += arr[i]*(n-i)*(i+1)
print(ans)

#Python program to compute number of divisors of all numbers up to n efficiently
def efficientDivisor(n):
    divisors = [0 for i in range(n+1)]
    for i in range(1, n+1):
        for j in range(2*i, n+1, i):
            divisors[j]+=1
    print(divisors)

#Graphs; it's traversals; it's operations
from collections import defaultdict as dd
class Graph:
    def __init__(self, graphStructure = None):
        if graphStructure == None:
            graphStructure = dd(lambda : [])
        self.graphStructure = graphStructure
    
    def addNode(self, vertex, edge):
        self.graphStructure[vertex].append(edge)
        #Uncomment the next line if using undirected graphs, sometimes it can help
        #self.graphStructure[edge].append(vertex)
    
    def bfs(self, vertex):
        visited = {vertex}
        queue = [vertex]

        while queue:
            curr = queue.pop(0)
            print(curr, end = " ")
            for adjacentVertex in self.graphStructure[curr]:
                if adjacentVertex not in visited:
                    visited.add(adjacentVertex)
                    queue.append(adjacentVertex)
    
    def dfs(self, vertex):
        visited = {vertex}
        stack = [vertex]

        while stack:
            curr = stack.pop()
            print(curr, end = " ")
            for adjacentVertex in self.graphStructure[curr]:
                if adjacentVertex not in visited:
                    visited.add(adjacentVertex)
                    stack.append(adjacentVertex)

    #------------------------------------------------------------------#
    
    def DFSforTotalConnectedComponents(self, node, visitedNodes):
        visitedNodes.add(node)
        for adjacentVertex in self.graphStructure[node]:
            if adjacentVertex not in visitedNodes:
               self.DFSforTotalConnectedComponents(adjacentVertex, visitedNodes)

    def totalConnectedComponents(self, graph):
        ans = 0; visitedNodes = set([])
        for node in graph.graphStructure:
            if node not in visitedNodes:
                self.DFSforTotalConnectedComponents(node, visitedNodes)
                ans+=1
        return ans

#Linked lists and it's operations
class llnode:
    def __init__(self, val):
        self.val = val
        self.next = None
    
class SinglyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None

#-----------------------------------------------------------------------------------------------#

#Basic 0-1 Knapsack, DP and Recursive approach (FOR MAX WEIGHT AND MAX COST/VALUE)
def knapsackDP(capacity, weights, values):

    dp = [0 for i in range(capacity+1)] 
    for i in range(len(weights)): 
        for w in range(capacity, weights[i]-1, -1):
            dp[w] = max( dp[w], dp[w-weights[i]]+values[i] )
 
    return dp[capacity]

def knapsack(capacity, weights, values, n):
    if n == 0 or capacity == 0: return 0

    if weights[n-1] > capacity: return knapsack(capacity, weights, values, n-1)

    else: return max( values[n-1] + knapsack(capacity - weights[n-1], weights, values, n-1), knapsack(capacity, weights, values, n-1))

#Basic 0-1 Knapsack, DP and Recursive approach (FOR MAX WEIGHT AND MIN COST/VALUE)
def knapsackDP(capacity, weight, cost):
    
    dp = [10**20 for i in range(capacity+1)]; dp[0] = 0        
    for i in range(len(weight)):        
        for w in range(capacity, weights[i]-1,-1):
            dp[w] = min(dp[w], dp[w-weight[i]] + cost[i])
    
    return dp[capacity]

def knapsack(capacity, weights, costs, n):
    if capacity == 0: return 0
    if n == 0: return 10**20

    if weights[n-1] > capacity: return knapsack(capacity, weights, costs, n-1)

    else: return min( costs[n-1] + knapsack(capacity - weights[n-1], weights, costs, n-1), knapsack(capacity, weights, costs, n-1))

#Knapsack in which one element can be selected multiple times (FOR MAX WEIGHT AND MIN COST/VALUE)
def minKnapsack(capacity, weights, cost):
    dp = [10**20 for i in range(capacity+1)]
    dp[0] = 0
    for i in range(len(weights)):
        for w in range(weights[i], capacity+1):
            dp[w] = min(dp[w], dp[w-weights[i]] + cost[i])
    
    return dp[capacity]
# Longest Common Subsequence
text1 = "abcde"
text2 = "ace"

dp = [[0 for i in range(len(text1)+1)] for j in range(len(text2)+1)]

for i in range(1, len(text2)+1):
    for j in range(1, len(text1)+1):
        if text2[i-1] != text1[j-1]:
            dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        else:
            dp[i][j] = dp[i-1][j-1]+1

print(dp[len(text2)][len(text1)])
# Longest Palindromic Subsequence
text1 = "abfbnshghgiwsnfjsjbi"
text2 = text1[::-1]

dp = [[0 for i in range(len(text1)+1)] for j in range(len(text2)+1)]

for i in range(1, len(text2)+1):
    for j in range(1, len(text1)+1):
        if text2[i-1]==text1[j-1]:
            dp[i][j] = dp[i-1][j-1]+1
        else:
            dp[i][j] = max(dp[i-1][j], dp[i][j-1])

print(dp[len(text2)][len(text1)])

#__________________________________________________#

#Turn a tree given by edge inputs into a proper tree [dictionary implementation]
def properTree(tree, root):
    visited = set([root])
    stack = [root]
    ans = dd(lambda : [])
    while stack:
        curr = stack.pop()

        for adj in tree[curr]:            
            if adj not in visited:
                ans[curr].append(adj)
                visited.add(adj)
                stack.append(adj)
    
    return ans

#Convert a sorted array into a BST
def sortedArrToBST(left, right):
    if left>right: return None

    mid = (left + right) // 2

    root = TreeNode(nums[mid])
    root.left = sortedArrToBST(left, mid-1)
    root.right = sortedArrToBST(mid+1, right)

    return root

#--------------------------------------------------------------------------------------------#
#Largest sum contiguous subarray - Kadane's Algo
def kadanesAlgo(arr):
    msf = -10**18
    meh = 0
    for i in arr:
        
        meh += i
        if i > meh:
            meh = i

        if meh>msf:
            msf = meh
    
    return msf
#----------------------------------------------------------------------------------------------#
#Dijkstra's Algo
from heapq import *
graph = {'A':[(5, 'D'), (7, 'B')], 'D':[(3, 'C')], 'B':[(8, 'E')], 'C':[(4, 'E')], 'E':[(5, 'F')] }

cost = {'A':0, 'B':10**10, 'C':10**10, 'D':10**10, 'E':10**10, 'F':10**10}

queue = [(0,'A')]; heapify(queue)

while queue:
    currprice, currnode = heappop(queue)
    if currnode not in graph: continue
    
    for adjprice, adjnode in graph[currnode]:
        if cost[adjnode] > currprice + adjprice:
            cost[adjnode] = currprice + adjprice
            heappush(queue, (currprice + adjprice, adjnode))

print(cost)

#OR

from heapq import *
graph = {0:[(100, 1), (500, 2)], 1:[(100, 2)]}

cost = {0:0, 1:10**10, 2:10**10}

queue = [(0,0)]; heapify(queue)

while queue:
    currprice, currnode = heappop(queue)
    if currnode not in graph: continue
    
    for adjprice, adjnode in graph[currnode]:
        if cost[adjnode] > currprice + adjprice:
            cost[adjnode] = currprice + adjprice
            heappush(queue, (currprice + adjprice, adjnode))

print(cost)

#---------------------------------------------------------------------------------------------#
#Topological Sort - Generic

numberOfNodes = 4
graph = {1:[0], 2:[0], 3:[1,2]}

visited = set([])
marked  = set([])

ans = []

def trav(node):
    #print(node)
    if node in marked: return True
    if node not in graph: ans.append(node); marked.add(node); visited.add(node); return True
    
    
    if node not in visited: visited.add(node)
    else: return False
    
    total = True
    for i in graph[node]:
        total = total and trav(i)
        if total == False: break
            
    if total == True: ans.append(node); marked.add(node); return True
    else: return False

for i in range(numberOfNodes):
    if i not in visited:
        if trav(i) == False: print([], "Cycle detected"); exit(0)

print(ans[::-1]); exit(0)

#----------------------------------------------------------------------------------------------#
#Bellman-Ford Algo
def bellmanFord(node, numberOfNodes, graph):
    cost = [10**10 if i!=node else 0 for i in range(numberOfNodes)]
    
    for _ in range(numberOfNodes-1):
        
        for nodes in graph:
            for price, adj in graph[nodes]:
                if cost[adj] > price + cost[nodes] :
                    cost[adj] = price + cost[nodes]
                
    return cost

#OR we can simply take the given list in the form [[from, to, price], [from, to, price].....]

def bellmanFord(node, numberOfNodes, arr):
    cost = [10**10 if i!=node else 0 for i in range(numberOfNodes+1)]
    
    for _ in range(numberOfNodes-1):                
        for frm, to, price in arr:
            if cost[to] > cost[frm] + price:
                cost[to] = cost[frm] + price
                
    return cost

#--------------------------------------------------------------------------------------------#

#Topological Sort - Kahn's Algo

graph = defaultdict(lambda : [])
indeg = dict([(i, 0) for i in range(1,numberOfNodes+1)])

#graph given in the format [[from, to], [from, to]]
for i in givenGraph: graph[i[0]].append(i[1]); indeg[i[1]]+=1

def kahn(graph, indeg, numberOfNodes):

    queue = [i for i in indeg if indeg[i]==0]

    ans = []; visited = set([])
    while queue:
        
        for i in range(len(queue)):
            curr = queue.pop(0)
            
            visited.add(curr)
            ans.append(curr)

            if curr not in graph: continue
                
            for adj in graph[curr]:
                indeg[adj]-=1
                if indeg[adj] == 0: queue.append(adj)

    return ans if len(visited) == numberOfNodes else None

#--------------------------------------------------------------------------------------#
#Minimum-Sum Difference DP 
def minsumdiff(arr):
    dp = [0 for i in range(sum(arr)//2 + 1)]

    for nums in arr:
        for w in range(sum(arr)//2 , nums-1, -1):
            dp[w] = max(dp[w], dp[w - nums] + nums)

    return sum(arr) - max(dp) - max(dp)
    #return sum(arr) - dp[sum(arr)//2]*2
#--------------------------------------------------------------------------------------#
#Hierholzer's Algorithm for Eularian Circuit - Directed Graph
def hierholzer(node, ans):
    if node not in graph or not graph[node]: ans.append(node)
    else:
        while graph[node]:
            hierholzer(graph[node].pop(0), ans)
        ans.append(node)
    return reversed(ans)

#OR

ans = []
def hierholzer(node):

    if node not in graph or not graph[node]: ans.append(node); return ""
        
    while graph[node]:
        hierholzer(graph[node].pop(0))
    ans.append(node)
#hierholzer(node); ans.reverse(); return ans
#-------------------------------------------------------------------------------------#
# Bipartite Graph or not

def bipartite(node):
    visited.add(node)
    queue = [node]
    
    while queue:
        curr = queue.pop(0)
        
        for adj in graph[curr]:
            if color[adj] == -1: color[adj] = abs(color[curr] - 1); queue.append(adj); visited.add(adj)
            elif color[adj] != color[curr]: pass
            else: return -1
        s
visited = set([])
color = [-1]*len(graph)

for num in range(len(graph)):
    if num not in visited:
        color[num] = 0
        if bipartite(num) == -1: print(False); exit(0)

print(True); exit(0)

# ==========================================================================================#

# Trie and it's basic functions

class Trie:
    def __init__(self):
        self.structure = {}
        
    def addWord(self, word):
        
        curr = self.structure
        for letter in word:
            
            if letter not in curr:
                curr[letter] = {}
            
            curr = curr[letter]
        
        curr['end'] = 1
        
    def getWordsIterative(self):
        
        root = self.structure
        stack = [(root, '')]
        
        while stack:
            
            currRoot, currWord = stack.pop()
            if 'end' in currRoot: print(currWord)
                
            for letter in currRoot:
                if letter != 'end':
                    stack.append((currRoot[letter], currWord+letter))  
    
    
    def getWordsRecursive(self, root, word):
        
        if 'end' in root: 
            print(word)
        
        for letter in root:
            if letter != 'end':
                self.getWordsRecursive(root[letter], word+letter)
        
        
trie = Trie()

trie.addWord('aeroplane')
trie.addWord('aerosmith')
trie.addWord('aether')
trie.addWord('him')
trie.addWord('himali')
trie.addWord('himalaya')

print(f'\nIterative will give you a wierd order even idk:')
trie.getWordsIterative()
print(f'\nRecursive will give you the exact order of insertion:')
trie.getWordsRecursive(trie.structure, '')

print('\nWe can even get lexicographic order with a few tweaks in the iterative code. That is for you to figure out 😉.')
