'''
Author: Albert Go
UNI: ag4474
COMS 4701: Homework 1
'''
import numpy as np

def p1(k: int) -> str:
	factorials = []
	for i in range(k):
		fact_i = helper_p1(k-i)
		factorials.append(str(fact_i))
	return factorials

def helper_p1(i: int):
	if i == 1:
		return i
	else:
		return i*helper_p1(i-1)


def p2_a(x: list, y: list) -> list:
    y2 = []
    y = helper_p2_a(y, y2)
    return y

def helper_p2_a(y1: list, y2: list):
    '''
    Recursive function that takes in two lists. The first list is parsed through to find the max value and is added to the second list.

    y1: type = list; the list that needs to be sorted in descending order and gets update
    y2: type = list; new list that ends up being y1 but sorted in descending order

    return y1 list as the updated version or recurses 
    '''
    if not y1:
        y1 = y2
        return y1
    else:
        max_num = 0
        index = None
        for i in range(len(y1)):
            if y1[i] > max_num:
                max_num = y1[i]
                index = i
        y2.append(max_num)
        y1.pop(index)
        return helper_p2_a(y1, y2)

def p2_b(x: list, y: list) -> list:
    x2 = []
    for i in range(len(x)):
        temp = x[len(x)-1-i]
        x2.append(temp)
    x = x2
    return x


def p2_c(x: list, y: list) -> list:
    combined = []
    for i in range(len(x)):
        if x[i] in y:
            place = y.index(x[i])
            y.pop(place)
        combined.append(x[i])
    combined = combined+y

    temp = []
    combined = helper_p2_c(combined, temp)
    return combined

def helper_p2_c(combined: list, temp: list):
    '''
    Recursive function that takes in two lists. The first list is parsed through to find the min value and is added to the second list.

    combined: type = list; the list that needs to be sorted in ascending order and gets updated
    temp: type = list; new list that ends up being combined but sorted in ascending order

    return combined list as the updated version or recurses 
    '''
    if not combined:
        combined = temp
        return combined
    else:
        min_num = 10000000
        index = None
        for i in range(len(combined)):
            if combined[i] < min_num:
                min_num = combined[i]
                index = i
        temp.append(min_num)
        combined.pop(index)
        return helper_p2_c(combined, temp)

def p2_d(x: list, y: list) -> list:
    new_list = []
    new_list.append(x)
    new_list.append(y)
    return new_list

def p3_a(x: set, y: set, z: set) -> set:
    union = set()
    for i in x:
        union.add(i)
    for j in y:
        union.add(j)
    for k in z:
        union.add(k)
    return union

def p3_b(x: set, y: set, z: set) -> set:
    intersection = set()
    for i in x:
        if (i in y) and (i in z):
            intersection.add(i)
    return intersection

def p3_c(x: set, y: set, z: set) -> set:
    unique = set()
    for i in x:
        if (i not in y) and (i not in z):
            unique.add(i)
    for j in y:
        if (j not in x) and (j not in z):
            unique.add(j)
    for k in z:
        if (k not in y) and (k not in x):
            unique.add(k)
    return unique

def p4_a() -> np.array:
    board = np.array([[1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 2, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]])
    return board


def p4_b(x: np.array) -> list:
    length, width = x.shape
    pawn_pos = None
    attacking_knights = []

    for i in range(length):
        for j in range(width):
            if x[i][j] == 2:
                pawn_pos = (i, j)

    coord_diff = []
    for i in range(length):
        for j in range(width):
            if x[i][j] == 1:
                coord_diff = (abs(i-pawn_pos[0]), abs(j-pawn_pos[1]))
                if (coord_diff == (2,1)) or (coord_diff == (1,2)):
                    attacking_knights.append((i, j))
    return attacking_knights


def p5_a(x: dict) -> int:
    num_isolated = 0
    for key in x:
        if not x.get(key):
            num_isolated += 1
    return num_isolated


def p5_b(x: dict) -> int:
    num_nonisolated = 0
    for key in x:
        if x.get(key):
            num_nonisolated += 1
    return num_nonisolated


def p5_c(x: dict) -> list:
    edges = []
    for key, value in x.items():
        for i in value:
            if ((key, i) not in edges) and ((i, key) not in edges):
                edges.append((key, i))
    return edges


def p5_d(x: dict) -> np.array:
    matrix_graph = np.zeros((7,7))
    index_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
    edges = p5_c(x) #we solved for all distinct edges in problem 5c; use this to our advantage
    for edge in edges:
        i = index_dict.get(edge[0])
        j = index_dict.get(edge[1])
        #you have to add values for (i, j) and (j, i) because an edge goes from A to D but also D to A
        matrix_graph[i][j] = 1
        matrix_graph[j][i] = 1
    return matrix_graph

class PriorityQueue(object):
    def __init__(self):
        self.priorty_dict = {'apple': 5.0, 'banana': 4.5, 'carrot': 3.3, 'kiwi': 7.4, 'orange': 5.0, 'mango': 9.1, 'pineapple': 9.1}
        self.queue = []

    def push(self, x):
        return self.queue.append(x) #add element to queue

    def pop(self):
        highest_priority = []
        max_value = 0 #keep track of the max priority
        for i in range(len(self.queue)):
            if self.priorty_dict.get(self.queue[i]) > max_value:
                highest_priority.clear() #every time there is a new max value, clear the priority list
                highest_priority.append(i)
                max_value = self.priorty_dict.get(self.queue[i]) #update the max value
            elif self.priorty_dict.get(self.queue[i]) == max_value: #if the current element has the same priority as the max, then add it to the highest priority list
                highest_priority.append(i)
        item = self.queue[highest_priority[0]] #only pop out the first item in the list; it will either be the only item or it will be the one earlier in the queue
        del self.queue[highest_priority[0]] #delete that item from the list to avoid repetition
        return item

    def is_empty(self):
        if not self.queue:
            return True
        else:
            return False        


if __name__ == '__main__':
    print(p1(k=8))
    print('-----------------------------')
    print(p2_a(x=[], y=[1, 3, 5]))
    print(p2_b(x=[2, 4, 6], y=[]))
    print(p2_c(x=[1, 3, 5, 7], y=[1, 2, 5, 6]))
    print(p2_d(x=[1, 3, 5, 7], y=[1, 2, 5, 6]))
    print('------------------------------')
    print(p3_a(x={1, 3, 5, 7}, y={1, 2, 5, 6}, z={7, 8, 9, 1}))
    print(p3_b(x={1, 3, 5, 7}, y={1, 2, 5, 6}, z={7, 8, 9, 1}))
    print(p3_c(x={1, 3, 5, 7}, y={1, 2, 5, 6}, z={7, 8, 9, 1}))
    print('------------------------------')
    print(p4_a())
    print(p4_b(p4_a()))
    print('------------------------------')
    graph = {
        'A': ['D', 'E'],
        'B': ['E', 'F'],
        'C': ['E'],
        'D': ['A', 'E'],
        'E': ['A', 'B', 'C', 'D'],
        'F': ['B'],
        'G': []
    }
    print(p5_a(graph))
    print(p5_b(graph))
    print(p5_c(graph))
    print(p5_d(graph))
    print('------------------------------')
    pq = PriorityQueue()
    pq.push('apple')
    pq.push('kiwi')
    pq.push('orange')
    while not pq.is_empty():
        print(pq.pop())

