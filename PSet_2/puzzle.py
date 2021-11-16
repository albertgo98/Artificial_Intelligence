from __future__ import division
from __future__ import print_function

import sys
import math
import time
import queue as Q
import psutil
import resource

#### SKELETON CODE ####
## The Class that Represents the Puzzle
class PuzzleState(object):
    """
        The PuzzleState stores a board configuration and implements
        movement instructions to generate valid children.
    """
    def __init__(self, config, n, parent=None, action="Initial", cost=0):
        """
        :param config->List : Represents the n*n board, for e.g. [0,1,2,3,4,5,6,7,8] represents the goal state.
        :param n->int : Size of the board
        :param parent->PuzzleState
        :param action->string
        :param cost->int
        """
        if n*n != len(config) or n < 2:
            raise Exception("The length of config is not correct!")
        if set(config) != set(range(n*n)):
            raise Exception("Config contains invalid/duplicate entries : ", config)

        self.n        = n
        self.cost     = cost
        self.parent   = parent
        self.action   = action
        self.config   = config
        self.children = []


        # Get the index and (row, col) of empty block
        self.blank_index = self.config.index(0)

    def display(self):
        """ Display this Puzzle state as a n*n board """

        for i in range(self.n):
            print(self.config[3*i : 3*(i+1)])

    def move_up(self):
        """ 
        Moves the blank tile one row up.
        :return a PuzzleState with the new configuration
        """
        new_config = list(self.config)
        row = math.floor(self.blank_index/3)
        col = self.blank_index % 3
        if row != 0:
            new_row = row-1
            swap_with = new_config[3*new_row+col] 
            empty_index = self.blank_index
            new_config[empty_index] = swap_with
            new_config[3*new_row+col] = 0
            self.action = "Up"
            return [new_config, self.action]

      
    def move_down(self):
        """
        Moves the blank tile one row down.
        :return a PuzzleState with the new configuration
        """
        new_config = list(self.config)
        row = math.floor(self.blank_index/3)
        col = self.blank_index % 3
        if row != 2:
            new_row = row+1
            swap_with = new_config[3*new_row+col] 
            empty_index = self.blank_index
            new_config[empty_index] = swap_with
            new_config[3*new_row+col] = 0
            self.action = "Down"
            return [new_config, self.action]
      
    def move_left(self):
        """
        Moves the blank tile one column to the left.
        :return a PuzzleState with the new configuration
        """
        new_config = list(self.config)
        row = math.floor(self.blank_index/3)
        col = self.blank_index % 3
        if col != 0:
            new_col = col-1
            swap_with = new_config[3*row+new_col] 
            empty_index = self.blank_index
            new_config[empty_index] = swap_with
            new_config[3*row+new_col] = 0
            self.action = "Left"
            return [new_config, self.action]

    def move_right(self):
        """
        Moves the blank tile one column to the right.
        :return a PuzzleState with the new configuration
        """
        new_config = list(self.config)
        row = math.floor(self.blank_index/3)
        col = self.blank_index % 3
        if col != 2:
            new_col = col+1
            swap_with = new_config[3*row+new_col] 
            empty_index = self.blank_index
            new_config[empty_index] = swap_with
            new_config[3*row+new_col] = 0
            self.action = "Right"
            return [new_config, self.action]
      
    def expand(self):
        """ Generate the child nodes of this node """
        
        # Node has already been expanded
        if len(self.children) != 0:
            return self.children
        
        # Add child nodes in order of UDLR
        children = [
            self.move_up(),
            self.move_down(),
            self.move_left(),
            self.move_right()]

        # Compose self.children of all non-None children states
        self.children = [state for state in children if state is not None]
        # print("Children: ", children)
        return self.children

# Function that Writes to output.txt

### Students need to change the method to have the corresponding parameters
def writeOutput(path_to_goal, cost_of_path, nodes_expanded, search_depth, max_search_depth, running_time, max_ram_usage):
    ### Student Code Goes here
    f = open("output.txt", "w")
    f.write("path_to_goal: " + str(path_to_goal) + '\n')
    f.write("cost_of_path: " + str(cost_of_path) + '\n')
    f.write("nodes_expanded: " + str(nodes_expanded) + '\n')
    f.write("search_depth: " + str(search_depth) + '\n')
    f.write("max_search_depth: " + str(max_search_depth) + '\n')
    f.write("running_time: " + str(running_time) + '\n')
    f.write("max_ram_usage: " + str(max_ram_usage) + '\n')

    f.close()

    return


def bfs_search(initial_state):
    """BFS search"""
    ### STUDENT CODE GOES HERE ###
    nodes = dict()
    explored = set()
    frontier = Q.Queue()
    frontier.put((initial_state.config, initial_state.action, 0))
    waiting = []
    waiting.append(initial_state.config)
    start = (tuple(initial_state.config), initial_state.action, 0)
    print(start)
    max_depth = 0
    nodes_expanded = 0

    while not frontier.empty():
        item = frontier.get()
        waiting.pop(0)
        # print(nodes)
        state = item[0]
        curr_depth = item[2]
        # print(state)
        explored.add(tuple(state))

        curr_state = PuzzleState(state, int(math.sqrt(len(state))))
        curr_state.parent = state
        curr_state.config = state

        if test_goal(state):
            print('Found goal')
            path_to_goal = []
            complete = False
            item = (tuple(item[0]), item[1], item[2])

            while not complete:
                # print(item)
                for p, c in nodes.items():
                    for i in c:
                        # print(i)
                        if i == item:
                            path_to_goal.append(i[1])
                            item = p

                if item == start:
                    complete = True

            cost_of_path = calculate_total_cost(path_to_goal)
            search_depth = cost_of_path
            path_to_goal.reverse()
            print(path_to_goal, ", ", nodes_expanded, ", ", max_depth)
            return path_to_goal, cost_of_path, nodes_expanded, search_depth, max_depth

        nodes_expanded += 1
        print(nodes_expanded)
        for child in curr_state.expand():
            if (child != None and (tuple(child[0]) not in explored) and (child[0] not in waiting)):
                # print(child)
                if item[2]+1 > max_depth:
                    max_depth = item[2]+1
                child.append(item[2]+1)
                frontier.put(child)
                waiting.append(child[0])
                if (tuple(item[0]), item[1], item[2]) in list(nodes.keys()):
                    nodes[(tuple(item[0]), item[1], item[2])].append((tuple(child[0]), child[1], child[2]))
                else:
                    nodes[(tuple(item[0]), item[1], item[2])] = [(tuple(child[0]), child[1], child[2])]

    return False

def dfs_search(initial_state):
    """DFS search"""
    ### STUDENT CODE GOES HERE ###
    nodes = dict()
    explored = set()
    frontier = []
    frontier.append((initial_state.config, initial_state.action, 0))
    waiting = []
    waiting.append(initial_state.config)
    start = (tuple(initial_state.config), initial_state.action, 0)
    print(start)
    max_depth = 0
    nodes_expanded = 0

    while frontier:
        item = frontier.pop()
        waiting.pop()
        state = item[0]
        curr_depth = item[2]
        explored.add(tuple(state))

        curr_state = PuzzleState(state, int(math.sqrt(len(state))))
        curr_state.parent = state
        curr_state.config = state

        if test_goal(state):
            print('here')
            path_to_goal = []
            complete = False
            item = (tuple(item[0]), item[1], item[2])

            while not complete:
                for p, c in nodes.items():
                    for i in c:
                        if i == item:
                            path_to_goal.append(i[1])
                            item = p
                if item == start:
                    complete = True

            cost_of_path = calculate_total_cost(path_to_goal)
            search_depth = cost_of_path
            path_to_goal.reverse()
            print(path_to_goal, ", ", nodes_expanded, ", ", max_depth)
            return path_to_goal, cost_of_path, nodes_expanded, search_depth, max_depth

        nodes_expanded += 1
        print(nodes_expanded)
        temp_children = []
        for child in curr_state.expand():
            if (child[0] != None and (tuple(child[0]) not in explored) and (child[0] not in waiting)):
                # print(child)
                if item[2]+1 > max_depth:
                    max_depth = item[2]+1
                child.append(item[2]+1)
                temp_children.append(child)
                if (tuple(item[0]), item[1], item[2]) in list(nodes.keys()):
                    nodes[(tuple(item[0]), item[1], item[2])].append((tuple(child[0]), child[1], child[2]))
                else:
                    nodes[(tuple(item[0]), item[1], item[2])] = [(tuple(child[0]), child[1], child[2])]
        if temp_children:
            temp_children.reverse()
            for child in temp_children:
                frontier.append(child)
                waiting.append(child[0])
    return False

def A_star_search(initial_state):
    """A * search"""
    ### STUDENT CODE GOES HERE ###
    nodes = dict()
    explored = set()
    frontier = Q.PriorityQueue()
    index_zero = initial_state.config.index(0)
    m_dist = calculate_manhattan_dist(index_zero, 0, 3)
    frontier.put((m_dist, 0, 1, initial_state.config, initial_state.action, 0))
    waiting = dict()
    waiting[tuple(initial_state.config)] = m_dist
    start = (tuple(initial_state.config), initial_state.action, 0)
    print(start)
    max_depth = 0
    nodes_expanded = 0

    while not frontier.empty():
        item = frontier.get()
        # print(item)
        state = item[3]
        # print("State: ", state)
        # print("Waiting: ", waiting)
        # waiting.remove([state, item[0]])
        curr_depth = item[5]
        # print(state)
        if tuple(state) not in explored:
            del waiting[tuple(state)]
            explored.add(tuple(state))

            curr_state = PuzzleState(state, int(math.sqrt(len(state))))
            curr_state.parent = state
            curr_state.config = state

            if test_goal(state):
                print('Found goal')
                path_to_goal = []
                complete = False
                item = (tuple(item[3]), item[4], item[5])

                while not complete:
                    # print(item)
                    for p, c in nodes.items():
                        for i in c:
                            # print(i)
                            if i == item:
                                path_to_goal.append(i[1])
                                item = p

                    if item == start:
                        complete = True
                        
                cost_of_path = calculate_total_cost(path_to_goal)
                search_depth = cost_of_path
                path_to_goal.reverse()
                print(path_to_goal, ", ", nodes_expanded, ", ", max_depth)
                return path_to_goal, cost_of_path, nodes_expanded, search_depth, max_depth

            nodes_expanded += 1
            print(nodes_expanded)
            for child in curr_state.expand():
                if (child != None and (tuple(child[0]) not in explored) and (tuple(child[0]) not in waiting)):
                    # print(child)
                    if curr_depth+1 > max_depth:
                        max_depth = curr_depth+1

                    priority2 = 0
                    if child[1] == "Up":
                        priority2 = 1
                    elif child[1] == "Down":
                        priority2 = 2
                    elif child[1] == "Left":
                        priority2 = 3
                    else:
                        priority2 = 4
                    q_size = frontier.qsize()
                    pred_dist = 0
                    for i in child[0]:
                        if i != 0:
                            ind = child[0].index(i)
                            pred_dist += calculate_manhattan_dist(ind, i, 3)
                    pred_dist = pred_dist+curr_depth+1
                    child.append(curr_depth+1)
                    child.insert(0, pred_dist)
                    child.insert(1, priority2)
                    child.insert(2, q_size+1)
                    # print(child)

                    frontier.put(child)
                    waiting[tuple(child[3])] = pred_dist

                    parent_node = (tuple(item[3]), item[4], item[5])
                    if parent_node in list(nodes.keys()):
                        nodes[parent_node].append((tuple(child[3]), child[4], child[5]))
                    else:
                        nodes[parent_node] = [(tuple(child[3]), child[4], child[5])]
                elif tuple(child[0]) in waiting:
                    pred_dist = 0
                    for i in child[0]:
                        if i != 0:
                            ind = child[0].index(i)
                            pred_dist += calculate_manhattan_dist(ind, i, 3)
                    pred_dist = pred_dist+curr_depth+1
                    if pred_dist < waiting.get(tuple(child[0])):
                        # print(pred_dist, waiting.get(tuple(child[0])))
                        priority2 = 0
                        if child[1] == "Up":
                            priority2 = 1
                        elif child[1] == "Down":
                            priority2 = 2
                        elif child[1] == "Left":
                            priority2 = 3
                        else:
                            priority2 = 4
                        q_size = frontier.qsize()
                        child.append(curr_depth+1)
                        child.insert(0, pred_dist)
                        child.insert(1, priority2)
                        child.insert(2, q_size+1)
                        frontier.put(child)
                        waiting[tuple(child[3])] = pred_dist

    return False

def calculate_total_cost(state):
    """calculate the total estimated cost of a state"""
    ### STUDENT CODE GOES HERE ###
    return len(state)

def calculate_manhattan_dist(idx, value, n):
    """calculate the manhattan distance of a tile"""
    ### STUDENT CODE GOES HERE ###
    row = math.floor(idx/n)
    col = idx % 3
    if value == 1:
        return row+abs(col-1)
    elif value == 2:
        return row+abs(col-2)
    elif value == 3:
        return abs(row-1)+col
    elif value == 4:
        return abs(row-1)+abs(col-1)
    elif value == 5:
        return abs(row-1)+abs(col-2)
    elif value == 6:
        return abs(row-2)+col
    elif value == 7:
        return abs(row-2)+abs(col-1)
    elif value == 8:
        return abs(row-2)+abs(col-2)

def test_goal(puzzle_state):
    """test the state is the goal state or not"""
    ### STUDENT CODE GOES HERE ###
    goal = [0, 1 ,2 ,3, 4, 5, 6, 7, 8]
    return puzzle_state == goal

# Main Function that reads in Input and Runs corresponding Algorithm
def main():
    search_mode = sys.argv[1].lower()
    begin_state = sys.argv[2].split(",")
    begin_state = list(map(int, begin_state))
    board_size  = int(math.sqrt(len(begin_state)))
    hard_state  = PuzzleState(begin_state, board_size)
    start_time  = time.time()
    process = psutil.Process()
    
    if   search_mode == "bfs": path_to_goal, cost_of_path, nodes_expanded, search_depth, max_depth = bfs_search(hard_state)
    elif search_mode == "dfs": path_to_goal, cost_of_path, nodes_expanded, search_depth, max_depth = dfs_search(hard_state)
    elif search_mode == "ast": path_to_goal, cost_of_path, nodes_expanded, search_depth, max_depth = A_star_search(hard_state)
    else: 
        print("Enter valid command arguments !")
        
    end_time = time.time()
    running_time = round(end_time-start_time, 8)
    max_ram_usage = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(10**-6), 8)

    print("Program completed in %.3f second(s)"%(end_time-start_time))

    writeOutput(path_to_goal, cost_of_path, nodes_expanded, search_depth, max_depth, running_time, max_ram_usage)

if __name__ == '__main__':
    main()
