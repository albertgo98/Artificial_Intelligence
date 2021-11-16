new_dict = {}
item = ((1,2,3,4), 3)
new_dict[item] =[([1,2,3,4], "hi")]
print(new_dict)

a = [[1,2,3,4], "Up"]
a.insert(0, 5)
a.insert(1, 1)
a.insert(2, 3)
print(a)

# def bfs_search(initial_state):
#     """BFS search"""
#     ### STUDENT CODE GOES HERE ###
#     nodes = dict()
#     explored = set()
#     frontier = Q.Queue()
#     frontier.put((initial_state.config, initial_state.action))
#     waiting = []
#     waiting.append(initial_state.config)
#     start = (tuple(initial_state.config), initial_state.action)

#     while not frontier.empty():
#         # print('here')
#         print(len(list(nodes.keys())))
#         item = frontier.get(0)
#         waiting.pop(0)
#         # print(nodes)
#         state = item[0]
#         print(state)
#         explored.add(tuple(state))

#         curr_state = PuzzleState(state, int(math.sqrt(len(state))))
#         curr_state.parent = state
#         curr_state.config = state

#         # initial_state.parent = state
#         # initial_state.config = state

#         if test_goal(state):
#             print('here')
#             path_to_goal = []
#             complete = False
#             item = (tuple(item[0]), item[1])
#             print(item)
#             while not complete:
#                 # print(item)
#                 for p, c in nodes.items():
#                     for i in c:
#                         # print(i)
#                         if i == item:
#                             path_to_goal.append(i[1])
#                             item = p
#                 # path_to_goal.insert(0, item[1])
#                 # item = list(nodes.keys())[list(nodes.values()).index(item)]
#                 # print(path_to_goal)
#                 if item == start:
#                     complete = True
#             nodes_expanded = len(list(nodes.keys()))
#             cost_of_path = calculate_total_cost(path_to_goal)
#             search_depth = cost_of_path
#             path_to_goal.reverse()
#             print(path_to_goal, nodes_expanded)
#             return path_to_goal, cost_of_path, nodes_expanded, search_depth
#         # print('Come on')
#         for child in curr_state.expand():
#             if (child != None and ((tuple(child[0]) not in explored) and (child[0] not in waiting))):
#                 # print(child)
#                 frontier.put(child)
#                 waiting.append(child[0])
#                 if (tuple(item[0]), item[1]) in list(nodes.keys()):
#                     nodes[(tuple(item[0]), item[1])].append((tuple(child[0]), child[1]))
#                 else:
#                     nodes[(tuple(item[0]), item[1])] = [(tuple(child[0]), child[1])]

#     print("sad")

#     return False



# def A_star_search(initial_state):
#     """A * search"""
#     ### STUDENT CODE GOES HERE ###
#     nodes = dict()
#     explored = set()
#     frontier = Q.PriorityQueue()
#     index_zero = initial_state.config.index(0)
#     m_dist = calculate_manhattan_dist(index_zero, 0, 3)
#     frontier.put((m_dist, 0, 1, initial_state.config, initial_state.action, 0))
#     waiting = []
#     waiting.append([initial_state.config, m_dist])
#     start = (tuple(initial_state.config), initial_state.action, 0)
#     print(start)
#     max_depth = 0
#     nodes_expanded = 0

#     while not frontier.empty():
#         item = frontier.get()
#         # print(item)
#         state = item[3]
#         # print("State: ", state)
#         # print("Waiting: ", waiting)
#         waiting.remove([state, item[0]])
#         curr_depth = item[5]
#         # print(state)
#         if tuple(state) not in explored:
#             explored.add(tuple(state))

#             curr_state = PuzzleState(state, int(math.sqrt(len(state))))
#             curr_state.parent = state
#             curr_state.config = state

#             if test_goal(state):
#                 print('Found goal')
#                 path_to_goal = []
#                 complete = False
#                 item = (tuple(item[3]), item[4], item[5])

#                 while not complete:
#                     # print(item)
#                     for p, c in nodes.items():
#                         for i in c:
#                             # print(i)
#                             if i == item:
#                                 path_to_goal.append(i[1])
#                                 item = p

#                     if item == start:
#                         complete = True
                        
#                 cost_of_path = calculate_total_cost(path_to_goal)
#                 search_depth = cost_of_path
#                 path_to_goal.reverse()
#                 print(path_to_goal, ", ", nodes_expanded, ", ", max_depth)
#                 return path_to_goal, cost_of_path, nodes_expanded, search_depth, max_depth

#             nodes_expanded += 1
#             print(nodes_expanded)
#             for child in curr_state.expand():
#                 if (child != None and (tuple(child[0]) not in explored) and (child[0] not in waiting)):
#                     # print(child)
#                     if curr_depth+1 > max_depth:
#                         max_depth = curr_depth+1

#                     priority2 = 0
#                     if child[1] == "Up":
#                         priority2 = 1
#                     elif child[1] == "Down":
#                         priority2 = 2
#                     elif child[1] == "Left":
#                         priority2 = 3
#                     else:
#                         priority2 = 4
#                     q_size = frontier.qsize()
#                     index0 = child[0].index(0)
#                     pred_dist = calculate_manhattan_dist(index0, 0, 3)+curr_depth+1
#                     child.append(curr_depth+1)
#                     child.insert(0, pred_dist)
#                     child.insert(1, priority2)
#                     child.insert(2, q_size+1)
#                     # print(child)

#                     frontier.put(child)
#                     waiting.append([child[3], pred_dist])

#                     parent_node = (tuple(item[3]), item[4], item[5])
#                     if parent_node in list(nodes.keys()):
#                         nodes[parent_node].append((tuple(child[3]), child[4], child[5]))
#                     else:
#                         nodes[parent_node] = [(tuple(child[3]), child[4], child[5])]
#                 elif child[0] in waiting:
#                     for i in range(len(waiting)):
#                         if waiting[i][0] == child[0]:
#                             index0 = child[0].index(0)
#                             pred_dist = calculate_manhattan_dist(index0, 0, 3)+curr_depth+1
#                             if waiting[i][1] > pred_dist:
#                                 priority2 = 0
#                                 if child[1] == "Up":
#                                     priority2 = 1
#                                 elif child[1] == "Down":
#                                     priority2 = 2
#                                 elif child[1] == "Left":
#                                     priority2 = 3
#                                 else:
#                                     priority2 = 4
#                                 q_size = frontier.size()
#                                 child.append(curr_depth+1)
#                                 child.insert(0, pred_dist)
#                                 child.insert(1, priority2)
#                                 child.insert(2, q_size+1)
#                                 frontier.put(child)

#     return False