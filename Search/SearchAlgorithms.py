from simpleai.search import uniform_cost, greedy, astar, SearchProblem

GOAL = '''1-2-3
8- -4
7-6-5'''

INITIAL = '''5-8-6
3- -1
4-2-7'''


class EightPuzzle(SearchProblem):
    def actions(self, cur_state):
        rows = string_to_list(cur_state)
        row_empty, col_empty = get_location(rows, ' ')

        actions = []
        if row_empty > 0:
            actions.append(rows[row_empty - 1][col_empty])
        if row_empty < 2:
            actions.append(rows[row_empty + 1][col_empty])
        if col_empty > 0:
            actions.append(rows[row_empty][col_empty - 1])
        if col_empty < 2:
            actions.append(rows[row_empty][col_empty + 1])

        return actions

    def result(self, state, action):
        rows = string_to_list(state)
        row_empty, col_empty = get_location(rows, ' ')
        row_new, col_new = get_location(rows, action)

        rows[row_empty][col_empty], rows[row_new][col_new] = \
            rows[row_new][col_new], rows[row_empty][col_empty]

        return list_to_string(rows)

    def is_goal(self, state):
        return state == GOAL

    def heuristic(self, state):
        rows = string_to_list(state)

        distance = 0

        for number in '12345678 ':
            row_new, col_new = get_location(rows, number)
            row_new_goal, col_new_goal = goal_positions[number]

            distance += abs(row_new - row_new_goal) + \
                abs(col_new - col_new_goal)

        return distance


def list_to_string(input_list):
    return '\n'.join(['-'.join(x) for x in input_list])


def string_to_list(input_string):
    return [x.split('-') for x in input_string.split('\n')]


def get_location(rows, input_element):
    for i, row in enumerate(rows):
        for j, item in enumerate(row):
            if item == input_element:
                return i, j


goal_positions = {}
rows_goal = string_to_list(GOAL)
for number in '12345678 ':
    goal_positions[number] = get_location(rows_goal, number)

# result = uniform_cost(EightPuzzle(INITIAL))
# result = greedy(EightPuzzle(INITIAL))
# result = astar(EightPuzzle(INITIAL))

for i, (action, state) in enumerate(result.path()):
    print()
    if action == None:
        print('Initial')
    elif i == len(result.path()) - 1:
        print('moved', action, 'complete')
    else:
        print('moved', action, 'not yet complete')

    print(state)
print('moved', i, 'times')
