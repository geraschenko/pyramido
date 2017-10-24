import numpy as np
from pyramido import possible_moves, leaf_node_value, PyramidoPosition, current_player, toArray
# from ttt import possible_moves, leaf_node_value, TicTacToePosition, current_player, toArray

class Node(object):
    def __init__(self, position, priors):
        # input('building node for\n%s' % str(position))
        self.edges = []
        for child, probability in priors.items():
            self.edges.append(Edge(child, probability))
        if not self.edges:
            self.value = leaf_node_value(position)  # game must define this

    def visit(self, nodes, policy):
        '''Visits edge with maximum adjusted value, adding new nodes if needed.
        Returns value of final leaf node reached.'''
        if self.edges:
            i = np.argmax([e.adjusted_value for e in self.edges])
            return self.edges[i].visit(nodes, policy)
        else:
            # input('got result value %f' % self.value)
            return self.value

class Edge(object):

    def __init__(self, target_position, prior):
        # input('building edge to\n%s' % str(target_position))
        self.target_position = target_position
        self.visit_count = 0
        self.total_action_value = 0.0
        self.prior = prior

    @property
    def action_value(self):
        if self.visit_count:
            return self.total_action_value / self.visit_count
        # TODO: this should probably come from the policy?
        return 0.5

    @property
    def upper_confidence(self):
        return self.prior / (1.0 + self.visit_count)

    @property
    def adjusted_value(self):
        # TODO: tune the scale of upper_confidece. This is something like
        # "minimum times to explore a move"
        return self.action_value + 10 * self.upper_confidence

    def visit(self, nodes, policy):
        '''Traverses this edge, adding new nodes if needed.
        Returns value of final leaf node reached.'''
        # print('visiting edge to\n%s\n' % str(self.target_position))
        if self.target_position not in nodes:
            priors, _ = policy(self.target_position)
            nodes[self.target_position] = Node(self.target_position, priors)
        # Value must be from the point of view of the current player. This is
        # where the switch in point of view happens.
        result = 1.0 - nodes[self.target_position].visit(nodes, policy)
        self.visit_count += 1
        self.total_action_value += result
        return result


def MCTS(policy, position, iterations, temperature=1.0):
    priors, _ = policy(position)
    nodes = {position: Node(position, priors)}
    for _ in range(iterations):
        nodes[position].visit(nodes, policy)

    visit_counts = [e.visit_count for e in nodes[position].edges]
    distribution = np.power(visit_counts, 1.0 / temperature)
    distribution /= np.sum(distribution)
    values = [e.action_value for e in nodes[position].edges]
    if values:
        value = np.dot(distribution, values)
    else:
        value = leaf_node_value(position)

    for i, e in enumerate(nodes[position].edges):
        priors[e.target_position] = distribution[i]
        # print('%s\nvisits: %d, probability: %f, action_value: %f\n' % (str(e.target_position), e.visit_count, distribution[i], e.action_value))
    # print('value = %f' % value)

    return priors, value


def val(position):
    return 0.5

def pol(position):
    distribution = {}
    moves = possible_moves(position)  # game must define this
    for m in moves:
        distribution[m] = 1.0 / len(moves)
    return distribution, val(position)

def playTTT():
    iterations = 1000
    position = TicTacToePosition(toArray(0))
    dist, value = MCTS(pol, position, iterations)
    print('assessed value: %f' % value)
    while dist:
        max_probability = max(dist.values())
        for move, probability in dist.items():
            if probability == max_probability:
                print('Player %d moves to\n%s' % (current_player(position.array()), str(move)))
                position = move
                break
        dist, value = MCTS(pol, position, iterations)
        print('assessed value: %f' % value)

def pyramido_policy(position):
    moves = possible_moves(position)  # game must define this
    bottom_row_moves = []
    other_moves = []
    for m in moves:
        if np.logical_and.reduce(m.array()[0] == position.array()[0]):
            other_moves.append(m)
        else:
            bottom_row_moves.append(m)

    bottom_row_weight = 1.0
    if not other_moves:
        bottom_row_weight = 1.0
    if not bottom_row_moves:
        bottom_row_weight = 0.0

    distribution = {}
    for m in bottom_row_moves:
        distribution[m] = bottom_row_weight / len(bottom_row_moves)
    for m in other_moves:
        distribution[m] = (1.0 - bottom_row_weight) / len(other_moves)
    return distribution, val(position)


def humanPickMove(position):
    moves = possible_moves(position)
    for i, m in enumerate(moves):
        print('Option %d:\n%s\n' % (i, str(m)))
    selection = int(input('Selection:\n'))
    return moves[selection]

def computerPickMove(position, iterations):
    dist, value = MCTS(pyramido_policy, position, iterations)
    print('\nassessed value: %f' % value)
    max_probability = max(dist.values())
    for move, probability in dist.items():
        if probability == max_probability:
            return move


def playPyramido(human=0, iterations=1000):
    movenumber = 1
    position = PyramidoPosition(toArray(1))
    player = 1
    while possible_moves(position):
        if player == human:
            position = humanPickMove(position)
        else:
            position = computerPickMove(position, iterations)
        print('Move %d: Player %d moves to\n%s'
                % (movenumber, player, str(position)))
        movenumber += 1
        player = 3 - player

