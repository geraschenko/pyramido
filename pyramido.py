import numpy as np

# We encode the board in a 5x8 array. This leaves 10 empty spaces, in which we
# encode the current player and the winning player (if any).

def toInt(array):
    result = 0
    for h in range(5):
        for x in range(8):
            if x > 7 - h:
                continue
            result = 3*result + array[h, x]
    result = 3*result + array[4, 6]  # winner
    result = 3*result + array[4, 7]  # current player
    return result


def toArray(intPos):
    result = - np.ones((5, 8), np.int8)
    result[4, 7] = intPos % 3  # current player
    intPos /= 3
    result[4, 6] = intPos % 3  # winner
    intPos /= 3
    for h in reversed(range(5)):
        for x in reversed(range(8)):
            if x > 7 - h:
                continue
            result[h, x] = intPos % 3
            intPos /= 3
    return result


def testConversion():
    for _ in range(100):
        a = np.random.randint(0, 3, (5, 8))
        a[1, 7] = -1
        a[2, 6:] = [-1, -1]
        a[3, 5:] = [-1, -1, -1]
        a[4, 4:6] = [-1, -1]
        a[4, 7] = np.random.randint(1, 3)  # current player
        success = np.logical_and.reduce((a == toArray(toInt(a))).flatten())
        if not success:
            print('failed for\n%s\ntoInt = %d, toArray(toInt) =\n%s' %
                    (str(a), toInt(a), toArray(toInt(a))))
        i = np.random.randint(0, pow(3, 32))
        if i != toInt(toArray(i)):
            print('failed for\n%d\ntoArray = %s, toInt(toArray) =%d' %
                    (i, toArray(i), toInt(toArray(i))))


class PyramidoPosition(object):

    def __init__(self, array):
        self.int = toInt(array)

    def array(self):
        return toArray(self.int)

    def nn_format(self):
        a = self.array()
        result = np.zeros((2, 31))
        i = 0
        for h in range(5):
            for x in range(8):
                if x > 7 - h:
                    continue
                if a[h, x] != 0:
                    result[a[h, x] - 1, i] = 1
                i += 1
        result[a[4, 7] - 1, 30] = 1  # current player
        return result


    def __str__(self):
        a = self.array()
        s = ''
        s += '    ' + str(a[4][:4]) + '\n'
        s += '   ' + str(a[3][:5]) + '   Pieces used: 1:{} 2:{}\n'
        s += '  ' + str(a[2][:6]) + '  Current player: ' + str(current_player(a)) + '\n'
        s += ' ' + str(a[1][:7]) + '\n'
        s += str(a[0])
        # s += 'Current player: %d' % current_player(a) + '\n'
        s = s.replace('1', 'X')
        s = s.replace('2', 'O')
        s = s.replace('0', '_')
        counts = piece_counts(a)
        s = s.format(counts[1], counts[2])
        # s += 'Pieces used: X: %d , O: %d' % (counts[1], counts[2])
        if a[4, 6] != 0:
            s += '\n%s wins!' % ('X' if a[4,6]==1 else 'O')
        # s += '\n' + str(a) + ' ' + str(self.int) + '\n'
        return s


def current_player(array):
    return array[4, 7]


def other_player(player):
    return 3-player


def piece_counts(array):
    d = dict(zip(*np.unique(array, return_counts=True)))
    for i in [1, 2]:
        if i not in d:
            d[i] = 0
    for x in [6, 7]:
        if array[4, x] != 0:
            d[array[4, x]] -= 1
    return d


def falling_pieces(array, player):
    result = []
    for h in range(1, 5):
        for x in range(0, 8-h):
            if array[h, x] == player and array[h-1, x] == 0 and array[h-1, x+1] == 0:
                result.append((h, x))
    return result


def possible_moves(position):
    a = position.array()
    player = current_player(a)
    a[4, 7] = other_player(player)
    moves = []

    # Game over
    if a[4, 6] != 0:
        return moves

    top_row_count = dict(zip(*np.unique(a[4, :4], return_counts=True)))
    if player in top_row_count and top_row_count[player] >= 2:
        # Assume current player wins. This must be zeroed out if it turns out
        # no moves are available, or if the current player must fall from the
        # top row.
        a[4, 6] = player

    # Falling moves.
    falling = falling_pieces(a, player)
    if falling:
        for h, x in falling:
            if h == 4:
                # Falling from the top, so zero out winning field.
                a[4, 6] = 0
            p = a.copy()
            p[h, x] = 0
            p[h-1, x] = player
            moves.append(PyramidoPosition(p))
            p = a.copy()
            p[h, x] = 0
            p[h-1, x+1] = player
            moves.append(PyramidoPosition(p))
        return moves

    # Placing moves.
    if piece_counts(a)[player] < 13:
        for x in range(8):
            if a[0, x] == 0:
                p = a.copy()
                p[0, x] = player
                moves.append(PyramidoPosition(p))

    # Moving moves.
    for h in range(0, 4):
        for x in range(0, 8-h):
            if a[h, x] == player:
                # Move left.
                if 0 < x and a[h+1, x-1] == 0 and a[h, x-1] != 0:
                    p = a.copy()
                    p[h, x] = 0
                    p[h+1, x-1] = player
                    moves.append(PyramidoPosition(p))
                # Move right.
                if x+1 < 8-h and a[h+1, x] == 0 and a[h, x+1] != 0:
                    p = a.copy()
                    p[h, x] = 0
                    p[h+1, x] = player
                    moves.append(PyramidoPosition(p))

    if not moves:
        # Other player wins
        a[4, 6] = other_player(player)
        moves.append(PyramidoPosition(a))

    return moves


def leaf_node_value(position):
    a = position.array()
    w = a[4, 6]
    if w == 0:
        return 0.5
    elif w == current_player(a):
        return 1.0
    else:
        return 0.0
