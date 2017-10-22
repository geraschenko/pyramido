import numpy as np

def toInt(array):
    result = 0
    for i in range(3):
        for j in range(3):
            result = 3*result + array[i, j]
    return result

def toArray(intPos):
    result = np.zeros((3, 3), np.int8)
    for i in reversed(range(3)):
        for j in reversed(range(3)):
            result[i, j] = intPos % 3
            intPos /= 3
    return result

def testConversion():
    for _ in range(100):
        a = np.random.randint(0, 3, (3, 3))
        success = np.logical_and.reduce((a == toArray(toInt(a))).flatten())
        if not success:
            print('failed for\n%s\ntoInt = %d, toArray(toInt) =\n%s' %
                    (str(a), toInt(a), toArray(toInt(a))))
        i = np.random.randint(0, pow(3, 9))
        if i != toInt(toArray(i)):
            print('failed for\n%d\ntoArray = %s, toInt(toArray) =%d' %
                    (i, toArray(i), toInt(toArray(i))))


class TicTacToePosition(object):

    def __init__(self, array):
        self.int = toInt(array)

    def array(self):
        return toArray(self.int)

    def __str__(self):
        return str(toArray(self.int))

def winner(position):
    a = position.array()
    for i in range(3):
        if a[i, 0] and a[i, 0] == a[i, 1] and a[i, 0] == a[i, 2]:
            return a[i, 0]
        if a[0, i] and a[0, i] == a[1, i] and a[0, i] == a[2, i]:
            return a[0, i]
    if a[0, 0] and a[0, 0] == a[1, 1] and a[0, 0] == a[2, 2]:
        return a[0, 0]
    if a[0, 2] and a[0, 2] == a[1, 1] and a[0, 2] == a[2, 0]:
        return a[0, 2]
    return 0

def current_player(array):
    num_empty_squares = np.sum(array == 0)
    if num_empty_squares % 2 == 1:
        return 1
    else:
        return 2

def possible_moves(position):
    if winner(position):
        return []
    moves = []
    a = position.array()
    curr_player = current_player(a)

    for i in range(3):
        for j in range(3):
            if a[i, j] == 0:
                b = a.copy()
                b[i, j] = curr_player
                moves.append(TicTacToePosition(b))

    return moves

def leaf_node_value(position):
    w = winner(position)
    if w == 0:
        return 0.5
    elif w == current_player(position.array()):
        return 1.0
    else:
        return 0.0
