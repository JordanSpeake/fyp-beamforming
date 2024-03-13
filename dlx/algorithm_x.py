import numpy as np
import matplotlib.pyplot as plt
import itertools

class Data:
    def __init__(self, up, down, header, left=self, right=self):
        self.left = left
        self.right = right
        self.up = up
        self.down = down
        self.header = header

class ColumnHeader(Data):
    def __init__(self, left, right, size, name):
        self.left = left
        self.right = right
        self.up = self
        self.down = self
        self.header = self
        self.size = size
        self.name = name


class MainHeader:
    def __init__(self):
        self.left = self
        self.right = self

    def find_header(name):
        """Iterates through all connected column headers, returning the header with the given numerical name"""
        selected = self.right
        while selected is not self:
            if selected.name == name:
                return selected
        return None


class DLL_2D:
    def __init__(self, solution_matrix):
        self.main_header = MainHeader()
        # Insert all column headers
        for c, column in enumerate(solution_matrix.T):
            ones_in_column = np.sum(column)
            header = ColumnHeader(left=self.main_header.left, right=self.main_header, size=ones_in_column, name=c)
            self.main_header.left.right = header
            self.main_header.left = header
        # For each row:
        for _, row in enumerate(solution_matrix):
            # represent the 1s in solution_matrix
            for i, value in enumerate(row):
                if value == 1:
                    selected_header = self.main_header.find_header(i)
                    for _ in range(selected_header.size):
                        data = Data(up=selected_header.up, down=selected_header, header=selected_header)
                        selected_header.up.down = data
                        selected_header.up = data
            # connect the newly added data nodes

def generate_starting_matrix(num_elements, polyomino_size):
    result = []
    for bits in itertools.combinations(range(num_elements), polyomino_size):
        row = np.zeros(num_elements, dtype=int)
        row[list(bits)] = 1
        result.append(row)
    return np.asarray(result)

def main():
    # array = np.zeros((4, 4), dtype=int)
    # polyomino_size = 5
    example_matrix = np.array([
            [0, 0, 1, 0, 1, 1, 0],
            [1, 0, 0, 1, 0, 0, 1],
            [0, 1, 1, 0, 0, 1, 0],
            [1, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 1, 0, 1]
        ]
    )
    print("Generating initial 2D doubly linked list")
    dll_2d = DLL_2D(example_matrix)


main()
