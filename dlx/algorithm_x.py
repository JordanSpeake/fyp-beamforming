import numpy as np
import matplotlib.pyplot as plt
import itertools

class Data:
    def __init__(self, up, down, header, left=None, right=None):
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
        self.name = None

    def find_header(self, name):
        """Iterates through all connected column headers, returning the header with the given numerical name"""
        selected = self.right
        while selected.name is not None:
            print(selected.name)
            if selected.name == name:
                return selected
            selected = selected.right
        return None


class DLL_2D:
    def __init__(self, solution_matrix):
        self.main_header = MainHeader()
        # Insert all column headers
        for c, column in enumerate(solution_matrix.T):
            print(f"C: {c}")
            ones_in_column = np.sum(column)
            header = ColumnHeader(left=self.main_header.left, right=self.main_header, size=ones_in_column, name=c)
            self.main_header.left.right = header
            self.main_header.left = header
        # For each row:
        for _, row in enumerate(solution_matrix):
            # Create a list of points representing the row
            column_indices = []
            data_points = []
            for i, value in enumerate(row):
                if value == 1:
                    column_indices.append(i)
                    data_points.append(Data(up=self, down=self, header=None))
            # Connect the points up horizontally and vertically
            for i, data in enumerate(data_points):
                data.left = data_points[i-1]
                data.right = data_points[np.mod(i+1, len(data_points))]
                header = self.main_header.find_header(column_indices[i])
                data.header = header
                data.up = header.up
                data.down = header
                header.up.down = data
                header.up = data

def generate_starting_matrix(num_elements, polyomino_size):
    result = []
    for bits in itertools.combinations(range(num_elements), polyomino_size):
        row = np.zeros(num_elements, dtype=int)
        row[list(bits)] = 1
        result.append(row)
    return np.asarray(result)

def main():
    array = np.zeros((3, 3), dtype=int)
    polyomino_size = 3
    assert np.mod(array.size, polyomino_size) == 0
    print(f"Creating solution matrix for:\n    Array: {array.shape}\n    Polyomino: {polyomino_size}")
    solution_matrix = generate_starting_matrix(array.size, polyomino_size)
    print("Generating initial 2D doubly linked list")
    dll_2d = DLL_2D(solution_matrix)
    pass
    # Recursively solve
    # return solution.... somehow


main()
