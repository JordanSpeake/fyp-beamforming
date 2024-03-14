import numpy as np
import matplotlib.pyplot as plt
import itertools
import copy

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

class DLL_2D:
    def __init__(self, solution_matrix):
        self.main_header = MainHeader()
        self.solution_dimensions = solution_matrix.shape
        # Insert all column headers
        for c, column in enumerate(solution_matrix.T):
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
                header = self.find_header(column_indices[i])
                data.header = header
                data.up = header.up
                data.down = header
                header.up.down = data
                header.up = data

    def find_header(self, name):
        """Iterates through all connected column headers, returning the header with the given numerical name"""
        selected = self.main_header.right
        while selected.name is not None:
            if selected.name == name:
                return selected
            selected = selected.right
        return None

    def find_smallest_header(self):
        """Returns the column header with the smallest number of 1s in it's column"""
        selected = self.main_header.right
        smallest_size = selected.size
        smallest_header = self.main_header.right
        while selected.name is not None:
            if selected.size < smallest_size:
                smallest_size = selected.size
                smallest_header = selected
            selected = selected.right
        return smallest_header

    def get_rows_in_column(self, column_header):
        rows = []
        data = column_header.down
        while data is not column_header.header:
            rows.append(data)
            data = data.down
        return rows

    def row_bits_from_element(self, element):
        """Returns the value of a row, that the given element is a member of, in a form suitable for np arrays"""
        indices = []
        starting_header = element.header
        indices.append(starting_header.name)
        element = element.right
        column_header = element.header
        while column_header.name != starting_header.name:
            indices.append(column_header.name)
            element = element.right
            column_header = element.header
        out = np.zeros(self.solution_dimensions[1])
        out[indices] = 1
        return out

def generate_starting_matrix(num_elements, polyomino_size):
    result = []
    for bits in itertools.combinations(range(num_elements), polyomino_size):
        row = np.zeros(num_elements, dtype=int)
        row[list(bits)] = 1
        result.append(row)
    return np.asarray(result)

def dancing_links_x(dll_2d, solution):
    if dll_2d.main_header.right == dll_2d.main_header:
        return solution
    else:
        next_dll_2d = copy.deepcopy(dll_2d)
        selected_column = next_dll_2d.find_smallest_header()
        rows_in_column = next_dll_2d.get_rows_in_column(selected_column)
        selected_row = rows_in_column[np.random.randint(0, len(rows_in_column))]
        solution.append(dll_2d.row_bits_from_element(selected_row))
        # pick one at random, add it to the partial solution
        # remove all columns satisfied by that row
        # recurse


def main():
    array = np.zeros((3, 3), dtype=int)
    polyomino_size = 3
    assert np.mod(array.size, polyomino_size) == 0
    print(f"Creating solution matrix for:\n    Array: {array.shape}\n    Polyomino: {polyomino_size}")
    solution_matrix = generate_starting_matrix(array.size, polyomino_size)
    print("Generating initial 2D doubly linked list")
    dll_2d = DLL_2D(solution_matrix)
    print("Solving...")
    solution = []
    solution = dancing_links_x(dll_2d, solution)


main()
