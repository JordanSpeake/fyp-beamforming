import numpy as np
import matplotlib.pyplot as plt
import itertools

"""An implementation of Donald Knuth's Dancing Links Algorithm X"""
"""https://arxiv.org/pdf/cs/0011047.pdf"""

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
        smallest_header = self.main_header.right
        while selected.name is not None:
            if selected.size < smallest_header.size:
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

    def get_columns_in_row(self, row_element):
        columns = []
        data = row_element.right
        while data is not row_element:
            columns.append(data)
            data = data.right
        return columns

    def row_bits_from_element(self, element):
        """Returns the value of a row, that the given element is a member of, in a form suitable for np arrays"""
        indices = [element.header.name]
        data = element.right
        while data is not element:
            indices.append(data.header.name)
            data = data.right
        out = np.zeros(self.solution_dimensions[1])
        out[indices] = 1
        return out


    def cover_column(self, column):
        header = column.header
        header.right.left = header.left
        header.left.right = header.right
        i = header.down
        while i is not header:
            j = i.right
            while j is not i:
                j.down.up = j.up
                j.up.down = j.down
                j.header.size -= 1
                j = j.right
            i = i.down

    def uncover_column(self, column):
        header = column.header
        i = header.up
        while i is not header:
            j = i.left
            while j is not i:
                j.header.size += 1
                j.down.up = j
                j.up.down = j
                j = j.left
            i = i.up
        header.right.left = header
        header.left.right = header


def search(dll_2d, solution, k):
    if dll_2d.main_header.right is dll_2d.main_header:
        return solution
    c = dll_2d.find_smallest_header()
    dll_2d.cover_column(c)
    r = c.down
    while r is not c:
        o_k = r
        if len(solution) <= k:
            solution.append(dll_2d.row_bits_from_element(r))
        else:
            solution[k] = dll_2d.row_bits_from_element(r)
        j = r.right
        while j is not r:
            dll_2d.cover_column(j)
            j = j.right
        solution = search(dll_2d, solution, k+1)
        # Lines below added in for early termination
        # - - -
        if dll_2d.main_header.right is dll_2d.main_header:
            return solution
        # - - -
        c = r.header
        r = o_k
        j = r.left
        while j is not r:
            dll_2d.uncover_column(j)
            j = j.left
        r = r.down
    dll_2d.uncover_column(c)
    return solution


def generate_starting_matrix(num_elements, polyomino_size):
    """Generate a matrix representing the exact cover problem of a given array and polyomino size"""
    # TODO - this generated array is TOO LARGE!
    # Can it be evaluated in another way? Can the DLL be directly generated?
    result = []
    for bits in itertools.combinations(range(num_elements), polyomino_size):
        row = np.zeros(num_elements, dtype=int)
        row[list(bits)] = 1
        result.append(row)
    return np.asarray(result)


def display(array, solution):
    fig = plt.figure()
    ax = fig.gca()
    ax.set_xticks(np.arange(start=0, stop=array[0]+1, step=1))
    ax.set_yticks(np.arange(start=0, stop=array[1]+1, step=1))
    plt.grid(True)
    colours = ["black", "blue", "orange", "green", "red"]
    for r, row in enumerate(solution):
        color = colours[np.mod(r, len(colours))]
        for e, element in enumerate(row):
            if element == 1:
                x, y = np.divmod(e, array[0])
                plt.fill_between([x, x+1], y, y-1, color=color)
    plt.show()

def main():
    array_dimensions = (10, 12)
    array = np.zeros(array_dimensions, dtype=int)
    polyomino_size = 10
    assert np.mod(array.size, polyomino_size) == 0
    print(f"Creating solution matrix for:\n    Array: {array.shape}\n    Polyomino: {polyomino_size}")
    solution_matrix = generate_starting_matrix(array.size, polyomino_size)
    print("Generating initial 2D doubly linked list")
    dll_2d = DLL_2D(solution_matrix)
    print("Solving...")
    solution = []
    solution = search(dll_2d, solution, k=0)
    print(f"Solution: {solution}")
    display(array_dimensions, solution)


main()
