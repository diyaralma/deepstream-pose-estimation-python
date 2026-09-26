import math
from typing import List, Tuple, Any

# Type aliases for clarity
# The bug was here: fixed instead of making the aliases parameterizable.
Vec1D = List[float]
Vec2D = List[List[float]]
Vec3D = List[List[List[float]]]

class PairGraph:
    """
    PairGraph class to manage row-column pair assignments.
    Tracks which rows and columns have been set.
    """

    def __init__(self, nrows: int, ncols: int):
        self.nrows = nrows
        self.ncols = ncols
        # Store the assignments: row -> col mapping
        self.row_to_col = [-1] * nrows
        # Store the assignments: col -> row mapping
        self.col_to_row = [-1] * ncols

    def clear(self):
        """Clear all assignments"""
        self.row_to_col = [-1] * self.nrows
        self.col_to_row = [-1] * self.ncols

    def set(self, row: int, col: int):
        """Set a pair assignment at (row, col)"""
        self.row_to_col[row] = col
        self.col_to_row[col] = row

    def reset(self, row: int, col: int):
        """Reset/clear a pair assignment at (row, col)"""
        self.row_to_col[row] = -1
        self.col_to_row[col] = -1

    def isRowSet(self, row: int) -> bool:
        """Check if a row has an assignment"""
        return self.row_to_col[row] != -1

    def isColSet(self, col: int) -> bool:
        """Check if a column has an assignment"""
        return self.col_to_row[col] != -1

    def rowForCol(self, col: int) -> int:
        """Get the row assigned to a column"""
        return self.col_to_row[col]

    def colForRow(self, row: int) -> int:
        """Get the column assigned to a row"""
        return self.row_to_col[row]


class CoverTable:
    """
    CoverTable class to track which rows and columns are covered.
    """

    def __init__(self, nrows: int, ncols: int):
        self.nrows = nrows
        self.ncols = ncols
        # Track which rows are covered
        self.row_covered = [False] * nrows
        # Track which columns are covered
        self.col_covered = [False] * ncols

    def clear(self):
        """Clear all coverage"""
        self.row_covered = [False] * self.nrows
        self.col_covered = [False] * self.ncols

    def coverRow(self, row: int):
        """Cover a row"""
        self.row_covered[row] = True

    def coverCol(self, col: int):
        """Cover a column"""
        self.col_covered[col] = True

    def uncoverRow(self, row: int):
        """Uncover a row"""
        self.row_covered[row] = False

    def uncoverCol(self, col: int):
        """Uncover a column"""
        self.col_covered[col] = False

    def isRowCovered(self, row: int) -> bool:
        """Check if a row is covered"""
        return self.row_covered[row]

    def isColCovered(self, col: int) -> bool:
        """Check if a column is covered"""
        return self.col_covered[col]

    def isCovered(self, row: int, col: int) -> bool:
        """Check if a position (row, col) is covered"""
        return self.row_covered[row] or self.col_covered[col]


# Helper method to subtract the minimum row from cost_graph
# FIX: use plain 'Vec2D' instead of 'Vec2D[float]'
def subtract_minimum_row(cost_graph: Vec2D, nrows: int, ncols: int) -> None:
    for i in range(nrows):
        # Iterate the find the minimum
        min_val = cost_graph[i][0]
        for j in range(ncols):
            val = cost_graph[i][j]
            if val < min_val:
                min_val = val

        # Subtract the Minimum
        for j in range(ncols):
            cost_graph[i][j] -= min_val


# Helper method to subtract the minimum col from cost_graph
# FIX: plain 'Vec2D' instead of 'Vec2D[float]'
def subtract_minimum_column(cost_graph: Vec2D, nrows: int, ncols: int) -> None:
    for j in range(ncols):
        # Iterate and find the minimum
        min_val = cost_graph[0][j]
        for i in range(nrows):
            val = cost_graph[i][j]
            if val < min_val:
                min_val = val

        # Subtract the minimum
        for i in range(nrows):
            cost_graph[i][j] -= min_val


def munkresStep1(cost_graph: Vec2D, star_graph: PairGraph, nrows: int,
                 ncols: int) -> None:
    for i in range(nrows):
        for j in range(ncols):
            if not star_graph.isRowSet(i) and not star_graph.isColSet(j) and (cost_graph[i][j] == 0):
                star_graph.set(i, j)


# Exits if '1' is returned
def munkresStep2(star_graph: PairGraph, cover_table: CoverTable) -> bool:
    k = star_graph.nrows if star_graph.nrows < star_graph.ncols else star_graph.ncols
    count = 0
    for j in range(star_graph.ncols):
        if star_graph.isColSet(j):
            cover_table.coverCol(j)
            count += 1
    return count >= k


def munkresStep3(cost_graph: Vec2D, star_graph: PairGraph,
                 prime_graph: PairGraph, cover_table: CoverTable, p: List[int],
                 nrows: int, ncols: int) -> bool:
    for i in range(nrows):
        for j in range(ncols):
            if cost_graph[i][j] == 0 and not cover_table.isCovered(i, j):
                prime_graph.set(i, j)
                if star_graph.isRowSet(i):
                    cover_table.coverRow(i)
                    cover_table.uncoverCol(star_graph.colForRow(i))
                else:
                    p[0] = i
                    p[1] = j
                    return True
    return False


def munkresStep4(star_graph: PairGraph, prime_graph: PairGraph,
                 cover_table: CoverTable, p: List[int]) -> None:
    # This process should be repeated until no star is found in prime's column
    while star_graph.isColSet(p[1]):
        # First find and reset any star found in the prime's columns
        s = [star_graph.rowForCol(p[1]), p[1]]
        star_graph.reset(s[0], s[1])

        # Set this prime to a star
        star_graph.set(p[0], p[1])

        # Repeat the same process for prime in cleared star's row
        p[0] = s[0]
        p[1] = prime_graph.colForRow(s[0])

    star_graph.set(p[0], p[1])
    cover_table.clear()
    prime_graph.clear()


def munkresStep5(cost_graph: Vec2D, cover_table: CoverTable,
                 nrows: int, ncols: int) -> None:
    valid = False
    min_val = 0.0
    for i in range(nrows):
        for j in range(ncols):
            if not cover_table.isCovered(i, j):
                if not valid:
                    min_val = cost_graph[i][j]
                    valid = True
                elif cost_graph[i][j] < min_val:
                    min_val = cost_graph[i][j]

    for i in range(nrows):
        if cover_table.isRowCovered(i):
            for j in range(ncols):
                cost_graph[i][j] += min_val

    for j in range(ncols):
        if not cover_table.isColCovered(j):
            for i in range(nrows):
                cost_graph[i][j] -= min_val


def munkres_algorithm(cost_graph: Vec2D, star_graph: PairGraph, nrows: int,
                      ncols: int) -> None:
    prime_graph = PairGraph(nrows, ncols)
    cover_table = CoverTable(nrows, ncols)
    prime_graph.clear()
    cover_table.clear()
    star_graph.clear()

    step = 0
    if ncols >= nrows:
        subtract_minimum_row(cost_graph, nrows, ncols)
    if ncols > nrows:
        step = 1

    p = [0, 0]
    done = False
    while not done:
        if step == 0:
            subtract_minimum_column(cost_graph, nrows, ncols)
            step = 1

        if step == 1:
            munkresStep1(cost_graph, star_graph, nrows, ncols)
            step = 2

        if step == 2:
            if munkresStep2(star_graph, cover_table):
                done = True
                break
            step = 3

        if step == 3:
            if not munkresStep3(cost_graph, star_graph, prime_graph, cover_table, p,
                                nrows, ncols):
                step = 5
            else:
                step = 4

        if step == 4:
            munkresStep4(star_graph, prime_graph, cover_table, p)
            step = 2

        if step == 5:
            munkresStep5(cost_graph, cover_table, nrows, ncols)
            step = 3