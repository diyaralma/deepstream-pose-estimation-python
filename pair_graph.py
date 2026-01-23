from typing import List, Tuple


class PairGraph:
    """
    A bipartite graph matching structure that maintains pairs between rows and columns.
    """

    def __init__(self, nrows: int, ncols: int):
        """
        Initializes a PairGraph with the specified number of rows and columns.

        Args:
            nrows: Number of rows in the graph
            ncols: Number of columns in the graph
        """
        # Store dimensions as read-only attributes (const in C++)
        object.__setattr__(self, 'nrows', nrows)
        object.__setattr__(self, 'ncols', ncols)

        # Initialize vectors with -1 (indicating no pair)
        self._rows: List[int] = [-1] * nrows
        self._cols: List[int] = [-1] * ncols

    def __setattr__(self, name: str, value) -> None:
        """
        Prevent modification of nrows and ncols after initialization (const behavior).
        """
        if name in ('nrows', 'ncols') and hasattr(self, name):
            raise AttributeError(f"Cannot modify const attribute '{name}'")
        super().__setattr__(name, value)

    def colForRow(self, row: int) -> int:
        """
        Returns the column index of the pair matching this row

        Args:
            row: The row index

        Returns:
            The column index paired with this row
        """
        return self._rows[row]

    def rowForCol(self, col: int) -> int:
        """
        Returns the row index of the pair matching this column

        Args:
            col: The column index

        Returns:
            The row index paired with this column
        """
        return self._cols[col]

    def set(self, row: int, col: int) -> None:
        """
        Creates a pair between row and col

        Args:
            row: The row index
            col: The column index
        """
        self._rows[row] = col
        self._cols[col] = row

    def isRowSet(self, row: int) -> bool:
        """
        Checks if a row has a pair

        Args:
            row: The row index

        Returns:
            True if the row is paired, False otherwise
        """
        return self._rows[row] >= 0

    def isColSet(self, col: int) -> bool:
        """
        Checks if a column has a pair

        Args:
            col: The column index

        Returns:
            True if the column is paired, False otherwise
        """
        return self._cols[col] >= 0

    def isPair(self, row: int, col: int) -> bool:
        """
        Checks if row and col form a pair

        Args:
            row: The row index
            col: The column index

        Returns:
            True if row and col are paired together, False otherwise
        """
        return self._rows[row] == col

    def reset(self, row: int, col: int) -> None:
        """
        Clears pair between row and col

        Args:
            row: The row index
            col: The column index
        """
        self._rows[row] = -1
        self._cols[col] = -1

    def clear(self) -> None:
        """
        Clears all pairs in graph
        """
        for i in range(self.nrows):
            self._rows[i] = -1
        for j in range(self.ncols):
            self._cols[j] = -1

    def numPairs(self) -> int:
        """
        Counts the number of pairs in the graph

        Returns:
            The number of active pairs
        """
        count = 0
        for i in range(self.nrows):
            if self._rows[i] >= 0:
                count += 1
        return count

    def pairs(self) -> List[Tuple[int, int]]:
        """
        Returns all pairs in the graph

        Returns:
            A list of tuples representing (row, col) pairs
        """
        p: List[Tuple[int, int]] = [None] * self.numPairs()
        count = 0
        for i in range(self.nrows):
            if self.isRowSet(i):
                p[count] = (i, self.colForRow(i))
                count += 1
        return p
