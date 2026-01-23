class CoverTable:
    def __init__(self, nrows, ncols):
        self.nrows = nrows
        self.ncols = ncols
        self.rows = [False] * nrows
        self.cols = [False] * ncols

    def coverRow(self, row):
        self.rows[row] = True

    def coverCol(self, col):
        self.cols[col] = True

    def uncoverRow(self, row):
        self.rows[row] = False

    def uncoverCol(self, col):
        self.cols[col] = False

    def isCovered(self, row, col):
        return self.rows[row] or self.cols[col]

    def isRowCovered(self, row):
        return self.rows[row]

    def isColCovered(self, col):
        return self.cols[col]

    def clear(self):
        for i in range(self.nrows):
            self.uncoverRow(i)
        for j in range(self.ncols):
            self.uncoverCol(j)