class FixedWindow:
    def __init__(self, alignment, window_size):
        self.alignment = alignment
        self.window_size = window_size

    def fit_transform(self, file_name):
        with open(file_name, "r") as runtime:
            line_count = len(runtime.read().rstrip('\n').split(' '))
        segmentation = []
        curr_line = 0
        for line in range(self.alignment, line_count, self.window_size):
            segmentation.append(line)
            curr_line = line
        if curr_line < line_count:
            segmentation.append(line_count)
        return segmentation