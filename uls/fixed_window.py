class FixedWindow:
    def __init__(self, alignment, window_size):
        self.alignment = alignment
        self.window_size = window_size

    def fit_transform(self, file_name):
        segmentation = []
        with open(file_name, "r") as runtime:
            for line in runtime:
                line_segmentation = []
                line_count = len(line.split(' '))
                if line_count <= self.alignment:
                    return []
                sum_of_windows = 0
                line_segmentation.append([0]*self.alignment)
                window = [0]*self.window_size
                nbr_of_windows = (line_count - self.alignment)/self.window_size
                last_window_len = (line_count - self.alignment)%self.window_size
                line_segmentation.append(int(nbr_of_windows)*window)
                line_segmentation.append(last_window_len*[0])
                for seg in line_segmentation:
                    sum_of_windows += len(seg)
                if line_count!= sum_of_windows:
                    print("self.alignment ", self.alignment)
                    print("self.window_size ", self.window_size)
                    print("line count ", line_count)
                    print("sum_of_windows ", sum_of_windows)
                    print(line)
                    print(line_segmentation)
                assert(line_count == sum_of_windows)
                segmentation.append(line_segmentation)
            
        

        return segmentation