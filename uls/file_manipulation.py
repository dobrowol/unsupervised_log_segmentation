from pathlib import Path


class FileSegmentation:
    def __init__(self):
        pass

    def segment_text_file(self, file_name, segmentation_method):
        segmentation_method.fit([file_name])
        return segmentation_method.transform([file_name])[0]  
    
    def segment_file(self, file_name, segmentation_method):
        self.file_name = file_name

        segmentation_method.fit([file_name])
        return segmentation_method.transform([file_name])[0]  