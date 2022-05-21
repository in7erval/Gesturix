import cv2


class FakeCamera:

    def __init__(self, image_fname):
        self.image = cv2.imread(image_fname)

    def isOpened(self):
        return True

    def read(self):
        return True, self.image

    def release(self):
        pass
