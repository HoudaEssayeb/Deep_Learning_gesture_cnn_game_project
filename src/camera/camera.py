import cv2


class Camera:
    """Handles webcam initialization, frame acquisition and release."""

    def __init__(self, index=0, width=640, height=480):
        self.index = index
        self.width = width
        self.height = height

        self.cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None
            raise RuntimeError(
                f"Unable to open camera with index {self.index}"
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read(self):
        """Capture and return one frame."""

        if self.cap is None or not self.cap.isOpened():
            return False, None

        success, frame = self.cap.read()
        
        print("Min:", frame.min(), "Max:", frame.max())
        print("Mean:", frame.mean())

        if not success or frame is None:
            return False, None

        return True, frame

    def release(self):
        """Release the webcam resource."""

        if self.cap is not None:
            self.cap.release()
            self.cap = None
