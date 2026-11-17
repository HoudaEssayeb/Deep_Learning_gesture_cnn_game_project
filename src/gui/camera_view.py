
import customtkinter as ctk
import cv2

from PIL import Image


class CameraView(ctk.CTkFrame):
    """Displays the webcam stream inside the GUI."""

    def __init__(self, master, width=640, height=480):
        super().__init__(master)

        self.width = width
        self.height = height

        self.image_label = ctk.CTkLabel(
            self,
            text="Webcam inactive",
            width=self.width,
            height=self.height,
            fg_color=("gray90", "gray15"),
            corner_radius=8
        )
        self.image_label.pack(
            fill="both",
            expand=True,
            padx=5,
            pady=5
        )

        # Keep a reference to the current image.
        self.current_image = None

    def update_frame(self, frame):
        """Convert and display an OpenCV BGR frame."""

        if frame is None:
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)

        image = image.resize(
            (self.width, self.height),
            Image.Resampling.LANCZOS
        )

        self.current_image = ctk.CTkImage(
            light_image=image,
            dark_image=image,
            size=(self.width, self.height)
        )

        self.image_label.configure(
            image=self.current_image,
            text=""
        )

    def show_message(self, message):
        """Display a message instead of the camera stream."""

        self.current_image = None

        self.image_label.configure(
            image=None,
            text=message
        )
