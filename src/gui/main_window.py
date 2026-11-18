import threading
import time

import customtkinter as ctk
import cv2
import queue
import threading
import time

from src.camera.camera import Camera
from src.model.predictor import GesturePredictor
from src.controller.gesture_stabilizer import GestureStabilizer
from src.controller.game_controller import GameController

from src.gui.camera_view import CameraView
from src.gui.performance_view import PerformanceView

class MainWindow(ctk.CTk):
    """Main GUI for the gesture-controlled game."""
    
    KEY_MAPPING = {
        "jump": "UP",
        "roll": "DOWN",
        "left": "LEFT",
        "right": "RIGHT"
    }

    def __init__(self):
        super().__init__()

        self.last_error = None

        self.performance_window = None

        # Window configuration.
        self.title("Gesture-Controlled Game | Deep Learning")
        self.geometry("1180x800")
        self.minsize(1000, 680)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Recognition state.
        self.is_running = False
        self.stop_event = threading.Event()
        self.worker_thread = None

        self.ui_queue = queue.Queue()
        self.after(30, self._process_ui_queue)

        # Recognition settings.
        self.confidence_threshold = 0.40
        self.game_control_enabled = False

        self._build_interface()

        self.protocol("WM_DELETE_WINDOW", self.on_close)


    def _publish_ui_event(self, event):
        """Send an event from the worker thread to the GUI thread."""
        self.ui_queue.put(event)


    def _process_ui_queue(self):
        """Process worker events safely in the main GUI thread."""

        try:
            while True:
                event = self.ui_queue.get_nowait()
                event_type = event[0]

                if event_type == "status":
                    self._set_status(event[1], event[2])

                elif event_type == "dashboard":
                    self._update_dashboard(*event[1:])

                elif event_type == "stopped":
                    self._on_recognition_stopped()

        except queue.Empty:
            pass

        if self.winfo_exists():
            self.after(30, self._process_ui_queue)


    def open_performance_dashboard(self):
        """Open the model performance dashboard."""

        # If the window already exists, bring it to front.
        if (
            self.performance_window is not None
            and self.performance_window.winfo_exists()
        ):
            self.performance_window.lift()
            self.performance_window.focus_force()
            return

        self.performance_window = PerformanceView(self)

        self.performance_window.protocol(
            "WM_DELETE_WINDOW",
            self._close_performance_dashboard
        )


    def _close_performance_dashboard(self):
        """Close the performance dashboard."""

        if (
            self.performance_window is not None
            and self.performance_window.winfo_exists()
        ):
            self.performance_window.destroy()

        self.performance_window = None
        

    def _build_interface(self):
        """Create the main application layout."""

        # Main container.
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(1, weight=1)

        # Header.
        header = ctk.CTkFrame(self, corner_radius=10)
        header.grid(
            row=0,
            column=0,
            columnspan=2,
            padx=15,
            pady=(15, 10),
            sticky="ew"
        )

        title = ctk.CTkLabel(
            header,
            text="GESTURE CONTROL SYSTEM",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title.pack(padx=20, pady=(12, 2))

        subtitle = ctk.CTkLabel(
            header,
            text="Real-Time Hand Gesture Recognition with MobileNetV2",
            text_color="gray"
        )
        subtitle.pack(padx=20, pady=(0, 12))

        # Left panel: webcam.
        left_panel = ctk.CTkFrame(self, corner_radius=10)
        left_panel.grid(
            row=1,
            column=0,
            padx=(15, 7),
            pady=5,
            sticky="nsew"
        )
        left_panel.grid_columnconfigure(0, weight=1)
        left_panel.grid_rowconfigure(1, weight=1)

        camera_title = ctk.CTkLabel(
            left_panel,
            text="WEBCAM",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        camera_title.grid(row=0, column=0, padx=10, pady=10)

        self.camera_view = CameraView(
            left_panel,
            width=480,
            height=360
        )

        self.camera_view.grid(
            row=1,
            column=0,
            padx=10,
            pady=5,
            sticky="nsew"
        )

        # Right panel: information and settings.
        right_panel = ctk.CTkScrollableFrame(
            self,
            corner_radius=10
        )

        right_panel.grid(
            row=1,
            column=1,
            padx=(7, 15),
            pady=5,
            sticky="nsew"
        )

        right_panel.grid_columnconfigure(0, weight=1)

        # System status.
        self._add_section_title(right_panel, "SYSTEM STATUS", 0)

        self.status_label = ctk.CTkLabel(
            right_panel,
            text="● Ready",
            text_color="#22C55E",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.status_label.grid(
            row=1,
            column=0,
            padx=15,
            pady=(5, 15),
            sticky="w"
        )

        # Gesture information.
        self._add_section_title(right_panel, "GESTURE RECOGNITION", 2)

        self.gesture_label = self._add_info(
            right_panel, "Detected gesture", "idle", 3
        )

        self.confidence_label = self._add_info(
            right_panel, "Confidence", "--", 4
        )

        self.latency_label = self._add_info(
            right_panel, "Inference time", "--", 5
        )

        self.command_label = self._add_info(
            right_panel, "Keyboard command", "None", 6
        )

        # Recognition settings.
        self._add_section_title(right_panel, "SETTINGS", 7)

        self.threshold_label = ctk.CTkLabel(
            right_panel,
            text="Confidence threshold: 40%"
        )
        self.threshold_label.grid(
            row=8,
            column=0,
            padx=15,
            pady=(8, 2),
            sticky="w"
        )

        self.threshold_slider = ctk.CTkSlider(
            right_panel,
            from_=0.30,
            to=0.95,
            number_of_steps=65,
            command=self._change_threshold
        )
        self.threshold_slider.set(self.confidence_threshold)
        self.threshold_slider.grid(
            row=9,
            column=0,
            padx=15,
            pady=(2, 12),
            sticky="ew"
        )

        self.game_control_switch = ctk.CTkSwitch(
            right_panel,
            text="Enable game keyboard control",
            command=self._toggle_game_control
        )
        self.game_control_switch.grid(
            row=10,
            column=0,
            padx=15,
            pady=10,
            sticky="w"
        )

        # Start and stop buttons.
        self.start_button = ctk.CTkButton(
            right_panel,
            text="START RECOGNITION",
            height=40,
            command=self.start_recognition
        )
        self.start_button.grid(
            row=11,
            column=0,
            padx=15,
            pady=(15, 5),
            sticky="ew"
        )


        self.stop_button = ctk.CTkButton(
            right_panel,
            text="STOP RECOGNITION",
            height=40,
            fg_color="#DC2626",
            hover_color="#B91C1C",
            command=self.stop_recognition,
            state="disabled"
        )
        self.stop_button.grid(
            row=12,
            column=0,
            padx=15,
            pady=(5, 5),
            sticky="ew"
        )

        # Model performance dashboard.
        self.performance_button = ctk.CTkButton(
            right_panel,
            text="MODEL PERFORMANCE",
            command=self.open_performance_dashboard
        )
        self.performance_button.grid(
            row=13,
            column=0,
            padx=15,
            pady=10,
            sticky="ew"
        )

        # Footer.
        footer = ctk.CTkLabel(
            self,
            text="CNN Gesture Recognition Project | MobileNetV2 | PyTorch",
            text_color="gray"
        )
        footer.grid(
            row=2,
            column=0,
            columnspan=2,
            padx=10,
            pady=10
        )

    def _add_section_title(self, parent, title, row):
        label = ctk.CTkLabel(
            parent,
            text=title,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        label.grid(
            row=row,
            column=0,
            padx=15,
            pady=(12, 3),
            sticky="w"
        )

    def _add_info(self, parent, title, initial_value, row):
        container = ctk.CTkFrame(parent, fg_color="transparent")
        container.grid(
            row=row,
            column=0,
            padx=15,
            pady=4,
            sticky="ew"
        )
        container.grid_columnconfigure(1, weight=1)

        name_label = ctk.CTkLabel(
            container,
            text=title,
            text_color="gray"
        )
        name_label.grid(row=0, column=0, sticky="w")

        value_label = ctk.CTkLabel(
            container,
            text=initial_value,
            font=ctk.CTkFont(weight="bold")
        )
        value_label.grid(row=0, column=1, padx=(8, 0), sticky="e")

        return value_label

    def _change_threshold(self, value):
        """Update the minimum confidence threshold."""

        self.confidence_threshold = float(value)

        self.threshold_label.configure(
            text=f"Confidence threshold: {self.confidence_threshold:.0%}"
        )

    def _toggle_game_control(self):
        """Enable or disable actual keyboard commands."""

        self.game_control_enabled = bool(
            self.game_control_switch.get()
        )

        if self.game_control_enabled:
            self._set_status("Game keyboard control enabled", "#22C55E")
            print("[GAME CONTROL] ENABLED")
        else:
            self._set_status("Game keyboard control disabled", "#F59E0B")
            print("[GAME CONTROL] DISABLED")

    def _set_status(self, message, color="white"):
        self.status_label.configure(
            text=f"● {message}",
            text_color=color
        )

    def start_recognition(self):
        """Start the recognition worker thread."""

        if self.worker_thread and self.worker_thread.is_alive():
            return

        self.stop_event.clear()
        self.last_error = None
        self.is_running = True

        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")

        self._set_status("Starting recognition...", "#F59E0B")
        self.camera_view.show_message("Starting webcam...")

        self.worker_thread = threading.Thread(
            target=self._recognition_loop,
            daemon=True
        )
        self.worker_thread.start()

    def stop_recognition(self):
        """Stop real-time gesture recognition."""
        if not self.is_running:
            return

        self.stop_event.set()

        self.stop_button.configure(state="disabled")
        self._set_status("Stopping recognition...", "#F59E0B")
        
    def _recognition_loop(self):
        """Capture frames, predict gestures and control the game."""

        camera = None

        try:
            self._publish_ui_event((
                "status",
                "Loading model...",
                "#F59E0B"
            ))

            self._publish_ui_event((
                "status",
                "Recognition active",
                "#22C55E"
            ))

            predictor = GesturePredictor()

            stabilizer = GestureStabilizer(confidence_threshold=self.confidence_threshold)

            # The controller is always configured to send real keyboard commands.
            # The GUI switch determines whether commands are actually executed.
            controller = GameController(dry_run=False)

            camera = Camera(index=0, width=640, height=480)

            inference_interval = 2
            frame_count = 0

            last_result = {
                "gesture": "idle",
                "confidence": 0.0,
                "inference_time_ms": 0.0
            }

            last_stable_result = {
                "stable_gesture": "idle",
                "action": None
            }

            last_ui_update = 0.0

            self.after(
                0,
                self._set_status,
                "Recognition active",
                "#22C55E"
            )



            while not self.stop_event.is_set():

                success, frame = camera.read()

                if not success:
                    raise RuntimeError("Unable to read webcam frame.")

                # Extract the centered hand region.
                roi, (x1, y1, x2, y2) = predictor.extract_roi(frame)

                # Run CNN inference every 2 frames.
                if frame_count % inference_interval == 0:

                    last_result = predictor.predict(roi)

                    gesture = last_result["gesture"]
                    confidence = last_result["confidence"]

                    # Stabilize the prediction.
                    stabilizer.confidence_threshold = (
                        self.confidence_threshold
                    )

                    last_stable_result = stabilizer.update(
                        gesture,
                        confidence
                    )

                    action = last_stable_result["action"]

                    # Execute only a new action event.
                    if action is not None:

                        if self.game_control_enabled:
                            key = controller.execute(action)
                            print(f"[GAME CONTROL] {action} -> {key}")

                        else:
                            print(
                                f"[GAME CONTROL DISABLED] "
                                f"Ignored action: {action}"
                            )

                frame_count += 1

                gesture = last_result["gesture"]
                confidence = last_result["confidence"]
                latency = last_result["inference_time_ms"]

                stable_gesture = last_stable_result["stable_gesture"]

                # Draw the ROI.
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Prediction: {gesture}",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Stable: {stable_gesture}",
                    (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )

                # Limit GUI updates to approximately 20 FPS.
                current_time = time.perf_counter()

                if current_time - last_ui_update >= 0.05:

                    command = self.KEY_MAPPING.get(
                        stable_gesture,
                        "None"
                    )

                    self._publish_ui_event((
                        "dashboard",
                        frame.copy(),
                        gesture,
                        confidence,
                        stable_gesture,
                        latency,
                        command
                    ))

                    last_ui_update = current_time


        except Exception as error:
            self.last_error = str(error)

            if "camera" in self.last_error.lower() or \
            "webcam" in self.last_error.lower():
                status = "Camera unavailable"
            else:
                status = "Recognition error"

            self._publish_ui_event((
                "status",
                status,
                "#EF4444"
            ))

        finally:
            if camera is not None:
                camera.release()

            self.is_running = False

            self._publish_ui_event(("stopped",))
    
    def _update_dashboard(
        self,
        frame,
        gesture,
        confidence,
        stable_gesture,
        latency,
        command
    ):
        """Update GUI widgets from the main UI thread."""

        self.camera_view.update_frame(frame)

        self.gesture_label.configure(
            text=f"{stable_gesture} (raw: {gesture})"
        )

        self.confidence_label.configure(
            text=f"{confidence:.1%}"
        )

        self.latency_label.configure(
            text=f"{latency:.1f} ms"
        )

        self.command_label.configure(
            text=command
        )

    def _on_recognition_stopped(self):
        """Update the interface after recognition stops."""

        self.is_running = False

        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")

        if self.last_error:
            self.camera_view.show_message(self.last_error)
            self._set_status("Recognition error", "#EF4444")
            return

        self.camera_view.show_message("Webcam inactive")
        self._set_status("Ready", "#22C55E")

        self.gesture_label.configure(text="idle")
        self.confidence_label.configure(text="--")
        self.latency_label.configure(text="--")
        self.command_label.configure(text="None")

    def on_close(self):
        """Stop recognition before closing the application."""

        self.stop_event.set()
        self.destroy()


if __name__ == "__main__":
    app = MainWindow()
    app.mainloop()
