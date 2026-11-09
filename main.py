import time
import cv2
import torch
import customtkinter as ctk
from PIL import Image
from torchvision import transforms
from src.model.cnn_model import build_model, CLASS_NAMES
from src.controller.game_controller import GameController

MODEL_PATH = "models/best_model.pth"
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.75

class GestureApp:
    def __init__(self):
        ctk.set_appearance_mode("dark")
        self.root = ctk.CTk()
        self.root.title("Gesture CNN — Sujet 21")
        self.root.geometry("1000x700")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(num_classes=len(CLASS_NAMES))
        checkpoint = torch.load(MODEL_PATH, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device).eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.controller = GameController()
        self.cap = None
        self.running = False
        self.last_gesture = "idle"
        self.frame_count = 0
        self.fps = 0.0
        self.fps_start = time.perf_counter()

        self.video_label = ctk.CTkLabel(self.root, text="Caméra arrêtée")
        self.video_label.pack(padx=10, pady=10, expand=True)

        self.info = ctk.CTkLabel(self.root, text="Geste : — | Confiance : — | Latence CNN : — | FPS : —")
        self.info.pack(pady=8)

        self.threshold = ctk.CTkSlider(self.root, from_=0.50, to=0.95, number_of_steps=45)
        self.threshold.set(CONFIDENCE_THRESHOLD)
        self.threshold.pack(fill="x", padx=40, pady=8)
        self.threshold_label = ctk.CTkLabel(self.root, text=f"Seuil de confiance : {CONFIDENCE_THRESHOLD:.2f}")
        self.threshold_label.pack()

        self.threshold.configure(command=self.on_threshold_change)
        buttons = ctk.CTkFrame(self.root)
        buttons.pack(pady=10)
        ctk.CTkButton(buttons, text="Démarrer", command=self.start).pack(side="left", padx=8)
        ctk.CTkButton(buttons, text="Arrêter", command=self.stop).pack(side="left", padx=8)
        ctk.CTkButton(buttons, text="Quitter", command=self.close).pack(side="left", padx=8)

        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def on_threshold_change(self, value):
        self.threshold_label.configure(text=f"Seuil de confiance : {float(value):.2f}")

    def start(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            self.info.configure(text="Erreur : webcam indisponible.")
            return
        self.running = True
        self.update_frame()

    def stop(self):
        self.running = False
        self.controller.release_all()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.video_label.configure(image=None, text="Caméra arrêtée")

    @torch.inference_mode()
    def update_frame(self):
        if not self.running or self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.root.after(20, self.update_frame)
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        t0 = time.perf_counter()
        logits = self.model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        latency_ms = (time.perf_counter() - t0) * 1000
        confidence, index = torch.max(probabilities, dim=0)
        gesture = CLASS_NAMES[int(index)]
        conf = float(confidence)

        if conf >= float(self.threshold.get()):
            self.controller.apply(gesture)
            self.last_gesture = gesture
        else:
            self.controller.apply("idle")
            gesture = "uncertain"

        self.frame_count += 1
        elapsed = time.perf_counter() - self.fps_start
        if elapsed >= 1:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_start = time.perf_counter()

        cv2.putText(frame, f"{gesture} ({conf:.1%})", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 220, 80), 2)
        cv2.putText(frame, f"CNN: {latency_ms:.1f} ms | FPS: {self.fps:.1f}", (15, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        display = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        display.thumbnail((900, 500))
        ctk_image = ctk.CTkImage(light_image=display, dark_image=display, size=display.size)
        self.video_label.configure(image=ctk_image, text="")
        self.video_label.image = ctk_image
        self.info.configure(text=f"Geste : {gesture} | Confiance : {conf:.1%} | Latence CNN : {latency_ms:.1f} ms | FPS : {self.fps:.1f}")
        self.root.after(1, self.update_frame)

    def close(self):
        self.stop()
        self.root.destroy()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    GestureApp().run()
