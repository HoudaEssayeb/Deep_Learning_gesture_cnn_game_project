
import cv2
from src.camera.camera import Camera


def main():
    camera = Camera(index=0, width=640, height=480)
    window_name = "Camera Test"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Frame capture failed")
                break

            print("Frame received:", frame.shape)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(30) & 0xFF

            if key == ord("q"):
                break

            if cv2.getWindowProperty(
                window_name, cv2.WND_PROP_VISIBLE
            ) < 1:
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Camera released.")


if __name__ == "__main__":
    main()
