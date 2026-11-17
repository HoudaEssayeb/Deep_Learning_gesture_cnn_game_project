
from src.controller.gesture_stabilizer import GestureStabilizer

stabilizer = GestureStabilizer()

test_predictions = [
    ("jump", 0.92),
    ("jump", 0.87),
    ("jump", 0.90),
    ("jump", 0.85),
    ("right", 0.75),
    ("jump", 0.91),
    ("jump", 0.93),
    ("idle", 0.95),
    ("idle", 0.96),
    ("idle", 0.94),
    ("idle", 0.91),
]

for gesture, confidence in test_predictions:
    result = stabilizer.update(gesture, confidence)

    print(
        f"Prediction: {gesture:>5} | "
        f"Stable: {result['stable_gesture']:>5} | "
        f"Action: {str(result['action']):>5} | "
        f"Stable: {result['is_stable']}"
    )
