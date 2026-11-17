
from collections import Counter, deque
import time


class GestureStabilizer:
    """
    Stabilizes gesture predictions using a sliding window,
    a confidence threshold and an action cooldown.
    """

    VALID_GESTURES = {
        "idle",
        "jump",
        "left",
        "right",
        "roll"
    }

    def __init__(
        self,
        window_size=5,
        min_votes=4,
        confidence_threshold=0.70,
        cooldown=0.6
    ):
        if window_size <= 0:
            raise ValueError("window_size must be positive.")

        if not 1 <= min_votes <= window_size:
            raise ValueError("min_votes must be between 1 and window_size.")

        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1.")

        if cooldown < 0:
            raise ValueError("cooldown cannot be negative.")

        self.window_size = window_size
        self.min_votes = min_votes
        self.confidence_threshold = confidence_threshold
        self.cooldown = cooldown

        # Stores the last predictions.
        self.predictions = deque(maxlen=window_size)

        # Initial stable state.
        self.current_gesture = "idle"

        # Time of the last triggered action.
        self.last_action_time = float("-inf")

    def update(self, gesture, confidence):
        """
        Add a new prediction and return the stabilized result.

        Returns:
            dict: stable gesture, action event and confidence.
        """

        if gesture not in self.VALID_GESTURES:
            raise ValueError(f"Unknown gesture: {gesture}")

        if not 0 <= confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1.")

        # Low-confidence predictions are ignored as votes.
        if confidence >= self.confidence_threshold:
            self.predictions.append((gesture, confidence))
        else:
            self.predictions.append(None)

        result = {
            "stable_gesture": self.current_gesture,
            "action": None,
            "stable_confidence": None,
            "is_stable": False
        }

        # Wait until the prediction window is full.
        if len(self.predictions) < self.window_size:
            return result

        # Keep only valid, confident predictions.
        valid_predictions = [
            prediction
            for prediction in self.predictions
            if prediction is not None
        ]

        if not valid_predictions:
            return result

        # Count the occurrences of each gesture.
        gesture_counts = Counter(
            prediction[0]
            for prediction in valid_predictions
        )

        candidate, votes = gesture_counts.most_common(1)[0]

        # The gesture must receive enough votes.
        if votes < self.min_votes:
            return result

        # Calculate the average confidence for this gesture.
        candidate_confidence = sum(
            confidence
            for predicted_gesture, confidence in valid_predictions
            if predicted_gesture == candidate
        ) / votes

        previous_gesture = self.current_gesture

        # Update the stable state.
        self.current_gesture = candidate

        result["stable_gesture"] = self.current_gesture
        result["stable_confidence"] = candidate_confidence
        result["is_stable"] = True

        # Trigger an action only when entering a new non-idle gesture.
        if candidate != previous_gesture and candidate != "idle":
            current_time = time.monotonic()

            if current_time - self.last_action_time >= self.cooldown:
                result["action"] = candidate
                self.last_action_time = current_time

        return result

    def reset(self):
        """Reset the prediction window and stable state."""
        self.predictions.clear()
        self.current_gesture = "idle"
        self.last_action_time = float("-inf")
