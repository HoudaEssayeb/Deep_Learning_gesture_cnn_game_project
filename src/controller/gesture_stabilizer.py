
from collections import Counter, deque
import time


class GestureStabilizer:
    """
    Stabilizes gesture predictions and triggers actions.

    - First action: immediately after stabilization.
    - Same gesture: repeat every 5 seconds.
    - Different gesture: trigger immediately.
    - Idle: reset the repetition timer.
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
        window_size=3,
        min_votes=2,
        confidence_threshold=0.40,
        repeat_interval=5.0
    ):
        if window_size <= 0:
            raise ValueError("window_size must be positive.")

        if not 1 <= min_votes <= window_size:
            raise ValueError(
                "min_votes must be between 1 and window_size."
            )

        if not 0 <= confidence_threshold <= 1:
            raise ValueError(
                "confidence_threshold must be between 0 and 1."
            )

        if repeat_interval <= 0:
            raise ValueError("repeat_interval must be positive.")

        self.window_size = window_size
        self.min_votes = min_votes
        self.confidence_threshold = confidence_threshold
        self.repeat_interval = repeat_interval

        self.predictions = deque(maxlen=window_size)

        self.current_gesture = "idle"
        self.last_action_time = None

    def update(self, gesture, confidence):
        """
        Process a new prediction.

        Returns:
            dict: stable gesture, action event and confidence.
        """

        if gesture not in self.VALID_GESTURES:
            raise ValueError(f"Unknown gesture: {gesture}")

        if not 0 <= confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1.")

        # Add prediction to the sliding window.
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

        # Keep valid predictions.
        valid_predictions = [
            prediction
            for prediction in self.predictions
            if prediction is not None
        ]

        if not valid_predictions:
            return result

        # Count votes.
        gesture_counts = Counter(
            prediction[0]
            for prediction in valid_predictions
        )

        candidate, votes = gesture_counts.most_common(1)[0]

        if votes < self.min_votes:
            return result

        # Calculate average confidence.
        candidate_confidence = sum(
            conf
            for predicted_gesture, conf in valid_predictions
            if predicted_gesture == candidate
        ) / votes

        current_time = time.monotonic()

        result["stable_gesture"] = candidate
        result["stable_confidence"] = candidate_confidence
        result["is_stable"] = True

        # Idle resets the repetition timer.
        if candidate == "idle":
            self.current_gesture = "idle"
            self.last_action_time = None
            return result

        # A new gesture triggers immediately.
        if candidate != self.current_gesture:
            self.current_gesture = candidate
            result["action"] = candidate
            self.last_action_time = current_time
            return result

        # Repeat the same gesture after the configured interval.
        if (
            self.last_action_time is not None
            and current_time - self.last_action_time >= self.repeat_interval
        ):
            result["action"] = candidate
            self.last_action_time = current_time

        return result

    def reset(self):
        """Reset the prediction window and gesture state."""

        self.predictions.clear()
        self.current_gesture = "idle"
        self.last_action_time = None
