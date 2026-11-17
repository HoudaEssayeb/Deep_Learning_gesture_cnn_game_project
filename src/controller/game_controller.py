
import pyautogui


class GameController:
    """
    Sends keyboard commands to control the game
    according to the stabilized gesture.
    """

    KEY_MAPPING = {
        "jump": "up",
        "roll": "down",
        "left": "left",
        "right": "right"
    }

    VALID_GESTURES = {
        "idle",
        "jump",
        "roll",
        "left",
        "right"
    }

    def __init__(self, dry_run=False):
        """
        Args:
            dry_run: If True, simulate commands without
                     sending actual keyboard presses.
        """
        self.dry_run = dry_run
        self.active_key = None

        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.05

        print("Game controller initialized.")
        print("Dry run:", self.dry_run)

    def execute(self, gesture):
        """
        Execute the keyboard command associated with a gesture.

        Args:
            gesture: Stabilized gesture or action event.

        Returns:
            The keyboard key sent, or None.
        """

        if gesture not in self.VALID_GESTURES:
            raise ValueError(f"Unknown gesture: {gesture}")

        if gesture == "idle":
            return None

        key = self.KEY_MAPPING[gesture]

        if self.dry_run:
            print(f"[DRY RUN] Gesture: {gesture} -> Key: {key}")
        else:
            pyautogui.press(key)
            print(f"Gesture: {gesture} -> Key pressed: {key}")

        return key

    def release_all(self):
        """Release any key that may still be held."""

        if self.active_key is not None:
            try:
                if not self.dry_run:
                    pyautogui.keyUp(self.active_key)
            finally:
                self.active_key = None
