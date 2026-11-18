
import pyautogui


class GameController:
    """
    Sends keyboard commands according to detected gestures.
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
        self.dry_run = dry_run

        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0

        print("Game controller initialized.")
        print("Dry run:", self.dry_run)

    def execute(self, gesture):
        """
        Execute one keyboard action.
        """

        if gesture not in self.VALID_GESTURES:
            raise ValueError(f"Unknown gesture: {gesture}")

        if gesture == "idle":
            return None

        key = self.KEY_MAPPING[gesture]

        if self.dry_run:
            print(f"[DRY RUN] {gesture} -> {key}")
        else:
            pyautogui.press(key)
            print(f"Gesture: {gesture} -> Key pressed: {key}")

        return key

    def release_all(self):
        """No continuously held key is used."""

        pass
