
from src.controller.game_controller import GameController


def test_game_controller():
    controller = GameController(dry_run=True)

    test_gestures = [
        "jump",
        "roll",
        "left",
        "right",
        "idle"
    ]

    print("\nGame controller test")
    print("--------------------")

    for gesture in test_gestures:
        key = controller.execute(gesture)
        print(f"Result: {gesture} -> {key}")

    print("\nTest completed successfully.")


if __name__ == "__main__":
    test_game_controller()
