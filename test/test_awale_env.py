import pytest
from awale_env import Awale


@pytest.fixture
def game():
    return Awale()


def test_initial_state(game):
    _extracted_from_test_reset_2(game)


def test_is_valid_move(game):
    assert game._is_valid_move(0)
    assert game._is_valid_move(5)
    assert not game._is_valid_move(6)
    assert not game._is_valid_move(11)
    game.current_player = 1
    assert game._is_valid_move(6)
    assert game._is_valid_move(11)
    assert not game._is_valid_move(0)
    assert not game._is_valid_move(5)
    game.state = [0] * 12
    assert not game._is_valid_move(0)


def test_capture(game):
    game.state = [4, 4, 4, 4, 4, 3, 2, 2, 3, 2, 4, 4]
    game.scores = [0, 0]
    captured = game._capture_seeds(9)
    assert game.state == [4, 4, 4, 4, 4, 3, 0, 0, 0, 0, 4, 4]
    assert captured == 9


def test_step_valid_move(game):
    state, reward, done, info = game.step(0)
    assert state == [0, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4]
    assert not done
    assert game.current_player == 1


def test_step_invalid_move(game):
    state, reward, done, info = game.step(6)
    assert state == [4] * 12
    assert reward == -1
    assert not done
    assert info == {"invalid": True}
    assert game.current_player == 0


def test_capture_after_move(game):
    game.state = [4, 4, 4, 4, 4, 3, 2, 2, 3, 2, 4, 4]
    state, reward, done, info = game.step(3)
    assert state == [4, 4, 4, 0, 5, 4, 0, 0, 3, 2, 4, 4]
    assert game.scores == [6, 0]


def test_game_end_by_score(game):
    game.scores = [24, 23]
    game.state = [
        1,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
    ]  # Ensure there's at least one seed to move
    game.current_player = 0  # Ensure it's player 0's turn
    state, reward, done, info = game.step(0)
    assert done
    assert reward > 99
    assert info["winner"] == 0


def test_game_end_by_empty_side(game):
    game.state = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]  # Player 0's side is empty
    game.current_player = 0
    state, reward, done, info = game.step(5)  # Player 0 tries to make a move
    assert done
    assert info["winner"] == 1


def test_reset(game):
    game.step(0)
    game.reset()
    _extracted_from_test_reset_2(game)


def _extracted_from_test_reset_2(game):
    assert game.state == [4] * 12
    assert game.scores == [0, 0]
    assert game.current_player == 0


def test_render(game):
    # This is a simple test to ensure render method runs without error
    try:
        game.render()
    except Exception as e:
        pytest.fail(f"render() raised {type(e).__name__} unexpectedly!")
