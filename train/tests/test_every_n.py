from types import SimpleNamespace
from typing import Any, cast

from ..src.callbacks.every_n import EveryNCallback


def _trainer(
    *,
    global_step: int = 0,
    max_steps: int = 10,
    current_epoch: int = 0,
    max_epochs: int = 5,
) -> Any:
    return cast(
        Any,
        SimpleNamespace(
            global_step=global_step,
            max_steps=max_steps,
            current_epoch=current_epoch,
            max_epochs=max_epochs,
        ),
    )


def test_default_run_is_false_and_does_not_crash() -> None:
    callback = EveryNCallback()
    assert callback.run(_trainer()) is False


def test_n_steps_respects_first_and_last_flags() -> None:
    callback = EveryNCallback(n_steps=2, first=False, last=False)
    assert callback.run(_trainer(global_step=0)) is False
    assert callback.run(_trainer(global_step=2)) is True
    assert callback.run(_trainer(global_step=10, max_steps=10)) is False


def test_n_epochs_respects_first_and_last_flags() -> None:
    callback = EveryNCallback(n_epochs=2, first=False, last=False)
    assert callback.run(_trainer(current_epoch=0)) is False
    assert callback.run(_trainer(current_epoch=2)) is True
    assert callback.run(_trainer(current_epoch=6, max_epochs=6)) is False


def test_n_evals_int_schedule() -> None:
    callback = EveryNCallback(n_evals=2, first=False)
    assert callback.run(_trainer()) is False
    callback.on_validation_end(_trainer(), cast(Any, None))
    assert callback.run(_trainer()) is False
    callback.on_validation_end(_trainer(), cast(Any, None))
    assert callback.run(_trainer()) is True


def test_n_evals_list_schedule() -> None:
    callback = EveryNCallback(n_evals=[1, 3])
    assert callback.run(_trainer()) is False
    callback.on_validation_end(_trainer(), cast(Any, None))
    assert callback.run(_trainer()) is True
    callback.eval_count = 3
    assert callback.run(_trainer()) is True


def test_state_dict_roundtrip_and_sanity_reset() -> None:
    callback = EveryNCallback()
    callback.eval_count = 4
    state = callback.state_dict()
    assert state == {"eval_count": 4}

    restored = EveryNCallback()
    restored.load_state_dict(state)
    assert restored.eval_count == 4

    restored.on_sanity_check_end(_trainer(), cast(Any, None))
    assert restored.eval_count == 0
