"""Tests for the shared Rich progress bar factory."""

from rich.progress import Progress

from utils.progress import create_progress


class TestCreateProgress:
    def test_returns_progress_instance(self):
        prog = create_progress()
        assert isinstance(prog, Progress)

    def test_two_calls_independent(self):
        assert create_progress() is not create_progress()

    def test_add_task_and_advance(self):
        prog = create_progress()
        task_id = prog.add_task("working", total=4)
        prog.advance(task_id, 2)
        task = prog.tasks[task_id]
        assert task.completed == 4 // 2
        prog.stop()
