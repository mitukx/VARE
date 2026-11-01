import vare_run003_fixture_dependency_that_does_not_exist  # noqa: F401


def test_unreachable() -> None:
    raise AssertionError("module import should fail before collection")
