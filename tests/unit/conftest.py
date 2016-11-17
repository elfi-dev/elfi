import pytest
import elfi.env as env

# http://stackoverflow.com/questions/22627659/run-code-before-and-after-each-test-in-py-test
@pytest.yield_fixture(autouse=True)
def run_around_tests():
    # Code that will run before your test, for example:

    # A test function will be run at this point
    yield
    # Code that will run after your test, for example:

    env.set(inference_tasks = {})