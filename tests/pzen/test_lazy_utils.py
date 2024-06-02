from pzen.lazy_utils import Lazy


def test_lazy():

    num_calls = 0

    def costly_func() -> float:
        nonlocal num_calls
        num_calls += 1
        return 123.456

    def example_usage(
        # Usage should in general allow both eager and lazy passing, i.e., `T | Lazy[T]`,
        # and the default value is expressed as a Lazy instance to avoid its evaluation at
        # function declaration time, different from just `arg: float = costly_func()`.
        arg: float | Lazy[float] = Lazy(lambda: costly_func()),
    ):
        # Using `Lazy.get` makes it convenient to work with `T | Lazy[T]`.
        assert Lazy.get(arg) == 123.456

    assert num_calls == 0

    example_usage()
    assert num_calls == 1

    example_usage()
    assert num_calls == 1
