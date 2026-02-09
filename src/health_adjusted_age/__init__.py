"""Main module for my-python-project."""


def hello(name: str = "World") -> str:
    """Return a greeting message.

    Args:
        name: The name to greet

    Returns:
        A greeting message
    """
    return f"Hello, {name}!"


if __name__ == "__main__":
    print(hello())
