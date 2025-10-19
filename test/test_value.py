from src.libvalue import Value

def test_basic():
    x = Value(2)
    y = Value(3)
    z = x + y * y + x * y + x ** y
    z.backward()
    print(f"{z.describe()}")
    print(f"{x.describe()}")
    print(f"{y.describe()}")


if __name__ == "__main__":
    test_basic()
    print("All tests passed!")