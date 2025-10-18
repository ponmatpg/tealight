from src.libvalue import Value

def test_basic():
    x = Value(10)
    y = Value(5.0)
    z = x + y  * y + x / y 
    print(f"z={z.describe()}")
    print(f"z={z.describe()}")

if __name__ == "__main__":
    test_basic()
    print("All tests passed!")