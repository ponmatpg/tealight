from src import libexample

def test_calculator():
    calc = libexample.Calculator(10)
    assert calc.get_value() == 10
    
    result = calc.add(5)
    assert result == 15
    assert calc.get_value() == 15
    
    desc = calc.describe()
    assert "15" in desc
    print(f"Calculator description: {desc}")

def test_multiply():
    result = libexample.multiply(6, 7)
    assert result == 42
    print(f"Multiply result: {result}")

if __name__ == "__main__":
    test_calculator()
    test_multiply()
    print("All tests passed!")