import pytest 
import square

def test_square():
  result = square(5)
  assert result == 25 
  assert isinstance(result,float) # test will pass if all assertions pass.
  


# checkinf for None values
# assert var is None (Don't do this: assert var == None) 

# If in result it is .F. means Failure (When: an exception is raised when running unit tests)
# Action: Fix the function or unit test 

# assert 0.1+0.1+0.1 == 0.3 (Don't do this: floats are not compared this way for precision issues) 
# assert 0.1+0.1 == pytest.approx(0.2)