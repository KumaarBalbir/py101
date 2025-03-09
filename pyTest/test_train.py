import pytest 
import models.train as train 

class TestTrainModel(object):
  
  @pytest.mark.xfail           # test suite remains green, even this test fails (xfail: expect this test to fail)
  def test_on_linear_data(self):
    pass
  
  def test_on_npy_data(self):
    pass