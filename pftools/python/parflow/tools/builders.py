# builders.py
# functions for helping build ParFlow scripts


class EcoSlimBuilder:

  def __init__(self, run_name):
      self.run_name = run_name

  def key_add(self, keys=['PrintVelocities']):
      self.run_name.Solver.PrintVelocities = True
      return self

  def write_slimin(self):
      return