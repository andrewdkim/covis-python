import numpy as np


class Explicit:
  def __init__(self, trials, rules) -> None:
    self.rules = rules
    self.trials = trials

    self.delta_c = 0.0025
    self.delta_e = 0.02
    self.gamma = 1
    self.lam = 5
    self.zk_prev = 0.25 #at trial 0, this is zk(0)

  def discriminant(x):
    pass
  def weight(self): 
    y_i = self.salience() + self.gamma
    pass

  def poisson_dist(self):
    return np.random.poisson(lam=self.lam)

  def normal_dist(self):
    return np.random.normal() #TODO



  def salience(self):
    #if correct
    zk_n = self.zk_prev + self.delta_c
    #if incorrect
    zk_n = self.zk_prev - self.delta_e

    
  
