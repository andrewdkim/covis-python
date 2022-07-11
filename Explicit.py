class Explicit:
  def __init__(self, trials) -> None:
    self.delta_c = 0.0025
    self.delta_e = 0.02
    self.gamma = 1
    self.lam = 5
    self.zk_prev = 0.25 #at trial 0, this is zk(0)
    self.trials = trials

  def discriminant(x):
    pass
  def weight(self): 
    y_i = self.salience() + self.gamma
    pass

  def salience(self):
    #if correct
    zk_n = self.zk_prev + self.delta_c
    #if incorrect
    zk_n = self.zk_prev - self.delta_e

    
  
