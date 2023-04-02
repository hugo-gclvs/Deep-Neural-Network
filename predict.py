from forward_propa import *

def predict(X, parametres):
  
  activations = forward_propagation(X, parametres)

  C = len(parametres) // 2

  Af = activations['A' + str(C)]
  
  return Af >= 0.5