function [alpha, alphax, alphas] = steplength(x, s, Dx, Ds, eta)

  alphax = -1/min(min(Dx./x),-1); alphax = min(1, eta * alphax);
  alphas = -1/min(min(Ds./s),-1); alphas = min(1, eta * alphas);
  alpha = min(alphax, alphas);
