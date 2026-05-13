import torch
from torch import nn
from typing import List

def make_positive(x):
	return nn.functional.softplus(x)

def make_positive_inv(y):
	y = torch.tensor([y]) if type(y) != torch.Tensor else y
	return torch.log(y.exp() - 1.0)

class Kernel(nn.Module):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def forward(self, x1: torch.tensor, x2: torch.tensor) -> torch.tensor:
		raise(NotImplementedError)

def squared_dist(x1, x2):
	x1_norm = (x1 ** 2).sum(dim=1, keepdim=True)
	x2_norm = (x2 ** 2).sum(dim=1, keepdim=True)

	norm = x1_norm + x2_norm.T - 2 * x1 @ x2.T
	return norm

class RBF(Kernel):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.l_raw = nn.Parameter(torch.tensor(0.0))
		self.var_raw = nn.Parameter(torch.tensor(0.0))

	def forward(self, x1: torch.tensor, x2: torch.tensor) -> torch.tensor:
		l = make_positive(2 * self.l_raw)
		var = make_positive(self.var_raw)

		norm = squared_dist(x1 / l, x2 / l)

		return var * torch.exp(-0.5 * norm)

class PeriodicKernel(Kernel):
	def __init__(self, period=1.0, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.l_raw = nn.Parameter(torch.tensor(0.0))
		self.var_raw = nn.Parameter(torch.tensor(0.0))
		self.p = period
		self.C = torch.pi / self.p
	
	def forward(self, x1: torch.tensor, x2: torch.tensor) -> torch.tensor:
		var = make_positive(self.var_raw)
		l = make_positive(2 * self.l_raw)

		diff = x1 - x2.T

		return var * torch.exp(-2 * torch.sin(self.C * diff) ** 2 / l)

class PeriodicWeightKernel(Kernel):
	def __init__(self, period=1.0, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.l_raw = nn.Parameter(torch.tensor(0.0))
		self.var_raw = nn.Parameter(torch.tensor(0.0))
		self.p = period
		self.C = self.p / (2*torch.pi)
	
	def phi(self, x):
		return 0.5 * (x - self.C * torch.sin(x / self.C))

	def forward(self, x1: torch.tensor, x2: torch.tensor) -> torch.tensor:
		var = make_positive(self.var_raw)
		l = make_positive(self.l_raw)

		d = squared_dist(self.phi(x1) / l, self.phi(x2) / l)

		K = var * torch.exp(-0.5 * d)
		return K

class NSPeriodicKernel(PeriodicKernel):
	# https://arxiv.org/pdf/2407.03608
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		N = 8
		self.w = nn.Sequential(
			nn.Linear(2, N),
			nn.Softplus(),
			nn.Linear(N, 1),
			nn.Softplus(),
		)
		self.phi = lambda x: torch.cat([torch.sin(2*self.C*x), torch.cos(2*self.C*x)], dim=1)
	
	def forward(self, x1: torch.tensor, x2: torch.tensor) -> torch.tensor:
		l = make_positive(2 * self.l_raw)
		w1, w2 = self.w(self.phi(x1)), self.w(self.phi(x2))
		w = w1 @ w2.T

		diff = x1 - x2.T
		return w * torch.exp(-2 * torch.sin(self.C * diff)**2 / l)
