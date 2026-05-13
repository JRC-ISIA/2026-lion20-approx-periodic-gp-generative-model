import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
from typing import List, Literal
from einops import einsum, rearrange
from itertools import cycle, chain
from src.kernels import make_positive, make_positive_inv

from scipy.linalg import cholesky_banded, cho_solve_banded

import matplotlib.pyplot as plt

class SWDataset(torch.utils.data.Dataset):
	def __init__(self, X, y, batch_size):
		self.X = X
		self.y = y
		self.B = batch_size

	def __getitem__(self, index):
		x = self.X[index:index+self.B]
		y = self.y[index:index+self.B]
		return x, y

	def __len__(self):
		return len(self.X) - self.B + 1

class GPPW(nn.Module):
	def __init__(
		self,
		X: torch.tensor,
		Y: torch.tensor,
		kernels: List[nn.Module],
		weight_kernel: nn.Module,
		learn_out_var = True,
		num_latents: int = 1
	):
		super().__init__()

		self.learn_out_var = learn_out_var

		self.kernels = nn.ModuleList(kernels)
		self.weight_kernel = weight_kernel

		self.n_realizations = X.shape[0]
		self.X = X.reshape(self.n_realizations, -1, 1)			# (N, P, 1)
		if len(Y.shape) == 2:			
			Y = Y.unsqueeze(-1)					    			# (N, P, 1)
		self.y = Y

		self.x_full = X.reshape(-1, 1)
		self.y_full = Y.permute(2, 0, 1).reshape(-1, 1)

		self.N = Y.shape[0]
		self.P = Y.shape[1]
		self.D = Y.shape[2]
		self.Q = len(kernels)

		if self.D == 1:
			self.A = torch.ones([1, 1, 1])
		else:
			self.A = nn.Parameter(0.1 * torch.randn(self.Q, self.D, num_latents))
		self.B: torch.Tensor				# Q x D x D

		self.var_n_raw = nn.Parameter(torch.zeros(self.D))
		self.var_o_raw = nn.Parameter(torch.zeros(self.D))

		self.KL: torch.tensor
		self.alpha: torch.tensor

	def posterior_weighting(self, x1, x2, cov, with_noise=False):
		var_o = make_positive(self.var_o_raw)

		W = torch.kron(torch.ones(self.D, self.D), self.weight_kernel(x1, x2))
		S = cov * W

		if with_noise and self.learn_out_var and len(x1) == len(x2):
			s_n = torch.ones(len(x1))
			S_n = torch.diag(torch.kron(var_o, s_n.reshape(-1)))
			S += S_n

		return S

	def set_Kxx(self):
		Kxx = self.kernel(self.x_full, self.x_full,
			make_positive(self.var_n_raw))
		self.KL = self.cholesky(Kxx)
		self.alpha = torch.cholesky_solve(self.y_full, self.KL)

	def set_B(self):
		self.B = self.get_B()

	def get_B(self):
		return einsum(self.A, self.A, 'Q D1 Qr, Q D2 Qr -> Q D1 D2')

	@staticmethod
	def cholesky(K, jitter=1e-8, attempts=6):
		for i in range(attempts):
			try:
				L = torch.linalg.cholesky(K)
				return L

			except RuntimeError as e:
				if 'cholesky' not in str(e).lower():
					raise
				K += jitter * 10**i * torch.eye(*K.shape)
				if i == attempts-1:
					plt.imshow(K.detach())
					plt.show()

		raise RuntimeError(
			   f'Cholesky failed after adding {jitter * 10**i:.1e} jitter.'
		)

	def kernel(self, x1, x2, var_n=None):
		if self.training:
			B = self.get_B()
		else:
			B = self.B

		if var_n is None:
			S_n = 0.0
		else:
			s_n = torch.ones(len(x1))
			S_n = torch.diag(torch.kron(var_n, s_n.reshape(-1)))

		K = [k(x1, x2) for k in self.kernels]
		
		return sum(torch.kron(b, k) for b, k in zip(B, K)) + S_n

	def nll(self, x, y, weight=False, mean=0.0, cov=1.0, weight_noise=True):
		if weight:
			y = y - mean
			K = self.posterior_weighting(x, x, cov, weight_noise)
		else:
			var_n = make_positive(self.var_n_raw)
			K = self.kernel(x, x, var_n)

		L = self.cholesky(K)
		alpha = torch.cholesky_solve(y, L)

		Z = y.numel() * torch.tensor(2.0 * torch.pi).log()

		return 0.5 * (y.T @ alpha + 2*torch.log(torch.diag(L)).sum() + Z)

	def fit(self, lr:float, iterations: int, batch_size=2):
		params = self.parameters()
		opt = Adam(lr=lr, params=params)

		ds = torch.utils.data.TensorDataset(self.X, self.y)
		dl = torch.utils.data.DataLoader(
			ds,
			batch_size=batch_size,
			shuffle=True,
			drop_last=True
		)
		losses = [None] * iterations

		for it in (pbar:=tqdm(range(iterations))):
			loss = 0
			for (x, y) in dl:
				x, y = x.reshape(-1, 1), y.permute(2, 0, 1).reshape(-1, 1)
				loss += self.nll(x, y)
			loss.backward()
			opt.step()
			opt.zero_grad()

			losses[it] = loss.item()
			pbar.set_description(
				f'it: {it % iterations:<{int(np.log10(iterations))}} - '
				f'loss: {loss.item():.3e}'
			)

		with torch.no_grad():
			self.var_n_raw.set_(
				make_positive_inv(
					make_positive(self.var_n_raw) * self.N
				)
			)
			self.set_B()
			self.set_Kxx()

		return losses

	def fit_weight(self, lr: float, iterations: int, banded=False):
		params = list(chain(self.weight_kernel.parameters(), [self.var_o_raw]))
		opt = Adam(lr=lr, params=params)

		losses = [None] * iterations

		x, y = self.x_full, self.y_full

		with torch.no_grad():
			m, S = self.predict(x, with_noise=False)

		for it in (pbar:=tqdm(range(iterations))):
			loss = self.nll(x, y, weight=True, mean=m, cov=S)
			loss.backward()
			opt.step()
			opt.zero_grad()

			losses[it] = loss.item()
			pbar.set_description(
				f'it: {it % iterations:<{int(np.log10(iterations))}} - '
				f'loss: {loss.item():.3e}'
			)

		return losses
	
	def predict(self, x_star, with_noise=False):
		x_star = x_star.reshape(-1, 1)
		with torch.no_grad():
			k_star = self.kernel(self.x_full, x_star)
			mu = k_star.T @ self.alpha

			v = torch.linalg.solve_triangular(self.KL, k_star, upper=False)
			cov = self.kernel(x_star, x_star) - v.T @ v
			cov = (cov + cov.T) / 2

			if with_noise:
				var_n = make_positive(self.var_n_raw) / self.N
				e = var_n * torch.eye(len(x_star))
				cov += e

		return mu, cov

	def generate(self, x_star, with_noise=False):
		x_star = x_star.reshape(-1, 1)
		mu, cov = self.predict(x_star, with_noise=False)
		S = self.posterior_weighting(x_star, x_star, cov, with_noise) 
		S = (S + S.T) / 2 + 1e-8 * torch.eye(len(S))

		return mu, S
	
	def sample(self, m, S):
		return m + self.cholesky(S) @ torch.randn_like(m)
