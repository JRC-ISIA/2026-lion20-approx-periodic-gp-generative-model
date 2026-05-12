from src.kernels import *

@torch.no_grad
def get_data(periodic = False, full_n = 100, full_r = 10):
	x_full = torch.linspace(-1, full_r+1, (full_r+1) * full_n)[..., None]
	K1 = PeriodicKernel(1.0)
	K2 = PeriodicWeightKernel(1.0)
	K1.l_raw.set_(torch.tensor(0.015))
	K2.l_raw.set_(torch.tensor(0.18))
	K1.var_raw.set_(make_positive_inv(1.0))
	K2.var_raw.set_(make_positive_inv(1.0))

	k1 = K1(x_full, x_full)
	k2 = K2(x_full, x_full)
	if periodic:
		k2 = 1.0
		K2 = lambda x1, x2: 1.0
	prior_var = 0.01
	Kxx = k1 * k2 + prior_var * torch.eye(len(x_full))

	x_test = torch.linspace(0, full_r, full_r * full_n)[..., None]
	Kstar = K1(x_test, x_full) * K2(x_test, x_full)
	Kstarstar = K1(x_test, x_test) * K2(x_test, x_test)

	e = torch.randn_like(x_test)
	L = torch.linalg.cholesky(Kxx)

	# first dim
	y1 = torch.sin(2*torch.pi * (x_full))
	alpha = torch.cholesky_solve(y1, L)
	m1 = Kstar @ alpha
	v = torch.linalg.solve_triangular(L, Kstar.T, upper=False) 
	S1 = Kstarstar - v.T @ v
	S1 = (S1 + S1.T) / 2 + 1e-8 * torch.eye(len(S1))
	y_full1 = m1 + torch.linalg.cholesky(S1) @ e

	# second dim
	y2 = -torch.sin(2 * 2 * torch.pi * (x_full))
	alpha = torch.cholesky_solve(y2, L)
	m2 = Kstar @ alpha
	v = torch.linalg.solve_triangular(L, Kstar.T, upper=False) 
	S2 = Kstarstar - v.T @ v
	S2 = (S2 + S2.T) / 2 + 1e-8 * torch.eye(len(S2))
	y_full2 = m2 + torch.linalg.cholesky(S2) @ e

	# synthesis
	y_full = torch.cat([y_full1, y_full2], 1) 
	m = torch.cat([m1, m2], 0) 
	S = torch.block_diag(S1, S2)

	cycle_idcs = x_test.floor().reshape(-1).to(torch.int)
	return x_test, y_full, cycle_idcs, m, S

def to_train(x_full, y_full, full_n, r, n):
	assert n <= full_n
	assert len(x_full) >= n * r
	x_train = torch.empty([r, n, 1])
	y_train = torch.empty([r, n, 2])

	for i in range(r):
		idcs = torch.randperm(full_n)[:n]
		idcs, _ = torch.sort(idcs)
		x_train[i] = x_full[idcs + full_n * i]
		y_train[i] = y_full[idcs + full_n * i]

	return x_train, y_train
