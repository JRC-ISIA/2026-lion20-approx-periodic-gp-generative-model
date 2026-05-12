from src.kernels import *

@torch.no_grad
def get_data(periodic = False, full_n = 100, full_r = 10):
	# x_full = torch.linspace(-1, full_r+1, (full_r+1) * full_n)[..., None]
	x_full = torch.linspace(-1, full_r+1, (full_r + 2) * full_n)[..., None]
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
	prior_var = 1.5
	# prior_var = 0.1
	Kxx = k1 * k2 + prior_var * torch.eye(len(x_full))

	# x_test = torch.linspace(0, full_r, full_r * full_n)[..., None]
	x_test = torch.linspace(0, full_r, full_r * full_n)[..., None]
	Kstar = K1(x_test, x_full) * K2(x_test, x_full)
	Kstarstar = K1(x_test, x_test) * K2(x_test, x_test)

	y = torch.sin(2*torch.pi * x_full)

	L = torch.linalg.cholesky(Kxx)
	alpha = torch.cholesky_solve(y, L)
	m = Kstar @ alpha
	v = torch.linalg.solve_triangular(L, Kstar.T, upper=False) 
	S = Kstarstar - v.T @ v
	S = (S + S.T) / 2 + 1e-8 * torch.eye(len(S))

	y_full = m + torch.linalg.cholesky(S) @ torch.randn_like(m)

	cycle_idcs = x_test.floor().reshape(-1).to(torch.int)
	return x_test, y_full, cycle_idcs, m, S

def to_train(x_full, y_full, full_n, r, n):
	assert n <= full_n
	assert len(x_full) >= n * r
	x_train = torch.empty([r, n, 1])
	y_train = torch.empty([r, n, 1])

	for i in range(r):
		idcs = torch.randperm(full_n)[:n]
		idcs, _ = torch.sort(idcs)
		x_train[i] = x_full[idcs + full_n * i]
		y_train[i] = y_full[idcs + full_n * i]

	return x_train, y_train
