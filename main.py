import numpy as np
import cvxpy as cp

def generate_random_covariance(n):
    A = np.random.randn(n, n)
    Sigma = A.T @ A
    return Sigma / np.max(np.abs(Sigma))

def solve_sdp(Sigma):
    n = Sigma.shape[0]
    e = np.ones(n)
    X = cp.Variable((n, n), PSD=True)
    objective = cp.Minimize(cp.trace(Sigma @ X))
    constraints = [cp.trace(np.outer(e, e) @ X) == 1]
    
    prob = cp.Problem(objective, constraints)
    try:
        result = prob.solve(solver=cp.MOSEK) if 'MOSEK' in cp.installed_solvers() else prob.solve(solver=cp.SCS)
    except cp.error.SolverError:
        return None, None

    return X.value if prob.status == cp.OPTIMAL else None, prob.status

def extract_portfolio_weights(X):
    if X is None:
        return None

    vals, vecs = np.linalg.eigh(X)
    idx = np.argsort(vals)[::-1]
    vals, vecs = vals[idx], vecs[:, idx]

    lambda_max = max(vals[0], 0)
    w = np.sqrt(lambda_max) * vecs[:, 0]

    if np.sum(w) < 0:
        w = -w

    w = w / np.sum(w)
    w = np.maximum(w, 0)
    return w / np.sum(w)

def sdp_portfolio(n=5):
    Sigma = generate_random_covariance(n)
    X, status = solve_sdp(Sigma)

    if X is None:
        print("Solver failed or problem is infeasible.")
        return

    w_approx = extract_portfolio_weights(X)
    print("\n===== SDP Portfolio Optimization Result =====")
    print("Solver Status    :", status)
    print("Optimal Risk     :", np.trace(Sigma @ X))
    print("Portfolio Weights:", w_approx)
    print("Sum of Weights   :", np.sum(w_approx))
    print("Portfolio Risk   :", w_approx.T @ Sigma @ w_approx)

if __name__ == "__main__":
    sdp_portfolio()
