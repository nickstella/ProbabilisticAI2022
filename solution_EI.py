import math
import numpy as np
from scipy.stats import norm
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

domain = np.array([[0, 5]])
SAFETY_THRESHOLD = 1.2
SEED = 0

""" Solution """

class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """

        kernel_f = .5*Matern(length_scale=.5, nu=2.5)
        kernel_v = math.sqrt(2)*Matern(length_scale=.5, nu=1.5)
        self.f_map = GaussianProcessRegressor(kernel = kernel_f)
        self.v_map = GaussianProcessRegressor(kernel = kernel_v)

        self.past_points = []

        pass


    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """

        # TODO: enter your code

        return self.optimize_acquisition_function()

        # In implementing this function, you may use optimize_acquisition_function() defined below.


    def optimize_acquisition_function(self):
        """
        Optimizes the acquisition function.

        Returns
        -------
        x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that maximize the acquisition function.
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * \
                 np.random.rand(domain.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=domain,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *domain[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        return np.atleast_2d(x_values[ind])

    def acquisition_function(self, x):
        """
        Compute the acquisition function.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f

        Returns
        ------
        af_value: float
            Value of the acquisition function at x
        """

        # TODO: enter your code here

        # Looking at https://github.com/jmetzen/bayesian_optimization/blob/master/bayesian_optimization/acquisition_functions.py

        mu_f, sigma_f = self.f_map.predict([x], return_std = True)
        mu_v, sigma_v = self.v_map.predict([x], return_std = True)
        PI_v = norm.cdf(SAFETY_THRESHOLD + .01, loc = mu_v, scale = sigma_v)

        if PI_v < .5:
            return PI_v
        else
            # Expected Improvement computation
            opt_f = max(self.past_points[:,1])
            a = (mu_f - opt_f - .01) / sigma_f
            erf = math.erf(a / math.sqrt(2))
            EI_f = (.5 * a * (1 + erf) + np.exp(-a**2 / 2) / math.sqrt(2*math.pi)) * sigma_f
            return PI_v * EI_f

        raise NotImplementedError


    def add_data_point(self, x, f, v):
        """
        Add data points to the model.

        Parameters
        ----------
        x: np.ndarray
            Hyperparameters
        f: np.ndarray
            Model accuracy
        v: np.ndarray
            Model training speed
        """

        new_point = np.append(np.append(x, np.atleast_2d(f), axis=1), np.atleast_2d(v), axis = 1)
        self.past_points = np.concatenate((self.past_points, new_point), axis = 0)

        x_train = np.transpose(np.atleast_2d(self.past_points[:, 0]))
        f_train = np.transpose(np.atleast_2d(self.past_points[:, 1]))
        v_train = np.transpose(np.atleast_2d(self.past_points[:, 2]))
        self.f_map.fit(x_train, f_train)
        self.v_map.fit(x_train, v_train)

        # TODO: enter your code here
        raise NotImplementedError

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """

        # TODO: enter your code here

        filter = (self.past_points[:,2] > SAFETY_THRESHOLD + .01)
        sol = self.past_points[filter]

        try:
            opt_index_x = np.argmax(sol[:, 1])
        except:
            opt_index_x= np.argmax(self.past_points[:, 1])
            sol = self.past_points

        return sol[opt_index_x, 0]

        raise NotImplementedError


""" Toy problem to check code works as expected """

def check_in_domain(x):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= domain[None, :, 0]) and np.all(x <= domain[None, :, 1])


def f(x):
    """Dummy objective"""
    mid_point = domain[:, 0] + 0.5 * (domain[:, 1] - domain[:, 0])
    return - np.linalg.norm(x - mid_point, 2)  # -(x - 2.5)^2


def v(x):
    """Dummy speed"""
    return 2.0

def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*domain[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val > SAFETY_THRESHOLD]
    np.random.seed(SEED)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]
    return x_init



def main():
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)


    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, domain.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {domain.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x)
        cost_val = v(x)
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = np.atleast_2d(agent.get_solution())
    assert solution.shape == (1, domain.shape[0]), \
        f"The function get solution must return a numpy array of shape (" \
        f"1, {domain.shape[0]})"
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the ' \
        f'domain, {solution} returned instead'

    # Compute regret
    if v(solution) < 1.2:
        regret = 1
    else:
        regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret{regret}')


if __name__ == "__main__":
    main()