import numpy as np
import bf_utils as bf

def self_covariance(matrix):
    return np.matmul(np.conjugate(np.transpose(matrix)), matrix)


def update_weight_vector(w_old):
    gain_vector = update_gain_vector()
    w_old +

class RLS:
    def __init__(self, num_el, weighting_factor):
        self.num_el = num_el
        self.weighting_factor = weighting_factor
        self.current_array_output = np.atleast_2d(np.zeros(num_el))
        self.weight_vector = np.atleast_2d(np.ones(num_el, dtype=complex))
        self.inverse_covariance_matrix = np.atleast_2d(np.zeros([num_el, num_el], dtype=complex))
        self.gain_vector = np.atleast_2d(np.zeros(num_el, dtype=complex))

    def step(self, array_output):
        self.current_array_output = array_output
        self.update_gain_vector()
        self.update_inverse_covariance_matrix()
        self.update_weight_vector()

    def update_gain_vector(q_old):
        numerator = 1/weighting_factor * np.matmul(self.inverse_covariance_matrix


rls = RLS(8, 0.5)
