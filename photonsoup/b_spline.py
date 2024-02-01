import numpy as np


class BSpline:
    """ Class for generating B-spline basis functions and associated spline map. """

    def __init__(self, num_points, degree, num_knots):
        """
        Initialize B-spline basis functions.

        Parameters:
        - num_points (int): Number of evaluation points.
        - degree (int): Degree of the spline.
        - num_knots (int): Number of knots.
        """

        # Construct knot vector
        self.knot_vector = np.array([0] * degree + list(range(num_knots - degree + 1)) + [num_knots - degree] * degree)
        # Uniformly spaced evaluation points
        eval_points = np.linspace(0, num_knots - degree, num_points)
        # Initialize basis matrix
        self.basis = np.zeros((num_points, num_knots))
        # Identify left and right indices in the knot vector for each evaluation point
        left_indices = np.clip(np.floor(eval_points).astype(int), 0, num_knots - degree - 1)
        right_indices = left_indices + degree + 1
        # Compute basis functions
        self._compute_basis(eval_points, left_indices, right_indices, degree)

    def _compute_basis(self, u, left, right, degree):
        """ Internal method for computing B-spline basis functions. """
        for idx, l in enumerate(left):
            self.basis[idx, l] = 1.0
            basis_buffer = np.zeros_like(self.basis)
            for j in range(1, degree + 1):
                basis_buffer[idx, l:l + j] = self.basis[idx, l:l + j]
                self.basis[idx, l] = 0.0
                for i in range(j):
                    fraction = basis_buffer[idx, l + i] / (
                                self.knot_vector[right[idx] + i] - self.knot_vector[right[idx] + i - j])
                    self.basis[idx, l + i] += fraction * (self.knot_vector[right[idx] + i] - u[idx])
                    self.basis[idx, l + i + 1] = fraction * (u[idx] - self.knot_vector[right[idx] + i - j])

    def create_spline_map(self, event_indices, map_length):
        """
        Creates a spline map based on event indices and map length.

        Parameters:
        - event_indices (list or ndarray): Indices of events to consider.
        - map_length (int): Length of the map to generate.

        Returns:
        - tuple: Spline map and dictionary representation.
        """
        spline_map = np.zeros((self.basis.shape[1], map_length))

        for i in range(spline_map.shape[0]):
            for event_index in event_indices:
                if event_index + self.basis.shape[0] > map_length:
                    continue
                spline_map[i, event_index:event_index + self.basis.shape[0]] = self.basis[:, i]
        return spline_map
