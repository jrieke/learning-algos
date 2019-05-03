import numpy as np
import matplotlib.pyplot as plt


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'."""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def plot_multiple(*arrays_and_labels):
    """Helper function for debugging to plot multiple arrays in the same plot."""
    for arr, lab in arrays_and_labels:
        plt.plot(arr, label=lab)
    plt.legend()
    plt.show()