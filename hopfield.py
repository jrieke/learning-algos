import numpy as np
import matplotlib.pyplot as plt



def plot_pattern(p):
    plt.imshow(p.reshape(5, 5), cmap='gray')
    plt.colorbar()
    plt.show()


num_states = 25
W = np.zeros((num_states, num_states))
threshold = 0

pattern1 = np.ones(num_states)
pattern1[::3] = -1
plot_pattern(pattern1)

pattern2 = np.ones(num_states)
pattern2[::2] = -1
plot_pattern(pattern2)

patterns = np.array([pattern1, pattern2])

# "Training" via Hebbian learning.
for (i, j), _ in np.ndenumerate(W):
    if i == j:
        W[i, j] = 0
    else:
        W[i, j] = np.mean(patterns[:, i] * patterns[:, j])


# Retrieve from test pattern.
test_pattern = pattern2.copy()
test_pattern[:10] = 1
plot_pattern(test_pattern)
states = test_pattern.copy()

for step in range(10):
    # Update step.
    for i in range(num_states):
        if np.sum(W[i] * states) > threshold:
            states[i] = 1
        else:
            states[i] = -1
    plot_pattern(states)
