# Python Implementation of the Hampel Filter

## Installation
To install the package execute the following command.

```bash
git clone https://github.com/nobuyukioishi/hampel_filter.git
pip install ./hampel_filter
```

## Usage
Currently, python 1d-list, NumPy 1d-array, Pandas Series are supported for input `x`.

### Function-based

```python
from hampel_filter import hampel_filter
outlier_indices = hampel_filter(x=x, window_size=5, n_sigma=3)  # x: timeseries values
```

### Class-based
The class-based implementation provides some additional functionality such as checking the upper/lower boundaries for paramter tuning.
```python
from hampel_filter import HampelFilter
hf = HampelFilter(window_size=5, n_sigma=3)
outlier_indices = hf.apply(x).get_indices()  # x: timeseries values
lower_boundaries, upper_boundaries = hf.get_boundaries()
```

## Code Sample
To obtain the outlier indices:
```python
import numpy as np
from hampel_filter import hampel_filter

# Prepare sample data
x = np.linspace(-np.pi, np.pi, 200)
sin_x = np.sin(x)
# set outliers
sin_x[[50, 150]] = 0  
sin_x[75] = -1
sin_x[125] = 1

# get outlier indices
outlier_indices = hampel_filter(x=sin_x, window_size=5, n_sigma=3) # 50, 75, 125, 150
```

To visualize the boundaries for parameter tuning:
```python
import numpy as np
from hampel_filter import HampelFilter
import matplotlib.pyplot as plt

# Prepare sample data
x = np.linspace(-np.pi, np.pi, 200)
sin_x = np.sin(x)
# set outliers
sin_x[[50, 150]] = 0  
sin_x[75] = -1
sin_x[125] = 1

# get outlier indices
hf = HampelFilter(window_size=5, n_sigma=3)
outlier_indices = hf.apply(x=sin_x).get_indices() # 50, 75, 125, 150
lower_bounds, upper_bounds = hf.get_boundaries()

fig, axs = plt.subplots(1, 2, figsize=(15, 5))
axs[0].plot(x, sin_x, label="values")
axs[0].plot(x[(hf.window_size-1)//2:-(hf.window_size-1)//2], lower_bounds, label="lower bounds")
axs[0].plot(x[(hf.window_size-1)//2:-(hf.window_size-1)//2], upper_bounds, label="upper_bounds")
axs[0].scatter(x[outlier_indices], sin_x[outlier_indices], c='r', label="outliers")
axs[0].set_title(f"Outlier detection using Hampel filter and its boundaries\n(window_size={hf.window_size}, n_sigma={hf.n_sigma})")
axs[0].legend()

# get outlier indices
hf = HampelFilter(window_size=7, n_sigma=5)
outlier_indices = hf.apply(x=sin_x).get_indices() # 50, 150
lower_bounds, upper_bounds = hf.get_boundaries()

axs[1].plot(x, sin_x, label="values")
axs[1].plot(x[(hf.window_size-1)//2:-(hf.window_size-1)//2], lower_bounds, label="lower bounds")
axs[1].plot(x[(hf.window_size-1)//2:-(hf.window_size-1)//2], upper_bounds, label="upper_bounds")
axs[1].scatter(x[outlier_indices], sin_x[outlier_indices], c='r', label="outliers")
axs[1].set_title(f"Outlier detection using Hampel filter and its boundaries\n(window_size={hf.window_size}, n_sigma={hf.n_sigma})")
axs[1].legend()
plt.show()
```
![Boundaries with different parameters](./images/parameter_tuning_example.png?raw=true)

## References
- https://www.mathworks.com/help/signal/ref/hampel.html
- https://github.com/MichaelisTrofficus/hampel_filter