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
outlier_indices = hf.apply(x)  # x: timeseries values
lower_boundaries, upper_boundaries = hf.get_boundaries()
```

## Code Sample
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

## References
- https://www.mathworks.com/help/signal/ref/hampel.html
- https://github.com/MichaelisTrofficus/hampel_filter