| route / variant | gain | n | median g [IQR] | g > 0 | interval excludes 0 on the predicted side | median B | model vs network hit fraction |
|---|---|---|---|---|---|---|---|
| route 1 MLE (95 % Wald) | 0.8 | 20 | **+4.13** [+3.02, +4.63] | 20/20 | **20/20** | -- | -- |
| route 1 MLE (95 % Wald) | 1.0 | 20 | **+0.32** [-0.34, +0.79] | 12/20 | **15/20** | -- | -- |
| route 1 MLE (95 % Wald) | 1.2 | 20 | **-2.63** [-3.56, -2.03] | 0/20 | **20/20** | -- | -- |
| route 1 NUTS (94 % HDI) | 0.8 | 20 | **+3.95** [+2.95, +4.50] | 20/20 | **20/20** | -- | -- |
| route 1 NUTS (94 % HDI) | 1.0 | 20 | **+0.33** [-0.34, +0.81] | 12/20 | **15/20** | -- | -- |
| route 1 NUTS (94 % HDI) | 1.2 | 20 | **-2.63** [-3.56, -2.03] | 0/20 | **20/20** | -- | -- |
| route 2, no hit term (95 % profile) | 0.8 | 20 | **+4.13** [+3.01, +4.58] | 20/20 | **20/20** | 6.59 | 0.004 vs 0.858 |
| route 2, no hit term (95 % profile) | 1.0 | 20 | **+0.52** [-0.20, +0.79] | 13/20 | **14/20** | 6.04 | 0.197 vs 0.955 |
| route 2, no hit term (95 % profile) | 1.2 | 20 | **-2.59** [-3.77, -2.11] | 1/20 | **19/20** | 6.38 | 0.664 vs 0.979 |
| route 2, hit term (deadline commitment |dv_T| >= 2) (95 % profile) | 0.8 | 20 | **+2.51** [+1.88, +3.06] | 20/20 | **20/20** | 1.44 | 0.448 vs 0.463 |
| route 2, hit term (deadline commitment |dv_T| >= 2) (95 % profile) | 1.2 | 20 | **-2.53** [-3.23, -2.19] | 0/20 | **20/20** | 1.80 | 0.874 vs 0.880 |
| route 2, hit term (Bernoulli, ever crossed) (95 % profile) | 0.8 | 20 | **+5.90** [+4.56, +6.82] | 20/20 | **20/20** | 0.97 | 0.814 vs 0.858 |
| route 2, hit term (Bernoulli, ever crossed) (95 % profile) | 1.2 | 20 | **-3.51** [-4.32, -2.95] | 1/20 | **19/20** | 1.38 | 0.964 vs 0.979 |
| route 2, hit term (crossing-time) (95 % profile) | 0.8 | 20 | **+8.28** [+7.63, +8.97] | 20/20 | **20/20** | 0.94 | 0.846 vs 0.858 |
| route 2, hit term (crossing-time) (95 % profile) | 1.2 | 20 | **-1.19** [-2.43, +0.32] | 6/20 | **14/20** | 0.83 | 0.983 vs 0.979 |
