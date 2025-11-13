Parametric Curve Fitting — Submission

🔹 Unknown Variables (Final Answer)
θ = 16.261477479573223°  
M = 0.0037787667237349685  
X = 27.111637744933333  

These values satisfy all assignment bounds:
- 0° < θ < 50°
- −0.05 < M < 0.05
- 0 < X < 100


🔹 Final Parametric Equations

 Desmos Version (ready to paste)

x(t) = t*cos(16.261477479573°)  
    - e^(0.003778766724*abs(t))*sin(0.3t)*sin(16.261477479573°)  
    + 27.111637744933  

y(t) = 42  
    + t*sin(16.261477479573°)  
    + e^(0.003778766724*abs(t))*sin(0.3t)*cos(16.261477479573°)

Domain: (6 ≤ t ≤ 60)

---

LaTeX Version (same expression in assignment-approved format)

\[
\left(
t\cos(16.261477^{\circ})
- e^{\,0.003778766724\,|t|}\sin(0.3t)\sin(16.261477^{\circ})
+ 27.111637744933,
\;
42 + t\sin(16.261477^{\circ})
+ e^{\,0.003778766724\,|t|}\sin(0.3t)\cos(16.261477^{\circ})
\right)
\]


🔹 Method (Short Explanation)

- The provided dataset `xy_data.csv` contained only **(x, y)** values;  
  **the corresponding t values were not provided**.
- To recover the unknown parameters (θ, M, X), I implemented an  
  (Alternating Optimization + Projection Method):

 Algorithm Steps:
1. Initialize t values uniformly in [6, 60].
2. For each point (xᵢ, yᵢ), find the tᵢ that best matches the parametric model  
   (bounded scalar minimization).
3. Enforce monotonicity of t (since curve samples must be ordered).
4. Refit θ, M, X using bounded nonlinear least squares (soft L1 loss).
5. Repeat for multiple initial θ guesses (multi-start) and choose the best in-bounds result.

This approach gives a smooth, consistent parameter recovery and obeys all assignment constraints.


🔹 Fit Quality Metrics (for completeness)

L1_total = 27475.800964536007  
L1_mean  = 18.317200643024005  
RMSE     = 15.791891212554846  

These are the best values (within the allowed parameter bounds).


🔹 Files Included in This Repository

- `final_parameters.txt`  
- `desmos_equation.txt`  
- `fit_code.py` (All computations were performed in Google Colab.)  
- `plots/fit_alternating.png`  
- `plots/residuals_vs_t.png`  
- `plots/residuals_hist.png`

All files together reproduce the final submitted values.



🔹 Notes

As per the assignment instructions only the unknown variable values and equations are required, but additional analysis (plots and code) is included for completeness and clarity.



---End of Submission---
