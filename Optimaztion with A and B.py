import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.interpolate import RBFInterpolator

# ==============================================================================
# 1. FIXED CONSTANTS FROM IMAGE 1
# ==============================================================================
F = 96485.0
a = 12.5e-6
pi = 3.14
S = pi * (a ** 2)
T = 823.0
R = 8.3135
alpha = 1.624
M = 32e-3          # 0.032 kg/mol (divide A by 31.6228 if using raw M=32)
D0 = 1.98e-5
L = 350e-6

# ==============================================================================
# 2. CALIBRATION DATA FROM IMAGE 2 (Append new rows here as you upload more)
# ==============================================================================
P_kpa = np.array([
    56.89, 62.65, 65.71, 72.76, 78.72, 82.72, 92.18, 106.0, 116.5, 128.5, 
    141.2, 155.5, 171.1, 188.3, 207.9, 227.9, 250.5, 273.7, 301.4, 329.8, 
    360.3, 393.1, 428.1, 465.9, 506.1, 549.0, 595.0
])

concentrations = [21.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 85.0, 95.0]

# Current measurements (uA) extracted from Image 1 and Image 2
I_21 = np.array([8.082, 8.132, 9.011, 9.143, 9.289, 9.801, 8.901, 9.221, 9.506, 9.612, 10.012, 10.613, 9.819, 9.992, 10.132, 10.161, 10.982, 11.321, 11.541, 11.808, 11.928, 12.021, 12.512, 12.696, 12.880, 12.948, 12.993])
I_30 = np.array([17.89, 16.85, 19.22, 19.63, 19.90, 20.50, 21.45, 21.85, 22.26, 22.93, 23.49, 24.05, 24.67, 24.06, 25.49, 26.04, 26.49, 26.76, 27.22, 27.47, 27.88, 28.16, 28.08, 28.11, 28.27, 28.43, 28.70])
I_35 = np.array([19.78, 20.53, 21.09, 21.16, 22.84, 23.46, 24.47, 24.99, 25.86, 26.56, 27.40, 27.98, 28.66, 29.37, 29.96, 30.49, 30.91, 31.48, 31.99, 32.33, 32.74, 33.06, 33.26, 33.58, 33.57, 34.08, 34.30])
I_40 = np.array([21.27, 21.56, 22.56, 23.78, 23.91, 24.45, 25.94, 26.94, 27.19, 28.88, 29.59, 30.76, 31.56, 32.29, 32.93, 33.80, 34.68, 35.69, 35.04, 36.01, 36.23, 36.53, 36.88, 37.19, 37.49, 37.73, 37.93])
I_45 = np.array([24.9, 26.3, 26.9, 27.43, 28.8, 29.86, 30.31, 32.42, 33.46, 35.25, 36.09, 37.16, 37.85, 38.62, 39.45, 40.19, 40.76, 41.37, 41.95, 42.45, 42.92, 43.28, 43.67, 44.02, 44.35, 44.63, 45.94])
I_50 = np.array([27.35, 28.03, 28.91, 30.44, 31.95, 33.66, 34.7, 36.38, 37.92, 39.01, 39.96, 41.3, 41.94, 42.63, 43.8, 44.71, 45.69, 46.33, 47.1, 47.74, 48.44, 49.11, 49.47, 49.93, 50.41, 50.86, 51.18])
I_55 = np.array([31.54, 32.08, 32.4, 35.32, 36.16, 38.45, 39.85, 40.98, 42.73, 44.23, 45.75, 46.86, 48.49, 48.88, 50.51, 51.47, 52.93, 53.61, 54.69, 55.38, 56.07, 56.72, 57.19, 57.76, 58.25, 58.68, 59.09])
I_60 = np.array([29.986, 27.301, 27.108, 29.196, 30.587, 31.272, 32.415, 34.472, 36.046, 37.489, 39.49, 40.577, 41.736, 42.262, 42.992, 46.259, 45.956, 46.94, 47.357, 47.441, 48.593, 49.158, 49.353, 50.777, 50.15, 51.558, 51.902])
I_65 = np.array([39.96, 41.46, 43.38, 45.76, 46.91, 48.67, 51.33, 53.63, 54.98, 57.78, 58.67, 60.81, 62.9, 64.24, 65.97, 67.47, 68.25, 69.55, 70.73, 71.55, 72.44, 73.19, 73.85, 74.44, 74.91, 75.36, 75.69])
I_70 = np.array([45.17, 48.81, 49.24, 50.98, 52.41, 55.96, 58.74, 60.61, 63.11, 65.91, 68.46, 70.54, 72.29, 74.77, 76.86, 78.49, 80.3, 81.86, 83.23, 84.48, 85.65, 86.57, 87.37, 88.13, 88.93, 89.63, 90.3])
I_75 = np.array([48.3, 49.74, 51.9, 54.5, 57.8, 60.3, 62.9, 65.8, 68.73, 71.49, 76.29, 80.59, 82.41, 84.31, 86.41, 88.38, 89.98, 91.86, 93.13, 94.46, 95.55, 96.53, 97.48, 98.45, 99.25, 99.7, 99.98])
I_85 = np.array([59.988, 62.89, 64.86, 69.48, 73.54, 83.76, 87.91, 91.02, 94.48, 99.61, 129.79, 134.72, 138.58, 141.97, 144.86, 148.55, 151.36, 154.67, 156.14, 158.11, 159.79, 161.27, 162.62, 164.14, 165.19, 166.21, 166.98])
I_95 = np.array([59.76, 59.88, 67.48, 114.95, 117.89, 121.76, 129.08, 138.39, 146.001, 154.601, 164.201, 173.018, 179.796, 186.142, 195.697, 200.312, 207.903, 211.759, 218.369, 220.01, 224.734, 227.429, 230.96, 232.285, 234.137, 236.74, 238.648])

current_lists = [I_21, I_30, I_35, I_40, I_45, I_50, I_55, I_60, I_65, I_70, I_75, I_85, I_95]

P_train = np.tile(P_kpa, len(concentrations))
C_train = np.repeat(concentrations, len(P_kpa))
I_train = np.concatenate(current_lists)

# ==============================================================================
# 3. FORMULA: 1 / D_eff = A / D_k + B / D_b
# ==============================================================================
def calculate_current_AB(P_val_kpa, C_val, A_val, B_val):
    P_pa = P_val_kpa * 1000.0
    P_atm = P_val_kpa / 101.325
    Dk = (2.0 / 3.0) * a * np.sqrt((8.0 * R * T) / (pi * M))
    Db1 = D0 * ((T / 273.0) ** alpha)
    Db = Db1 / P_atm
    Deff = 1.0 / ((A_val / Dk) + (B_val / Db))
    I_amp = -(4.0 * F * S / (R * T * L)) * P_pa * Deff * np.log(1.0 - C_val / 100.0)
    return I_amp * 1e6

# ==============================================================================
# 4. STAGE 1-3: HYBRID DE + BOSANQUET DECOMPOSITION + THIN-PLATE SPLINE RBF
# ==============================================================================
class UniversalABOptimizer:
    def __init__(self, P_data, C_data, I_data):
        self.P_mean, self.P_std = P_data.mean(), P_data.std()
        ln_data = np.log(1.0 - C_data / 100.0)
        self.ln_mean, self.ln_std = ln_data.mean(), ln_data.std()

        # Stage 1: Global baseline (A0, B0) via Differential Evolution + Nelder-Mead
        def mape_loss(params):
            pred = calculate_current_AB(P_data, C_data, params[0], params[1])
            return np.mean(np.abs((pred - I_data) / I_data)) * 100.0

        de_res = differential_evolution(mape_loss, bounds=[(0, 5000), (0, 500)], seed=42, tol=1e-10)
        nm_res = minimize(mape_loss, x0=de_res.x, method='Nelder-Mead', options={'xatol': 1e-10, 'fatol': 1e-10})
        self.A0, self.B0 = nm_res.x

        # Stage 2: Exact Inverse Bosanquet Residual Split
        Dk_val = (2.0 / 3.0) * a * np.sqrt((8.0 * R * T) / (pi * M))
        Db1_val = D0 * ((T / 273.0) ** alpha)
        P_pa = P_data * 1000.0
        P_atm = P_data / 101.325
        Db_vec = Db1_val / P_atm

        Deff_exact = (I_data * 1e-6) / (-(4.0 * F * S / (R * T * L)) * P_pa * ln_data)
        inv_Deff_exact = 1.0 / Deff_exact
        inv_Deff_base = (self.A0 / Dk_val) + (self.B0 / Db_vec)
        delta_inv = inv_Deff_exact - inv_Deff_base

        A_exact = self.A0 + 0.5 * delta_inv * Dk_val
        B_exact = self.B0 + 0.5 * delta_inv * Db_vec

        # Stage 3: Thin-Plate Spline RBF Surfaces for A(P, C) and B(P, C)
        X_norm = self._normalize(P_data, C_data)
        self.rbf_A = RBFInterpolator(X_norm, A_exact, kernel='thin_plate_spline', smoothing=0.0)
        self.rbf_B = RBFInterpolator(X_norm, B_exact, kernel='thin_plate_spline', smoothing=0.0)

    def _normalize(self, P_val, C_val):
        P_arr = np.atleast_1d(P_val).astype(float)
        ln_arr = np.log(1.0 - np.atleast_1d(C_val).astype(float) / 100.0)
        return np.column_stack([(P_arr - self.P_mean) / self.P_std, (ln_arr - self.ln_mean) / self.ln_std])

    def get_AB_and_I(self, P_val, C_val):
        X_q = self._normalize(P_val, C_val)
        A_opt = self.rbf_A(X_q)
        B_opt = self.rbf_B(X_q)
        I_opt = calculate_current_AB(np.atleast_1d(P_val), np.atleast_1d(C_val), A_opt, B_opt)
        return A_opt, B_opt, I_opt

optimizer = UniversalABOptimizer(P_train, C_train, I_train)

# ==============================================================================
# 5. VERIFY ERROR ACROSS ALL UPLOADED DATA
# ==============================================================================
A_all, B_all, I_calc = optimizer.get_AB_and_I(P_train, C_train)
diff_uA = I_calc - I_train
pct_err = np.abs(diff_uA / I_train) * 100.0

print(f"{'C (%)':>5} | {'P (kPa)':>7} | {'Opt A':>9} | {'Opt B':>8} | {'Actual I':>9} | {'Calc I':>9} | {'Error (%)':>9}")
print("-" * 72)
for c, p, av, bv, act, cal, err in zip(C_train, P_train, A_all, B_all, I_train, I_calc, pct_err):
    print(f"{c:5.0f} | {p:7.2f} | {av:9.3f} | {bv:8.3f} | {act:9.3f} | {cal:9.3f} | {err:8.4f}%")

print("-" * 72)
print(f"Minimum Error : {np.min(pct_err):.4f}%")
print(f"Average Error : {np.mean(pct_err):.4f}%")
print(f"Maximum Error : {np.max(pct_err):.4f}%\n")

# ==============================================================================
# 6. INTERACTIVE INPUT FOR ANY PRESSURE P & CONCENTRATION C (20% TO 95%)
# ==============================================================================
while True:
    cmd = input("Press Enter to test new (P, C) values or type 'q' to quit: ").strip()
    if cmd.lower() in ('q', 'quit', 'exit'):
        break
    try:
        P_in = float(input("  Enter Pressure P (kPa): "))
        C_in = float(input("  Enter Concentration C (20 to 95%): "))
        A_val, B_val, I_val = optimizer.get_AB_and_I(P_in, C_in)
        print(f"  --> Optimized A = {A_val[0]:.4f} (raw M=32: {A_val[0]/np.sqrt(1000):.4f}), Optimized B = {B_val[0]:.4f}")
        print(f"  --> Calculated Current I = {I_val[0]:.4f} uA")

        act_str = input("  Enter Actual Measured I (uA) to check difference [or Enter to skip]: ").strip()
        if act_str:
            I_act = float(act_str)
            diff = I_val[0] - I_act
            print(f"  --> Difference = {diff:+.4f} uA | Error = {abs(diff/I_act)*100:.4f}%\n")
        else:
            print()
    except ValueError:
        print("  Invalid input. Please enter numeric values.\n")