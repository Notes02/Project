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
P_kpa = np.array([56.89, 62.65, 65.71, 72.76, 78.72, 87.72, 96.18])
I_21  = np.array([8.082, 8.132, 9.011, 9.143, 9.289, 9.801, 8.901])
I_40  = np.array([21.27, 21.56, 22.56, 23.78, 23.91, 24.45, 25.94]) 

I_95  = np.array([59.76, 59.88, 67.48, 114.95, 117.89, 121.75, 129.08])

P_train = np.tile(P_kpa, 3)
C_train = np.repeat([21.0, 40.0, 95.0], len(P_kpa))
I_train = np.concatenate([I_21, I_40, I_95])

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