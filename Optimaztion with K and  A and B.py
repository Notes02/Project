import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.interpolate import RBFInterpolator

# ==============================================================================
# 1. FIXED CONSTANTS FROM IMAGE 1 (WITH M = 32 DIRECTLY)
# ==============================================================================
F = 96485.0
a = 12.5e-6
pi = 3.14
S = pi * (a ** 2)
T = 823.0
R = 8.3135
alpha = 1.624
M = 32.0           # Using M = 32 directly
D0 = 1.98e-5
L = 350e-6

# ==============================================================================
# 2. CALIBRATION DATA FROM IMAGE 2 (Append more rows anytime you upload data)
# ==============================================================================
P_kpa = np.array([56.89, 62.65, 65.71, 72.76, 78.72, 87.72, 96.18])
I_21  = np.array([8.082, 8.132, 9.011, 9.143, 9.289, 9.801, 8.901])
I_40  = np.array([21.27, 21.56, 22.56, 23.78, 23.91, 24.45, 25.94])
I_95  = np.array([59.76, 59.88, 67.48, 114.95, 117.89, 121.75, 129.08])

P_train = np.tile(P_kpa, 3)
C_train = np.repeat([21.0, 40.0, 95.0], len(P_kpa))
I_train = np.concatenate([I_21, I_40, I_95])

# ==============================================================================
# 3. FORMULA: K(P, C) / D_eff = A / D_k + B / D_b
# ==============================================================================
def calculate_current_KAB(P_val_kpa, C_val, K_val, A_val, B_val):
    P_pa = P_val_kpa * 1000.0        # kPa -> Pa
    P_atm = P_val_kpa / 101.325      # kPa -> atm

    Dk = (2.0 / 3.0) * a * np.sqrt((8.0 * R * T) / (pi * M))
    Db1 = D0 * ((T / 273.0) ** alpha)
    Db = Db1 / P_atm

    Deff = K_val / ((A_val / Dk) + (B_val / Db))
    I_amp = -(4.0 * F * S / (R * T * L)) * P_pa * Deff * np.log(1.0 - C_val / 100.0)
    return I_amp * 1e6  # Convert A -> uA

# ==============================================================================
# 4. OPTIMIZER: FIXED (A, B) + THIN-PLATE SPLINE RBF VARIABLE K(P, C)
# ==============================================================================
class SensorKABModel:
    def __init__(self, P_data, C_data, I_data):
        self.P_mean, self.P_std = P_data.mean(), P_data.std()
        ln_data = np.log(1.0 - C_data / 100.0)
        self.ln_mean, self.ln_std = ln_data.mean(), ln_data.std()

        # Stage 1: Optimize Fixed Constants A and B globally (with K=1 baseline)
        def base_mape(params):
            pred = calculate_current_KAB(P_data, C_data, 1.0, params[0], params[1])
            return np.mean(np.abs((pred - I_data) / I_data)) * 100.0

        de_res = differential_evolution(base_mape, bounds=[(0.0, 500.0), (0.0, 500.0)], seed=42, tol=1e-12)
        nm_res = minimize(base_mape, x0=de_res.x, method='Nelder-Mead', options={'xatol': 1e-12, 'fatol': 1e-12})
        self.A_fixed, self.B_fixed = nm_res.x

        # Stage 2: Exact Closed-Form K(P, C) at every calibration point
        Dk_val = (2.0 / 3.0) * a * np.sqrt((8.0 * R * T) / (pi * M))
        Db1_val = D0 * ((T / 273.0) ** alpha)
        P_pa = P_data * 1000.0
        P_atm = P_data / 101.325
        Db_vec = Db1_val / P_atm

        Deff_exact = (I_data * 1e-6) / (-(4.0 * F * S / (R * T * L)) * P_pa * ln_data)
        K_exact = Deff_exact * ((self.A_fixed / Dk_val) + (self.B_fixed / Db_vec))

        # Stage 3: Thin-Plate Spline RBF surface K(P, C)
        X_norm = self._normalize(P_data, C_data)
        self.rbf_K = RBFInterpolator(X_norm, K_exact, kernel='thin_plate_spline', smoothing=0.0)

    def _normalize(self, P_val, C_val):
        P_arr = np.atleast_1d(P_val).astype(float)
        ln_arr = np.log(1.0 - np.atleast_1d(C_val).astype(float) / 100.0)
        return np.column_stack([(P_arr - self.P_mean) / self.P_std, (ln_arr - self.ln_mean) / self.ln_std])

    def predict(self, P_val, C_val):
        X_q = self._normalize(P_val, C_val)
        K_var = self.rbf_K(X_q)
        I_out = calculate_current_KAB(np.atleast_1d(P_val), np.atleast_1d(C_val), K_var, self.A_fixed, self.B_fixed)
        return K_var, I_out

model = SensorKABModel(P_train, C_train, I_train)

# ==============================================================================
# 5. VERIFICATION TABLE & ERROR SUMMARY
# ==============================================================================
K_all, I_calc = model.predict(P_train, C_train)
diff_uA = I_calc - I_train
pct_err = np.abs(diff_uA / I_train) * 100.0

print(f"Fixed Constant A (M=32): {model.A_fixed:.6f}")
print(f"Fixed Constant B:        {model.B_fixed:.6f}\n")

print(f"{'C (%)':>5} | {'P (kPa)':>7} | {'Var K(P,C)':>10} | {'Actual I':>9} | {'Calc I':>9} | {'Diff (uA)':>10} | {'Error (%)':>9}")
print("-" * 74)
for c, p, kv, act, cal, d, err in zip(C_train, P_train, K_all, I_train, I_calc, diff_uA, pct_err):
    print(f"{c:5.0f} | {p:7.2f} | {kv:10.6f} | {act:9.3f} | {cal:9.3f} | {d:+10.4f} | {err:8.5f}%")

print("-" * 74)
print(f"Minimum Error : {np.min(pct_err):.6f}%")
print(f"Average Error : {np.mean(pct_err):.6f}%")
print(f"Maximum Error : {np.max(pct_err):.6f}%\n")

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
        K_val, I_val = model.predict(P_in, C_in)
        print(f"  --> Fixed A = {model.A_fixed:.6f}, Fixed B = {model.B_fixed:.6f}")
        print(f"  --> Adaptive K(P, C) = {K_val[0]:.6f}")
        print(f"  --> Calculated Current I = {I_val[0]:.4f} uA")

        act_str = input("  Enter Actual Measured I (uA) to see difference [or Enter to skip]: ").strip()
        if act_str:
            I_act = float(act_str)
            diff = I_val[0] - I_act
            print(f"  --> Difference = {diff:+.4f} uA | Error = {abs(diff/I_act)*100:.5f}%\n")
        else:
            print()
    except ValueError:
        print("  Invalid input. Please enter numeric values.\n")