import numpy as np
from scipy.optimize import minimize_scalar

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
M = 32e-3          # 32 g/mol -> 0.032 kg/mol (if using raw M=32, divide A by 31.6228)
D0 = 1.98e-5
L = 350e-6

# ==============================================================================
# 2. CALIBRATION DATASET FROM IMAGE 2 (Add more rows here when uploading more data)
# ==============================================================================
P_kpa = np.array([56.89, 62.65, 65.71, 72.76, 78.72, 87.72, 96.18])
I_21  = np.array([8.082, 8.132, 9.011, 9.143, 9.289, 9.801, 8.901])
I_40  = np.array([21.27, 21.56, 22.56, 23.78, 23.91, 24.45, 25.94])
I_95  = np.array([59.76, 59.88, 67.48, 114.95, 117.89, 121.75, 129.08])

P_train = np.tile(P_kpa, 3)
C_train = np.repeat([21.0, 40.0, 95.0], len(P_kpa))
I_train = np.concatenate([I_21, I_40, I_95])

# ==============================================================================
# 3. PHYSICAL FORMULA MODEL
# ==============================================================================
def calculate_current(P_val_kpa, C_val, A_val):
    """Calculates current I (in uA) for any Pressure P (kPa) and Concentration C (%)."""
    P_pa = P_val_kpa * 1000.0        # kPa -> Pa
    P_atm = P_val_kpa / 101.325      # kPa -> atm

    Dk = (2.0 / 3.0) * a * np.sqrt((8.0 * R * T) / (pi * M))
    Db1 = D0 * ((T / 273.0) ** alpha)
    Db = Db1 / P_atm

    Deff = 1.0 / ((A_val / Dk) + (1.0 / Db))
    I_amp = -(4.0 * F * S / (R * T * L)) * P_pa * Deff * np.log(1.0 - C_val / 100.0)
    return I_amp * 1e6  # Convert A -> uA

# ==============================================================================
# 4. BRENT'S BOUNDED OPTIMIZATION (MINIMIZING MAPE) FOR SINGLE CONSTANT A
# ==============================================================================
def optimize_single_A(P_data, C_data, I_actual_data):
    result = minimize_scalar(
        lambda A: np.mean(np.abs((calculate_current(P_data, C_data, A) - I_actual_data) / I_actual_data)) * 100.0,
        bounds=(1.0, 10000.0),
        method='bounded',
        options={'xatol': 1e-10}
    )
    return result.x

A_opt = optimize_single_A(P_train, C_train, I_train)

# ==============================================================================
# 5. SHOW FULL COMPARISON & ERROR SUMMARY ON UPLOADED DATA
# ==============================================================================
I_calc = calculate_current(P_train, C_train, A_opt)
diff_uA = I_calc - I_train
abs_pct_err = np.abs(diff_uA / I_train) * 100.0

print(f"Optimal Single Constant A (M=0.032 kg/mol): {A_opt:.6f}")
print(f"Optimal Single Constant A (if M=32 raw):    {A_opt / np.sqrt(1000):.6f}\n")

print(f"{'C (%)':>5} | {'P (kPa)':>7} | {'Actual I':>9} | {'Calc I':>9} | {'Diff (uA)':>10} | {'Error (%)':>9}")
print("-" * 64)
for c, p, act, cal, d, err in zip(C_train, P_train, I_train, I_calc, diff_uA, abs_pct_err):
    print(f"{c:5.0f} | {p:7.2f} | {act:9.3f} | {cal:9.3f} | {d:+10.3f} | {err:8.2f}%")

print("-" * 64)
print(f"Minimum Error : {np.min(abs_pct_err):.2f}%  (Min Abs Diff: {np.min(np.abs(diff_uA)):.4f} uA)")
print(f"Average Error : {np.mean(abs_pct_err):.2f}%  (Avg Abs Diff: {np.mean(np.abs(diff_uA)):.4f} uA)")
print(f"Maximum Error : {np.max(abs_pct_err):.2f}%  (Max Abs Diff: {np.max(np.abs(diff_uA)):.4f} uA)\n")

# ==============================================================================
# 6. INTERACTIVE INPUT MODE: ENTER ANY PRESSURE & C (20% TO 95%)
# ==============================================================================
while True:
    user_in = input("Enter Pressure P (kPa) and Concentration C (20-95%) [or 'q' to quit]: ").strip()
    if user_in.lower() in ('q', 'quit', 'exit'):
        break
    try:
        P_in = float(input("  Enter Pressure P (kPa): "))
        C_in = float(input("  Enter Concentration C (20 to 95%): "))
        if not (20.0 <= C_in <= 95.0):
            print("  Warning: C is outside 20% to 95% range, calculating anyway...")
        
        I_out = calculate_current(P_in, C_in, A_opt)
        print(f"  --> Calculated Current I = {I_out:.4f} uA (using A = {A_opt:.4f})")

        act_str = input("  Enter Actual Measured I (uA) to see difference [or press Enter to skip]: ").strip()
        if act_str:
            I_act_in = float(act_str)
            diff = I_out - I_act_in
            err_pct = abs(diff / I_act_in) * 100.0
            print(f"  --> Difference = {diff:+.4f} uA | Percentage Error = {err_pct:.2f}%\n")
        else:
            print()
    except ValueError:
        print("  Invalid number entered. Please try again.\n")