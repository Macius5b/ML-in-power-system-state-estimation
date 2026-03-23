# -*- coding: utf-8 -*-
"""
Attachment A: State Estimation Comparison - LS vs WLS
Author: Aleks Piszczek
Date: 17.09.2025

This script implements and compares two classical state estimation methods
in a simple 2-bus power system:
- LS (Least Squares)
- WLS (Weighted Least Squares)

The scenario includes one measurement with a gross error ("bad data") to
demonstrate the advantage of WLS in such conditions.
"""

# =============================================================================
# Step 1: Install and import libraries
# =============================================================================
try:
    import pandapower as pp
except ImportError:
    print("Installing pandapower...")
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandapower"])
    import pandapower as pp

import pandas as pd
import numpy as np

# =============================================================================
# Step 2: Define network and reference settings
# =============================================================================
np.random.seed(42)

TRUE_P_LOAD_MW = 0.1
TRUE_Q_LOAD_MVAR = 0.05
V0_NOMINAL = 1.02  # Slack bus voltage

def create_base_network():
    """Creates a simple 2-bus power system using pandapower."""
    net = pp.create_empty_network(name="2-Bus System")
    
    # Create buses
    bus_slack = pp.create_bus(net, vn_kv=0.4, name="Slack Bus")
    bus_load = pp.create_bus(net, vn_kv=0.4, name="Load Bus (PQ)")
    
    # Create line
    pp.create_line_from_parameters(
        net, from_bus=bus_slack, to_bus=bus_load, length_km=0.5,
        r_ohm_per_km=0.3, x_ohm_per_km=0.1, c_nf_per_km=50, max_i_ka=0.25,
        name="Line L1"
    )
    
    # External grid and load
    pp.create_ext_grid(net, bus=bus_slack, vm_pu=V0_NOMINAL, name="External Grid")
    pp.create_load(net, bus=bus_load, p_mw=TRUE_P_LOAD_MW, q_mvar=TRUE_Q_LOAD_MVAR, name="Load")
    
    return net, bus_slack, bus_load

net_true, bus_slack, bus_load = create_base_network()
pp.runpp(net_true)

true_v_load = net_true.res_bus.vm_pu[bus_load]
true_theta_load = net_true.res_bus.va_degree[bus_load]

# =============================================================================
# Step 3: Prepare Ybus and mathematical model
# =============================================================================
Ybus = net_true._ppc["internal"]["Ybus"].toarray()
Y00, Y01, Y10, Y11 = Ybus[0,0], Ybus[0,1], Ybus[1,0], Ybus[1,1]
G00, B00, G01, B01 = Y00.real, Y00.imag, Y01.real, Y01.imag
G10, B10, G11, B11 = Y10.real, Y10.imag, Y11.real, Y11.imag

def h_full(x_k):
    """Predicted measurements vector h(x) from state vector x_k=[V1, theta1]."""
    V1, theta1_rad = x_k
    theta0_rad = 0.0
    delta01 = theta0_rad - theta1_rad
    delta10 = -delta01
    
    P0_calc = V0_NOMINAL**2 * G00 + V0_NOMINAL * V1 * (G01 * np.cos(delta01) + B01 * np.sin(delta01))
    Q0_calc = -V0_NOMINAL**2 * B00 + V0_NOMINAL * V1 * (G01 * np.sin(delta01) - B01 * np.cos(delta01))
    P1_inj = V1**2 * G11 + V1 * V0_NOMINAL * (G10 * np.cos(delta10) + B10 * np.sin(delta10))
    P_load_calc = -P1_inj
    return np.array([V0_NOMINAL, P0_calc, Q0_calc, P_load_calc])

def jacobian_full(x_k):
    """Jacobian H of h(x) w.r.t. state x_k."""
    V1, theta1_rad = x_k
    theta0_rad = 0.0
    delta01 = theta0_rad - theta1_rad
    delta10 = -delta01
    
    dP0_dV1 = V0_NOMINAL * (G01 * np.cos(delta01) + B01 * np.sin(delta01))
    dP0_dtheta1 = V0_NOMINAL * V1 * (G01 * np.sin(delta01) - B01 * np.cos(delta01))
    dQ0_dV1 = V0_NOMINAL * (G01 * np.sin(delta01) - B01 * np.cos(delta01))
    dQ0_dtheta1 = V0_NOMINAL * V1 * (-G01 * np.cos(delta01) - B01 * np.sin(delta01))
    
    dP1_dV1 = 2 * V1 * G11 + V0_NOMINAL * (G10 * np.cos(delta10) + B10 * np.sin(delta10))
    dP1_dtheta1 = V1 * V0_NOMINAL * (-G10 * np.sin(delta10) + B10 * np.cos(delta10))
    
    return np.array([
        [0, 0],
        [dP0_dV1, dP0_dtheta1],
        [dQ0_dV1, dQ0_dtheta1],
        [-dP1_dV1, -dP1_dtheta1]
    ])

def calculate_load_from_state(x_k):
    """Calculate load P, Q from estimated state vector x_k."""
    V1_est, theta1_est_rad = x_k
    theta0_rad = 0.0
    delta10 = theta1_est_rad - theta0_rad
    P1_inj = V1_est**2 * G11 + V1_est * V0_NOMINAL * (G10 * np.cos(delta10) + B10 * np.sin(delta10))
    Q1_inj = -V1_est**2 * B11 + V1_est * V0_NOMINAL * (G10 * np.sin(delta10) - B10 * np.cos(delta10))
    return -P1_inj, -Q1_inj

# =============================================================================
# Step 4: Generate measurements with "bad data"
# =============================================================================
std_dev_v_slack = 0.002
std_dev_p_slack = 0.002
std_dev_q_slack = 0.002
std_dev_p_load_BAD = 0.05

v_meas = net_true.res_bus.vm_pu[bus_slack] + np.random.normal(0, std_dev_v_slack)
p_meas_slack = net_true.res_line.p_from_mw[0] + np.random.normal(0, std_dev_p_slack)
q_meas_slack = net_true.res_line.q_from_mvar[0] + np.random.normal(0, std_dev_q_slack)
p_meas_load_BAD = TRUE_P_LOAD_MW + np.random.normal(0, std_dev_p_load_BAD)

z = np.array([v_meas, p_meas_slack, q_meas_slack, p_meas_load_BAD])
variances = np.array([std_dev_v_slack**2, std_dev_p_slack**2, std_dev_q_slack**2, std_dev_p_load_BAD**2])
R_inv = np.diag(1 / variances)

# =============================================================================
# Step 5: State Estimation Algorithm (LS / WLS)
# =============================================================================
def estimate_state(z, method='WLS', R_inv=None, max_iter=10, tolerance=1e-6):
    """Iterative Gauss-Newton state estimation (LS or WLS)."""
    x_k = np.array([1.0, 0.0])  # Flat start
    for i in range(max_iter):
        H = jacobian_full(x_k)
        r = z - h_full(x_k)
        if method == 'WLS':
            delta_x = np.linalg.pinv(H.T @ R_inv @ H) @ H.T @ R_inv @ r
        else:  # LS
            delta_x = np.linalg.pinv(H.T @ H) @ H.T @ r
        x_k += delta_x
        if np.linalg.norm(delta_x) < tolerance:
            print(f"Estimation ({method}) converged after {i+1} iterations.")
            break
    else:
        print(f"Estimation ({method}) did not converge after {max_iter} iterations.")
    return x_k

x_ls = estimate_state(z, method='LS')
x_wls = estimate_state(z, method='WLS', R_inv=R_inv)

v_ls, theta_ls_deg = x_ls[0], np.rad2deg(x_ls[1])
p_ls, q_ls = calculate_load_from_state(x_ls)

v_wls, theta_wls_deg = x_wls[0], np.rad2deg(x_wls[1])
p_wls, q_wls = calculate_load_from_state(x_wls)

# =============================================================================
# Step 6: Present results
# =============================================================================
results = pd.DataFrame({
    "Metoda": ["True Value", "LS Implementation", "WLS Implementation"],
    "V [p.u.]": [true_v_load, v_ls, v_wls],
    "θ [°]": [true_theta_load, theta_ls_deg, theta_wls_deg],
    "P [MW]": [TRUE_P_LOAD_MW, p_ls, p_wls],
    "Q [MVAr]": [TRUE_Q_LOAD_MVAR, q_ls, q_wls]
})

# Compute errors
results['Błąd θ [°]'] = abs(results['θ [°]'] - true_theta_load)
results['Błąd V [%]'] = 100 * abs(results['V [p.u.]'] - true_v_load) / true_v_load
results['Błąd P [%]'] = 100 * abs(results['P [MW]'] - TRUE_P_LOAD_MW) / TRUE_P_LOAD_MW
results['Błąd Q [%]'] = 100 * abs(results['Q [MVAr]'] - TRUE_Q_LOAD_MVAR) / TRUE_Q_LOAD_MVAR
results.loc[0, ['Błąd θ [°]', 'Błąd V [%]', 'Błąd P [%]', 'Błąd Q [%]']] = 0.0

pd.set_option('display.width', 120)
pd.set_option('display.float_format', '{:.5f}'.format)

print("\n" + "="*85)
print(results.to_string(index=False))
print("="*85 + "\n")