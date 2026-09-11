import numpy as np

def estimate_fuel(tick_window: list[dict]) -> dict:
    """
    Physics-based fuel estimation.
    Uses MAF (Tier 1), MAP+RPM (Tier 2), or Throttle+RPM (Tier 3) depending on sensor availability.
    """
    if not tick_window:
        return {
            "fcr_gs": 0.0,
            "mileage_kmpl": None,
            "vss_kmph": 0.0,
            "method": "stopped",
            "tier": 0
        }

    # Noise averaging: average the last 3 ticks
    last_ticks = tick_window[-3:] if len(tick_window) >= 3 else tick_window
    
    def get_avg(key):
        vals = [t.get(key, 0.0) for t in last_ticks]
        return sum(vals) / len(vals) if vals else 0.0

    rpm = get_avg("rpm")
    vss = get_avg("vss")
    maf = get_avg("maf")
    map_kpa = get_avg("map_kpa")
    iat = get_avg("intake_air_temp")
    throttle = get_avg("throttle_pos")

    FCR_gs = 0.0
    tier = 0
    method = "stopped"

    # TIER 0: Engine off
    if rpm <= 200:
        tier = 0
        method = "stopped"
        FCR_gs = 0.0
    # TIER 1: MAF-based
    elif maf > 0.5:
        tier = 1
        method = "maf"
        AFR_stoich = 14.7
        FCR_gs = maf / AFR_stoich
    # TIER 2: MAP + RPM speed-density
    elif map_kpa * 1000 > 20000 and (iat + 273.15) > 233:
        tier = 2
        method = "map_rpm"
        VE = 0.85
        Disp = 1.598
        R_air = 287.05
        MAP_pa = map_kpa * 1000
        IAT_k = iat + 273.15
        rho_air = MAP_pa / (R_air * IAT_k)
        air_gs = (VE * rho_air * (Disp / 1000) * rpm) / (2 * 60) * 1000
        FCR_gs = air_gs / 14.7
    # TIER 3: Throttle + RPM heuristic
    else:
        tier = 3
        method = "throttle_heuristic"
        throttle_frac = throttle / 100.0
        engine_load = (throttle_frac * rpm) / 6000.0
        FCR_gs = max(0.0, 0.06 + 2.8 * engine_load)

    # MILEAGE
    FUEL_DENSITY_GL = 0.730
    mileage_kmpl = None
    if vss > 2.0:
        mileage_kmpl = min((vss / 3600.0) / (FCR_gs / FUEL_DENSITY_GL / 1000 + 1e-9), 50.0)

    return {
        "fcr_gs": FCR_gs,
        "mileage_kmpl": mileage_kmpl,
        "vss_kmph": vss,
        "method": method,
        "tier": tier
    }
