import time
def read_system_cpu_stat():
    try:
        with open("/proc/stat", "r") as f:
            for line in f:
                if line.startswith("cpu "):
                    parts = [float(x) for x in line.split()[1:]]
                    idle = parts[3] + parts[4]
                    total = sum(parts)
                    return idle, total
    except Exception:
        pass
    return 0.0, 0.0

idle1, total1 = read_system_cpu_stat()
time.sleep(0.5)
idle2, total2 = read_system_cpu_stat()
diff_idle = idle2 - idle1
diff_total = total2 - total1
usage = 100.0 * (diff_total - diff_idle) / diff_total if diff_total > 0 else 0.0
print(f"CPU Usage: {usage:.2f}%")
