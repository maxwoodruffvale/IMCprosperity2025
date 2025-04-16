import re
import subprocess

def run_backtest():
    cmd = ["prosperity3bt", "round3v2.py", "3", "--merge-pnl"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()

    # Look for final total profit line
    match = re.search(r"Total profit:\s*([\-\d,]+)", out)
    if match:
        profit_str = match.group(1).replace(",", "")
        return int(profit_str)
    else:
        return None  # or -999999

# Then in your loops:
final_pnl = run_backtest()
print("Got final PnL:", final_pnl)
