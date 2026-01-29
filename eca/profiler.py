import cProfile
import pstats
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple


def _make_runner_code(module_path: Path, fn_name: str, arg: str):
    # Create a runner that loads a module from a file path and calls the given function.
    return f"""
import importlib.util
from pathlib import Path
mp = Path(r'{str(module_path)}')
spec = importlib.util.spec_from_file_location('target_mod', mp)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.{fn_name}({arg})
"""


def detect_hotspot_functions(target_file: str, fn_name: str, arg: str, runs: int = 3, top_n: int = 3) -> List[Tuple[str,int]]:
    """Run the target file under cProfile multiple times and return top function names and call counts.

    Returns list of tuples `(func_name, ncalls)` ordered by cumulative time.
    """
    p = Path(target_file)
    tmp = Path(tempfile.gettempdir())
    stats_file = tmp / f"eca_profile_{p.stem}.prof"

    # build runner file
    runner_path = tmp / f"eca_profile_runner_{p.stem}.py"
    runner_code = _make_runner_code(p, fn_name, arg)
    runner_path.write_text(runner_code, encoding='utf8')

    # run subprocess under cProfile to generate stats file
    for _ in range(runs):
        try:
            subprocess.run([sys.executable, '-m', 'cProfile', '-o', str(stats_file), str(runner_path)], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            # if runner fails, continue
            continue

    # read stats
    ps = pstats.Stats(str(stats_file))
    ps.strip_dirs()
    ps.sort_stats('cumulative')
    stats = []
    count = 0
    for func, stat in ps.stats.items():
        if count >= top_n:
            break
        cc, nc, tt, ct, callers = stat
        # func is a tuple (filename, line, name)
        func_name = func[2] if len(func) > 2 else str(func)
        filename = func[0] if len(func) > 0 else ''
        # keep only functions from the target file or matching the target function name
        if filename and Path(filename).name == p.name:
            stats.append((func_name, nc))
            count += 1
        elif func_name == fn_name:
            stats.append((func_name, nc))
            count += 1

    try:
        runner_path.unlink()
    except Exception:
        pass
    try:
        stats_file.unlink()
    except Exception:
        pass
    return stats
