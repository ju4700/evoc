"""Entry point for minimal ECA prototype.

Usage: python -m eca --file examples/example.py --fn sum_squares --arg 100000 --time 10
"""
import argparse
from .core import OptimizeRunner
from .evolution import EvolutionaryOptimizer
import shutil
import time
import sys
from pathlib import Path


def apply_with_typewriter(path: str, content: str, delay: float = 0.002):
    """Overwrite `path` progressively with `content`, showing a typewriter effect in terminal.

    A backup `<path>.eca_backup` is created before modification.
    """
    p = Path(path)
    backup = p.with_suffix(p.suffix + '.eca_backup')
    shutil.copy2(p, backup)
    # write progressively in chunks (line by line)
    with open(p, 'w', encoding='utf8') as f:
        for line in content.splitlines(True):
            f.write(line)
            f.flush()
            # echo to terminal as typewriter
            for ch in line:
                sys.stdout.write(ch)
                sys.stdout.flush()
                time.sleep(delay)
    print('\n-- applied in-place (backup at {})'.format(backup))


def main():
    parser = argparse.ArgumentParser(description="ECA minimal optimizer")
    parser.add_argument("--file", required=True, help="Target python file")
    parser.add_argument("--fn", required=True, help="Target function name")
    parser.add_argument("--arg", required=True, help="Single numeric arg to pass to function")
    parser.add_argument("--time", type=int, default=10, help="Seconds budget for optimization")
    parser.add_argument("--mode", choices=['greedy','ea'], default='greedy', help="Optimization mode: 'greedy' simple transforms, 'ea' use evolutionary search")
    parser.add_argument("--inplace", action='store_true', default=True, help="Apply best variant directly into the target file with typewriter animation")
    parser.add_argument("--delay", type=float, default=0.002, help="Delay per character for typewriter animation (seconds)")
    parser.add_argument("--detect-hotspots", action='store_true', help="Run profiler to detect hotspots and focus evolution on them")
    args = parser.parse_args()

    if args.mode == 'greedy':
        runner = OptimizeRunner(target_file=args.file, fn_name=args.fn, arg=args.arg, time_budget=args.time)
        runner.generate_variants()
        best_name = None
        best_time = float('inf')
        best_src = None
        start = time.time()
        for i, (name, src) in enumerate(runner.variants):
            if time.time() - start > args.time:
                break
            print(f"Testing variant: {name}")
            rc, t, out, err = runner.write_variant_and_run(name, src, i)
            print(out)
            if t is not None and t < best_time:
                best_time = t
                best_name = name
                best_src = src
        print('Best:', best_name, best_time)
        if best_src:
            if args.inplace:
                # clean duplicate imports/decorators before applying
                try:
                    from .core import dedupe_source
                    best_src = dedupe_source(best_src)
                except Exception:
                    pass
                apply_with_typewriter(args.file, best_src, delay=args.delay)
            else:
                out_path = args.file + '.eca_best.py'
                open(out_path, 'w', encoding='utf8').write(best_src)
                print('Wrote best variant to', out_path)
    else:
        src = open(args.file, 'r', encoding='utf8').read()
        hotspots = None
        if args.detect_hotspots:
            try:
                from .profiler import detect_hotspot_functions
                stats = detect_hotspot_functions(args.file, args.fn, args.arg, runs=2, top_n=3)
                hotspots = [name for name, _ in stats]
                print('Detected hotspots:', hotspots)
            except Exception as e:
                print('Hotspot detection failed:', e)
                hotspots = None
        evo = EvolutionaryOptimizer(original_source=src, fn_name=args.fn, arg=args.arg, time_budget=args.time, hotspots=hotspots)
        best_src, best_time = evo.run()
        print('EA best time=', best_time)
        if best_src:
            if args.inplace:
                apply_with_typewriter(args.file, best_src, delay=args.delay)
            else:
                out_path = args.file + '.eca_ea_best.py'
                open(out_path, 'w', encoding='utf8').write(best_src)
                print('Wrote best variant to', out_path)


if __name__ == '__main__':
    main()
