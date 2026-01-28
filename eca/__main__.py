"""Entry point for minimal ECA prototype.

Usage: python -m eca --file examples/example.py --fn sum_squares --arg 100000 --time 10
"""
import argparse
from .core import OptimizeRunner
from .evolution import EvolutionaryOptimizer


def main():
    parser = argparse.ArgumentParser(description="ECA minimal optimizer")
    parser.add_argument("--file", required=True, help="Target python file")
    parser.add_argument("--fn", required=True, help="Target function name")
    parser.add_argument("--arg", required=True, help="Single numeric arg to pass to function")
    parser.add_argument("--time", type=int, default=10, help="Seconds budget for optimization")
    parser.add_argument("--mode", choices=['greedy','ea'], default='greedy', help="Optimization mode: 'greedy' simple transforms, 'ea' use evolutionary search")
    args = parser.parse_args()

    if args.mode == 'greedy':
        runner = OptimizeRunner(target_file=args.file, fn_name=args.fn, arg=args.arg, time_budget=args.time)
        runner.run()
    else:
        src = open(args.file, 'r', encoding='utf8').read()
        evo = EvolutionaryOptimizer(original_source=src, fn_name=args.fn, arg=args.arg, time_budget=args.time)
        best_src, best_time = evo.run()
        print('EA best time=', best_time)
        if best_src:
            out_path = args.file + '.eca_ea_best.py'
            open(out_path, 'w', encoding='utf8').write(best_src)
            print('Wrote best variant to', out_path)


if __name__ == '__main__':
    main()
