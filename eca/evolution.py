"""Simple DEAP-based evolutionary optimizer for ECA minimal prototype.

This module implements a lightweight evolutionary loop where individuals
are sequences of transformation operator indices applied to the original
source. It uses DEAP if available; otherwise falls back to a simple
random-search loop.
"""
import random
import time
import tempfile
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

try:
    from deap import base, creator, tools  # type: ignore
    DEAP_AVAILABLE = True
except Exception:
    DEAP_AVAILABLE = False

from .core import try_apply_transform, add_decorator_to_function


OPS = [
    'noop',
    'lru_cache',
    'numba_njit',
    'loop_to_comp',
]


def apply_ops_to_source(src: str, fn_name: str, ops_seq: List[int], hotspots: Optional[List[str]] = None) -> Optional[str]:
    cur = src
    for op in ops_seq:
        if op < 0 or op >= len(OPS):
            continue
        name = OPS[op]
        if name == 'noop':
            continue
        if name == 'lru_cache':
            # only apply to function if hotspots is None or fn_name is among hotspots
            if hotspots is None or fn_name in hotspots:
                dec = 'from functools import lru_cache\\n@lru_cache(maxsize=None)'
                v = add_decorator_to_function(cur, fn_name, dec)
                if v:
                    cur = v
        elif name == 'numba_njit':
            try:
                import numba  # type: ignore
                dec = 'from numba import njit\\n@njit'
                v = add_decorator_to_function(cur, fn_name, dec)
                if v:
                    cur = v
            except Exception:
                # skip if numba not available
                pass
        elif name == 'loop_to_comp':
            # loop->comp transform is function-agnostic but we only accept if hotspots is None
            if hotspots is None:
                v = try_apply_transform(cur)
                if v and v != cur:
                    cur = v
    return cur


def _write_temp_source(source: str) -> Path:
    tmp_dir = Path(tempfile.gettempdir())
    fname = f"eca_ea_tmp_{random.randrange(10**9)}.py"
    path = tmp_dir / fname
    path.write_text(source, encoding='utf8')
    return path


def _run_source_and_get_outputs(src_path: Path, fn_name: str, args_list: List[str], timeout: int = 5):
    runner = [
        f"from {src_path.stem} import {fn_name}",
    ]
    for a in args_list:
        runner.append(f"print('OUT', {a}, repr({fn_name}({a})))")
    code = '\n'.join(runner)
    runner_path = src_path.with_name(f"runner_{src_path.stem}_{random.randrange(10**9)}.py")
    runner_path.write_text(code, encoding='utf8')
    try:
        proc = subprocess.run([sys.executable, str(runner_path)], capture_output=True, text=True, timeout=timeout)
        out = proc.stdout
        results = []
        for line in out.splitlines():
            if line.startswith('OUT'):
                parts = line.split(None, 2)
                if len(parts) >= 3:
                    results.append(parts[2])
        return results
    except subprocess.TimeoutExpired:
        return None
    finally:
        try:
            runner_path.unlink()
        except Exception:
            pass


def time_variant_and_get_time(source: str, fn_name: str, arg: str, timeout: int = 5):
    src_path = _write_temp_source(source)
    runner_code = f"""
from {src_path.stem} import {fn_name}
import time
start = time.perf_counter()
res = {fn_name}({arg})
end = time.perf_counter()
print('TIME', end-start)
print('RESULT', res)
"""
    runner_path = src_path.with_name(f"runner_time_{src_path.stem}_{random.randrange(10**9)}.py")
    runner_path.write_text(runner_code, encoding='utf8')
    try:
        proc = subprocess.run([sys.executable, str(runner_path)], capture_output=True, text=True, timeout=timeout)
        out = proc.stdout
        t = None
        for line in out.splitlines():
            if line.startswith('TIME'):
                try:
                    t = float(line.split(None, 1)[1])
                except Exception:
                    t = None
        return t
    except subprocess.TimeoutExpired:
        return None
    finally:
        try:
            runner_path.unlink()
            src_path.unlink()
        except Exception:
            pass


class EvolutionaryOptimizer:
    def __init__(self, original_source: str, fn_name: str, arg: str, time_budget: int = 10, test_args=None, hotspots: Optional[List[str]] = None):
        self.src = original_source
        self.fn_name = fn_name
        self.arg = arg
        self.time_budget = time_budget
        self.test_args = test_args or ['0', '1', '10']
        self.hotspots = hotspots
        src_path = _write_temp_source(self.src)
        baseline = _run_source_and_get_outputs(src_path, self.fn_name, self.test_args)
        if baseline is None:
            raise RuntimeError('Failed to compute baseline outputs')
        self.baseline = baseline

    def _passes_correctness(self, candidate_src: str) -> bool:
        src_path = _write_temp_source(candidate_src)
        outs = _run_source_and_get_outputs(src_path, self.fn_name, self.test_args)
        if outs is None:
            return False
        return outs == self.baseline

    def run_random_search(self, ops_len=3, iterations=50):
        best = (None, float('inf'))
        start = time.time()
        for i in range(iterations):
            if time.time() - start > self.time_budget:
                break
            ops_seq = [random.randrange(len(OPS)) for _ in range(ops_len)]
            cand = apply_ops_to_source(self.src, self.fn_name, ops_seq, hotspots=self.hotspots)
            if not cand:
                continue
            if not self._passes_correctness(cand):
                continue
            t = time_variant_and_get_time(cand, self.fn_name, self.arg)
            if t is None:
                continue
            if t < best[1]:
                best = (cand, t)
        return best

    def run_deap(self):
        if not DEAP_AVAILABLE:
            return self.run_random_search()

        OPS_COUNT = len(OPS)
        seq_len = 3
        if not hasattr(creator, 'FitnessMin'):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, 'Individual'):
            creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register('attr_op', random.randrange, OPS_COUNT)
        toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.attr_op, n=seq_len)
        toolbox.register('population', tools.initRepeat, list, toolbox.individual)

        def eval_ind(ind):
            cand = apply_ops_to_source(self.src, self.fn_name, list(ind), hotspots=self.hotspots)
            if not cand:
                return (float('inf'),)
            if not self._passes_correctness(cand):
                return (float('inf'),)
            t = time_variant_and_get_time(cand, self.fn_name, self.arg)
            if t is None:
                return (float('inf'),)
            return (t,)

        toolbox.register('evaluate', eval_ind)
        toolbox.register('mate', tools.cxTwoPoint)
        toolbox.register('mutate', tools.mutUniformInt, low=0, up=OPS_COUNT-1, indpb=0.2)
        toolbox.register('select', tools.selTournament, tournsize=3)

        pop = toolbox.population(n=10)
        start = time.time()
        best = (None, float('inf'))
        while time.time() - start < self.time_budget:
            fitnesses = list(map(toolbox.evaluate, pop))
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit
                if fit[0] < best[1]:
                    cand = apply_ops_to_source(self.src, self.fn_name, list(ind), hotspots=self.hotspots)
                    best = (cand, fit[0])
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.5:
                    tools.cxTwoPoint(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            for mutant in offspring:
                if random.random() < 0.2:
                    tools.mutUniformInt(mutant, low=0, up=OPS_COUNT-1, indpb=0.2)
                    del mutant.fitness.values
            pop[:] = offspring

        return best

    def run(self):
        if DEAP_AVAILABLE:
            return self.run_deap()
        else:
            return self.run_random_search()
