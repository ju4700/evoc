import ast
import copy
import inspect
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from textwrap import dedent


class LoopToCompTransformer(ast.NodeTransformer):
    """Transform simple for-loop that appends to list into a list comprehension.

    Only handles patterns like:
        out = []
        for x in xs:
            out.append(f(x))
        return out
    """

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        new_body = []
        i = 0
        while i < len(node.body):
            stmt = node.body[i]
            if (i + 1 < len(node.body) and isinstance(stmt, ast.Assign)
                    and isinstance(node.body[i+1], ast.For)):
                assign = stmt
                for_node = node.body[i+1]
                # detect pattern: assign target = [] and for: target.append(expr)
                if (len(assign.targets) == 1 and isinstance(assign.targets[0], ast.Name)
                        and isinstance(assign.value, ast.List) and assign.value.elts == []):
                    target_name = assign.targets[0].id
                    # inspect for append in for body
                    if (len(for_node.body) == 1 and isinstance(for_node.body[0], ast.Expr)
                            and isinstance(for_node.body[0].value, ast.Call)):
                        call = for_node.body[0].value
                        if (isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name)
                                and call.func.attr == 'append' and call.func.value.id == target_name):
                            # build comprehension
                            elt = call.args[0]
                            comp = ast.Assign(targets=[ast.Name(id=target_name, ctx=ast.Store())],
                                              value=ast.ListComp(elt=elt, generators=[ast.comprehension(target=for_node.target, iter=for_node.iter, ifs=[], is_async=0)]))
                            new_body.append(comp)
                            i += 2
                            continue
            new_body.append(stmt)
            i += 1
        node.body = new_body
        return node


def try_apply_transform(source: str):
    try:
        tree = ast.parse(source)
        transformer = LoopToCompTransformer()
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)
        return ast.unparse(new_tree)
    except Exception:
        return None


def add_decorator_to_function(source: str, fn_name: str, decorator_src: str):
    tree = ast.parse(source)
    modified = False
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            dec = ast.parse(decorator_src).body[0].value
            node.decorator_list.insert(0, dec)
            modified = True
    if not modified:
        return None
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


class OptimizeRunner:
    def __init__(self, target_file: str, fn_name: str, arg: str, time_budget: int = 10):
        self.target_file = Path(target_file)
        self.fn_name = fn_name
        self.arg = arg
        self.time_budget = time_budget
        self.variants = []

    def load_source(self):
        return self.target_file.read_text()

    def generate_variants(self):
        src = self.load_source()
        self.variants = []
        # original
        self.variants.append(('original', src))

        # lru_cache
        lru_src = 'from functools import lru_cache\n@lru_cache(maxsize=None)'
        v = add_decorator_to_function(src, self.fn_name, lru_src)
        if v:
            self.variants.append(('lru_cache', v))

        # optional numba
        try:
            import numba  # type: ignore
            numba_src = 'from numba import njit\n@njit'
            v2 = add_decorator_to_function(src, self.fn_name, numba_src)
            if v2:
                self.variants.append(('numba_njit', v2))
        except Exception:
            pass

        # loop->comprehension
        t = try_apply_transform(src)
        if t and t != src:
            self.variants.append(('loop_to_comp', t))

    def write_variant_and_run(self, variant_name: str, source: str, run_id: int):
        tmp_dir = tempfile.gettempdir()
        fname = Path(tmp_dir) / f"eca_variant_{run_id}_{variant_name}.py"
        fname.write_text(source)

        # create runner code to import function and time it
        runner = dedent(f"""
        import time
        from {fname.stem} import {self.fn_name}

        def bench():
            start = time.perf_counter()
            res = {self.fn_name}({self.arg})
            end = time.perf_counter()
            print('RESULT', res)
            print('TIME', end - start)

        if __name__ == '__main__':
            bench()
        """)

        runner_file = Path(tmp_dir) / f"eca_runner_{run_id}.py"
        runner_file.write_text(runner)

        # run subprocess and capture output
        try:
            proc = subprocess.run([sys.executable, str(runner_file)], capture_output=True, text=True, timeout=10)
            out = proc.stdout
            err = proc.stderr
            # parse TIME
            tline = None
            for line in out.splitlines():
                if line.startswith('TIME'):
                    tline = float(line.split(None, 1)[1])
            return (proc.returncode, tline, out, err)
        except subprocess.TimeoutExpired:
            return (None, None, '', 'timeout')

    def run(self):
        print(f"Optimizing {self.target_file}::{self.fn_name} with budget {self.time_budget}s")
        self.generate_variants()
        print(f"Found {len(self.variants)} variants")
        results = []
        start = time.time()
        for i, (name, src) in enumerate(self.variants):
            if time.time() - start > self.time_budget:
                break
            print(f"Testing variant: {name}")
            code, t, out, err = self.write_variant_and_run(name, src, i)
            results.append((name, code, t, out, err))

        # report
        print('\nResults:')
        for r in results:
            name, code, t, out, err = r
            print('-', name, 'time=', t, 'rc=', code)
