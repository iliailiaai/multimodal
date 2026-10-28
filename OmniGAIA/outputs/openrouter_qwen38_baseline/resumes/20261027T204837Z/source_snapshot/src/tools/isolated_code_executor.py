"""Выполнение Python в отдельном процессе без сети, ключей и файлов теста."""

import asyncio
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

from tools.code_executor import ALLOWED_IMPORTS, UNSAFE_PATTERNS


CHILD = r'''
import ast, contextlib, json, resource, sys
resource.setrlimit(resource.RLIMIT_CPU, (25, 25))
resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
resource.setrlimit(resource.RLIMIT_FSIZE, (1024**2, 1024**2))
request = json.load(sys.stdin)
scope = {}
result = {"stdout": "", "result": None, "error": None}
try:
    exec(request["imports"], scope)
    # Вывод идёт в ограниченный файл, а не в неограниченный буфер памяти.
    with open("/tmp/stdout", "w+") as output:
        with contextlib.redirect_stdout(output):
            tree = ast.parse(request["code"])
            last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
            exec(compile(tree, "<agent>", "exec"), scope)
            if last is not None:
                value = eval(compile(ast.Expression(last.value), "<agent>", "eval"), scope)
                if value is not None:
                    result["result"] = str(value)
        output.seek(0)
        result["stdout"] = output.read(32768).strip()
except BaseException as exc:
    result["error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(result, ensure_ascii=False))
'''


def sandbox_command():
    if not shutil.which("bwrap"):
        raise RuntimeError("Для Python-инструмента нужен bubblewrap (bwrap).")
    command = [
        "bwrap", "--unshare-all", "--die-with-parent", "--new-session",
        "--clearenv", "--setenv", "PATH", "/usr/bin:/bin",
        "--setenv", "OPENBLAS_NUM_THREADS", "1",
        "--setenv", "OMP_NUM_THREADS", "1",
        "--setenv", "PYTHONIOENCODING", "utf-8",
    ]
    # Монтируются только библиотеки и интерпретатор. Репозиторий с ответами
    # и домашняя директория пользователя внутри процесса недоступны.
    for name in ["/usr", "/lib", "/lib64", sys.base_prefix, sys.prefix]:
        if Path(name).exists():
            command += ["--ro-bind", name, name]
    command += [
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
        "--tmpfs", "/work", "--chdir", "/work",
        sys.executable, "-I", "-c", CHILD,
    ]
    return command


async def isolated_code_executor(code: str):
    if not isinstance(code, str) or len(code) > 100000:
        return {"stdout": "", "result": None, "error": "Недопустимый размер кода."}
    if any(re.search(pattern, code) for pattern in UNSAFE_PATTERNS):
        return {"stdout": "", "result": None, "error": "Unsafe code pattern detected."}
    request = json.dumps({"code": code, "imports": ALLOWED_IMPORTS}).encode()
    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
        process = await asyncio.create_subprocess_exec(
            *sandbox_command(), stdin=asyncio.subprocess.PIPE,
            stdout=output, stderr=errors, env={"PATH": os.defpath},
        )
        try:
            await asyncio.wait_for(process.communicate(request), timeout=30)
        except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
            if process.returncode is None:
                process.kill()
            await process.wait()
            if isinstance(exc, asyncio.CancelledError):
                raise
            return {"stdout": "", "result": None, "error": "Code execution timed out after 30 seconds"}
        output.seek(0)
        errors.seek(0)
        text = output.read(65536).decode("utf-8", errors="replace")
        if process.returncode:
            detail = errors.read(2000).decode("utf-8", errors="replace")
            raise RuntimeError(f"Python sandbox exited with {process.returncode}: {detail}")
        return json.loads(text)
