"""Загрузка закреплённой модели с докачкой, журналом прогресса и проверкой хешей."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from datetime import datetime, timezone
import httpx

BASE = Path(__file__).resolve().parents[1]
DEST = BASE / 'CapRL-Qwen3VL-2B'
STATE = BASE / '.downloads/caprl_status.json'
REPO = 'internlm/CapRL-Qwen3VL-2B'
REVISION = 'b838cec8b6c4791ed6169d12db21db0b47bb70e5'
DEST.mkdir(parents=True, exist_ok=True)
lock = (BASE / '.downloads/caprl.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
started = time.monotonic()
state = {'model': REPO, 'revision': REVISION, 'pid': os.getpid(),
         'started_at': datetime.now(timezone.utc).isoformat(), 'directory': str(DEST)}
files = []

def saved_bytes():
    total = 0
    for item in files:
        path = DEST / item['rfilename']
        partial = path.with_name(path.name + '.partial')
        total += path.stat().st_size if path.exists() else partial.stat().st_size if partial.exists() else 0
    return total

def publish(phase, **values):
    state.update(state=phase, updated_at=datetime.now(timezone.utc).isoformat(), **values)
    if files:
        state.update(downloaded_bytes=saved_bytes(), total_bytes=sum(f['size'] for f in files))
    temp = STATE.with_suffix('.tmp')
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    temp.replace(STATE)

def verify(path, item):
    if path.stat().st_size != item['size']:
        raise RuntimeError('Неверный размер: ' + item['rfilename'])
    lfs = item.get('lfs')
    digest = hashlib.sha256() if lfs else hashlib.sha1()
    if not lfs:
        digest.update(f"blob {item['size']}\0".encode())
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    expected = lfs['sha256'] if lfs else item['blobId']
    if digest.hexdigest() != expected:
        raise RuntimeError('Контрольная сумма не совпала: ' + item['rfilename'])
    return digest.hexdigest()

try:
    publish('metadata')
    with httpx.Client(timeout=httpx.Timeout(90, connect=20), follow_redirects=True) as client:
        for attempt in range(6):
            try:
                response = client.get(f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}', params={'blobs': 'true'})
                response.raise_for_status()
                meta = response.json()
                assert meta['sha'] == REVISION
                break
            except Exception:
                if attempt == 5:
                    raise
                publish('retrying_metadata', attempt=attempt + 1)
                time.sleep(15)
        files = sorted([f for f in meta['siblings'] if not f['rfilename'].startswith('assets/')], key=lambda f: f['size'])
        for item in files:
            name = Path(item['rfilename'])
            if name.is_absolute() or '..' in name.parts:
                raise RuntimeError('Недопустимый путь в репозитории')
        required = sum(f['size'] for f in files) - saved_bytes()
        if shutil.disk_usage(DEST).free < required + 512 * 1024**2:
            raise RuntimeError('Недостаточно места для модели и запаса 512 МиБ')
        (BASE / '.downloads/caprl_manifest.json').write_text(json.dumps({'model': REPO, 'revision': REVISION, 'files': files}, indent=2))
        verified = {}
        for item in files:
            name = item['rfilename']
            target = DEST / name
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + '.partial')
            if target.exists():
                publish('verifying', current_file=name)
                verified[name] = verify(target, item)
                continue
            for attempt in range(12):
                offset = partial.stat().st_size if partial.exists() else 0
                if offset == item['size']:
                    break
                try:
                    publish('downloading', current_file=name, attempt=attempt + 1)
                    headers = {'Accept-Encoding': 'identity'}
                    if offset:
                        headers['Range'] = f'bytes={offset}-'
                    url = f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}'
                    with client.stream('GET', url, headers=headers, params={'download': 'true', 'probe_time': str(time.time_ns())}) as response:
                        response.raise_for_status()
                        if offset and response.status_code == 206:
                            if not response.headers.get('content-range', '').startswith(f'bytes {offset}-'):
                                raise RuntimeError('Сервер вернул неверный диапазон байтов')
                        elif response.status_code == 200:
                            offset = 0
                        else:
                            raise RuntimeError('Неожиданный HTTP-статус загрузки')
                        checkpoint = time.monotonic()
                        transferred = 0
                        with partial.open('ab' if offset else 'wb') as stream:
                            for block in response.iter_bytes(1024**2):
                                if offset + transferred + len(block) > item['size']:
                                    raise RuntimeError('Ответ превышает размер файла')
                                if shutil.disk_usage(DEST).free < 256 * 1024**2:
                                    raise RuntimeError('Свободного места осталось меньше 256 МиБ')
                                stream.write(block)
                                transferred += len(block)
                                now = time.monotonic()
                                if now - checkpoint >= 5:
                                    stream.flush()
                                    publish('downloading', current_file=name)
                                    checkpoint = now
                    if partial.stat().st_size != item['size']:
                        raise RuntimeError('Соединение закрыто до завершения файла')
                    break
                except (httpx.HTTPError, RuntimeError) as exc:
                    publish('retrying', current_file=name, error=type(exc).__name__, attempt=attempt + 1)
                    print(f'{name}: {type(exc).__name__}, попытка {attempt + 1}', flush=True)
                    if attempt == 11:
                        raise
                    time.sleep(min(10 * (attempt + 1), 60))
            publish('verifying', current_file=name)
            verified[name] = verify(partial, item)
            partial.replace(target)
            print('Проверен ' + name, flush=True)
        publish('completed', current_file=None, verified_hashes=verified, elapsed_seconds=round(time.monotonic() - started))
        print('Модель скачана и проверена', flush=True)
except Exception as exc:
    publish('failed', error_type=type(exc).__name__)
    print('Загрузка остановлена: ' + type(exc).__name__, flush=True)
    raise SystemExit(1)
