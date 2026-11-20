"""Resource limits for the Web UI generation request and its references."""
from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
from contextlib import aclosing, contextmanager, nullcontext
from pathlib import Path
from tempfile import SpooledTemporaryFile, mkdtemp
from typing import BinaryIO

from anyio import CancelScope
from fastapi import HTTPException, Request, UploadFile
from fastapi.routing import APIRoute
from filelock import FileLock, Timeout
from PIL import Image
from starlette.concurrency import run_in_threadpool
from starlette.formparsers import MultiPartException, MultiPartParser, parse_options_header

from remote.job_paths import confined_path
from remote.manifest import MAX_REFERENCES

MIB = 1024 * 1024
MAX_REFERENCE_BYTES = 35 * MIB
MAX_FORM_BYTES = 2 * MIB
MAX_REQUEST_BYTES = MAX_REFERENCES * MAX_REFERENCE_BYTES + MAX_FORM_BYTES
MAX_FORM_FIELDS = 24
MAX_FIELD_BYTES = 64 * 1024
MAX_IMAGE_SIDE = 8192
MAX_IMAGE_PIXELS = 4096**2
IMAGE_FORMATS = ("PNG", "JPEG", "WEBP", "AVIF", "GIF", "BMP", "TIFF")
STAGING_DIR = ".webui-staging"
STAGING_MARKER = ".cstudio-web-staging"
LOGGER = logging.getLogger(__name__)


def _staging_lock(root: Path, name: str, *, timeout: float) -> FileLock:
    return FileLock(confined_path(root, name, missing_ok=True), timeout=timeout, mode=0o600)


@contextmanager
def generation_staging(jobs: Path):
    root = confined_path(jobs, STAGING_DIR, directory=True, missing_ok=True)
    root.mkdir(mode=0o700, exist_ok=True)
    gate = _staging_lock(root, ".lock", timeout=5)
    with gate:
        folder = Path(mkdtemp(prefix="pending-", dir=root))
        try:
            (folder / STAGING_MARKER).touch(exist_ok=False)
            lease = _staging_lock(folder, ".lease", timeout=0)
            lease.acquire()
        except (OSError, ValueError, Timeout):
            shutil.rmtree(folder, ignore_errors=True)
            raise
    try:
        yield folder
    finally:
        try:
            with gate:
                lease.release()
                shutil.rmtree(folder)
        except (OSError, Timeout):
            LOGGER.warning("暫時無法移除工作台暫存目錄，重新啟動時會再清理")
        finally:
            lease.release()


def clean_generation_staging(jobs: Path) -> None:
    try:
        root = confined_path(jobs, STAGING_DIR, directory=True)
    except FileNotFoundError:
        return
    with _staging_lock(root, ".lock", timeout=5):
        for path in root.iterdir():
            if not re.fullmatch(r"pending-[a-z0-9_]{8}", path.name):
                continue
            try:
                folder = confined_path(root, path.name, directory=True)
                confined_path(folder, STAGING_MARKER)
                with _staging_lock(folder, ".lease", timeout=0):
                    pass
                shutil.rmtree(folder)
            except (FileNotFoundError, ValueError, Timeout):
                # Another process may still hold this lease. Only remove managed directories.
                continue
            except OSError:
                LOGGER.warning("暫時無法清理一個工作台暫存目錄")


class ReferenceMultipartParser(MultiPartParser):
    complete = False

    def on_part_begin(self) -> None:
        super().on_part_begin()
        self.part_bytes = 0

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        self.part_bytes += end - start
        is_file = self._current_part.file is not None
        limit = MAX_REFERENCE_BYTES if is_file else MAX_FIELD_BYTES
        if self.part_bytes > limit:
            detail = "單張參考圖片超過 35 MiB 上限" if is_file else "表單欄位超過 64 KiB 上限"
            raise HTTPException(status_code=413, detail=detail)
        super().on_part_data(data, start, end)

    def on_end(self) -> None:
        self.complete = True
        super().on_end()

    async def parse(self):
        try:
            form = await super().parse()
            if not self.complete:
                raise MultiPartException("Incomplete multipart form")
            return form
        except BaseException:
            # Also close the spool of an unfinished part, which is absent from FormData.
            for file in self._files_to_close_on_error:
                file.close()
            raise


def is_multipart(request: Request) -> bool:
    media_type, _ = parse_options_header(request.headers.get("content-type"))
    return media_type == b"multipart/form-data"


class GenerationRequest(Request):
    async def _get_form(self, **_limits):
        if self._form is None:
            if not is_multipart(self):
                form = await super()._get_form(max_files=MAX_REFERENCES, max_fields=MAX_FORM_FIELDS)
                fields = form.multi_items()
                if len(fields) > MAX_FORM_FIELDS or any(
                    len(name.encode()) + len(value.encode()) > MAX_FIELD_BYTES for name, value in fields
                ):
                    raise HTTPException(status_code=400, detail="表單欄位大小或數量超過上限")
                return form
            try:
                async with aclosing(self.stream()) as stream:
                    self._form = await ReferenceMultipartParser(
                        self.headers, stream, max_files=MAX_REFERENCES,
                        max_fields=MAX_FORM_FIELDS,
                    ).parse()
            except MultiPartException as exc:
                raise HTTPException(status_code=400, detail=(
                    "上傳表單不完整、格式錯誤或超過欄位與檔案數上限"
                )) from exc
        return self._form


class GenerationRoute(APIRoute):
    def get_route_handler(self):
        handle = super().get_route_handler()

        async def limited(request: Request):
            limit = MAX_REQUEST_BYTES if is_multipart(request) else MAX_FORM_BYTES
            lengths = request.headers.getlist("content-length")
            if lengths and (len(lengths) != 1 or not re.fullmatch(r"[0-9]{1,20}", lengths[0])):
                raise HTTPException(status_code=400, detail="不合法的 Content-Length")
            too_large = f"生成請求總大小超過 {limit // MIB} MiB 上限"
            if lengths and int(lengths[0]) > limit:
                raise HTTPException(status_code=413, detail=too_large)

            received = 0

            async def receive():
                nonlocal received
                message = await request.receive()
                if message["type"] == "http.request":
                    received += len(message.get("body", b""))
                    if received > limit:
                        raise HTTPException(status_code=413, detail=too_large)
                return message

            loop = asyncio.get_running_loop()
            with CancelScope() as cancel:
                with request.app.state.reserve_generation(lambda: loop.call_soon_threadsafe(cancel.cancel)) as attempt:
                    guarded = GenerationRequest(request.scope, receive)
                    guarded.state.generation = attempt
                    try:
                        response = await handle(guarded)
                    finally:
                        with CancelScope(shield=True):
                            await guarded.close()
            if attempt.preparation_cancelled:
                raise HTTPException(status_code=409, detail="已取消本機生成準備")
            return response

        return limited


def copy_artifact(source: Path, target: BinaryIO) -> int:
    """Snapshot no more than one reference's byte budget before invoking Pillow."""
    size = 0
    with source.open("rb") as file:
        if os.fstat(file.fileno()).st_size > MAX_REFERENCE_BYTES:
            return MAX_REFERENCE_BYTES + 1
        while block := file.read(min(64 * 1024, MAX_REFERENCE_BYTES - size + 1)):
            size += len(block)
            if size > MAX_REFERENCE_BYTES:
                break
            target.write(block)
    target.seek(0)
    return size


def write_reference_images(sources: list[UploadFile | Path], folder: Path) -> None:
    for index, source in enumerate(sources, start=1):
        label = f"參考圖片 <image{index}>"
        try:
            is_artifact = isinstance(source, Path)
            # Uploads are already spooled by multipart parsing. Only local artifacts need a copy.
            with SpooledTemporaryFile(max_size=MIB) if is_artifact else nullcontext(source.file) as file:
                size = copy_artifact(source, file) if is_artifact else source.size
                if not size:
                    raise HTTPException(status_code=400, detail=f"{label}是空檔")
                if size > MAX_REFERENCE_BYTES:
                    raise HTTPException(status_code=413, detail=f"{label}超過 35 MiB 上限")
                with Image.open(file, formats=IMAGE_FORMATS) as image:
                    if max(image.size) > MAX_IMAGE_SIDE or image.width * image.height > MAX_IMAGE_PIXELS:
                        raise HTTPException(status_code=413, detail=(
                            f"{label}超過邊長 {MAX_IMAGE_SIDE} 或 {MAX_IMAGE_PIXELS:,} 像素上限"
                        ))
                    mode = "RGBA" if image.mode in ("RGBA", "LA") or "transparency" in image.info else "RGB"
                    with image.convert(mode) as png:
                        png.save(folder / f"ref_{index}.png", format="PNG")
        except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise HTTPException(status_code=413, detail=f"{label}超過像素上限") from exc
        except (OSError, SyntaxError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"{label}無法讀取、已截斷或格式不支援") from exc


async def prepare_reference_images(sources: list[UploadFile | Path], folder: Path) -> None:
    task = asyncio.create_task(run_in_threadpool(write_reference_images, sources, folder))
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        # Join the encoder before the request closes its uploads or deletes its staging directory.
        with CancelScope(shield=True):
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not task.cancelled():
                task.exception()
        raise
