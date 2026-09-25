"""One native owner thread; cancellation never releases a still-running job."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import time

from .errors import ServiceError
from .images import decode_image
from .pipeline import PlatePipeline

LOG = logging.getLogger('plateai_service')


class InferenceRuntime:
    def __init__(self, config, factory=PlatePipeline):
        self.config, self.factory = config, factory
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='plate-native')
        self.pipeline = None
        self.ready = False
        self.busy = False
        self.failure = None
        self.initializations = 0
        self.completed_requests = 0
        self.startup_ms = None

    def start(self):
        loop = asyncio.get_running_loop()
        started = time.perf_counter()
        def load():
            pipeline = self.factory(self.config)
            pipeline.initialize()
            return pipeline
        def complete(future):
            self.initializations += 1
            self.startup_ms = (time.perf_counter()-started)*1000
            try:
                self.pipeline = future.result()
                self.ready = True
            except Exception as exc:
                self.failure = 'initialization_failed:' + type(exc).__name__
                LOG.error('model initialization failed type=%s', type(exc).__name__)
        future = self.executor.submit(load)
        future.add_done_callback(lambda done: loop.call_soon_threadsafe(complete, done) if not loop.is_closed() else None)

    def reserve(self):
        if not self.ready:
            raise ServiceError('model_not_ready', 503)
        if self.busy:
            raise ServiceError('busy', 429)
        self.busy = True

    def metadata(self):
        return dict(ready=self.ready, busy=self.busy, failure=self.failure,
                    startup_ms=self.startup_ms, initialization_count=self.initializations,
                    completed_requests=self.completed_requests,
                    **(self.pipeline.metadata() if self.pipeline else {'model_id':self.config.model_id}))

    def submit_reserved(self, data, mime):
        loop = asyncio.get_running_loop()
        def work():
            started = time.perf_counter()
            bgr = decode_image(data, mime)
            decoded = time.perf_counter()
            result = self.pipeline.predict(bgr)
            result['timings_ms']['decode'] = (decoded-started)*1000
            return result
        def complete(future):
            try:
                future.result()
            except ServiceError:
                pass
            except Exception as exc:
                self.ready = False
                self.failure = 'inference_failed:' + type(exc).__name__
                LOG.error('native inference failed type=%s', type(exc).__name__)
            self.completed_requests += 1
            self.busy = False
        future = self.executor.submit(work)
        # Register before wrap_future so readiness changes precede response completion.
        future.add_done_callback(lambda done: loop.call_soon_threadsafe(complete, done) if not loop.is_closed() else None)
        return asyncio.wrap_future(future)

    def close(self):
        self.ready = False
        self.executor.shutdown(wait=False, cancel_futures=False)
