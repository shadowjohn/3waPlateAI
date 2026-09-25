"""Explicit single-worker local entrypoint; does not install or download models."""
from __future__ import annotations

import argparse
from dataclasses import replace
import logging
import os
from pathlib import Path

from .config import ServiceConfig, validate_binding


def main():
    parser=argparse.ArgumentParser(description='Local YOLO + PP-OCRv3 full-scene API')
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=1788)
    parser.add_argument('--device',choices=('auto','cpu'))
    args=parser.parse_args()
    token=os.environ.get('PLATEAI_API_TOKEN') or None
    validate_binding(args.host,token)
    config=ServiceConfig.from_file(args.config)
    if args.device:config=replace(config,device=args.device)
    os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK','True')
    os.environ.setdefault('OMP_NUM_THREADS',str(config.threads))
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(name)s %(message)s')
    from .app import create_app
    import uvicorn
    uvicorn.run(create_app(config,token=token),host=args.host,port=args.port,
                workers=1,reload=False,access_log=False,limit_concurrency=32,
                timeout_keep_alive=5,timeout_graceful_shutdown=15)


if __name__=='__main__':main()
