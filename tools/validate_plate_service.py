"""Opt-in local HTTP evidence; writes only to the caller's explicit report path."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import time

import httpx
import numpy as np
from PIL import Image


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--url',default='http://127.0.0.1:1788')
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repetitions',type=int,default=200)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Report exists; choose a new output path')
    manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
    images=[]
    for row in manifest:
        data=Path(row['image']).read_bytes()
        assert hashlib.sha256(data).hexdigest()==row['image_sha256']
        images.append(data)
    headers={}
    token=os.environ.get('PLATEAI_API_TOKEN')
    if token:headers['Authorization']='Bearer '+token
    def stable(result):return {k:v for k,v in result.items() if k not in ('request_id','timings_ms')}
    with httpx.Client(base_url=args.url,headers=headers,timeout=15.,trust_env=False) as client:
        ready=client.get('/api/ready');ready.raise_for_status();before=ready.json()
        for _ in range(3):
            response=client.post('/api/predict',files={'file':('scene.jpg',images[0])});response.raise_for_status()
        encoded=base64.b64encode(images[0]).decode()
        inputs=[client.post('/api/predict',files={'file':('scene.jpg',images[0])}),
                client.post('/api/predict',json={'image_base64':encoded})]
        with Image.open(io.BytesIO(images[0])) as im:
            mime={'PNG':'image/png','JPEG':'image/jpeg','WEBP':'image/webp'}[im.format]
        inputs.append(client.post('/api/predict',json={'image_base64':f'data:{mime};base64,'+encoded}))
        for response in inputs:response.raise_for_status()
        assert all(stable(r.json())==stable(inputs[0].json()) for r in inputs)
        rows=[]
        for row,data in zip(manifest,images):
            started=time.perf_counter()
            response=client.post('/api/predict',files={'file':('scene.jpg',data)})
            elapsed=(time.perf_counter()-started)*1000
            response.raise_for_status()
            rows.append({'id':row['id'],'client_ms':elapsed,'result':response.json()})
        # Functional no-plate smoke, not a genuine negative validation set.
        blank=io.BytesIO();Image.new('RGB',(640,480),'white').save(blank,format='PNG')
        negative=client.post('/api/predict',files={'file':('blank.png',blank.getvalue())})
        negative.raise_for_status();assert negative.json()['status']=='no_plate'
        bad=client.post('/api/predict',json={'image_base64':'!!!'})
        assert bad.status_code==400
        multi_index=next(i for i,r in enumerate(manifest) if r['cohort'].startswith('multi') or r['id'].startswith('multi'))
        trials=[]
        for index in range(args.repetitions):
            image_index=0 if index%2==0 else multi_index
            started=time.perf_counter()
            response=client.post('/api/predict',files={'file':('scene.jpg',images[image_index])})
            elapsed=(time.perf_counter()-started)*1000
            response.raise_for_status()
            value=response.json()
            assert stable(value)==stable(rows[image_index]['result'])
            trials.append({'case':manifest[image_index]['id'],'client_ms':elapsed,'timings_ms':value['timings_ms']})
            if (index+1)%25==0:print(f'HTTP {index+1}/{args.repetitions}',flush=True)
        after=client.get('/api/ready').json()
        assert after['ready'] and after['initialization_count']==before['initialization_count']==1
        timings={}
        for case in dict.fromkeys(t['case'] for t in trials):
            values=[t['client_ms'] for t in trials if t['case']==case]
            timings[case]={'n':len(values),'p50_ms':statistics.median(values),'p95_ms':float(np.percentile(values,95))}
        report={'scope':'local HTTP regression on previously seen images, not deployment qualification',
                'three_inputs_identical':True,'ready_before':before,'ready_after':after,
                'scenes':rows,'trials':trials,'http_latency':timings,'blank_functional_no_plate':True}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'http_latency':timings,'initialization_count':after['initialization_count'],'scenes':len(rows)},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
