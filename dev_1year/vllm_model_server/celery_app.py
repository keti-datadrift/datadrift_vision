from celery import Celery,shared_task
from celery.result import AsyncResult
from fastapi import FastAPI
# from keywords_extractor import KeywordsExtractor
from time import time
import traceback
import GPUtil
GPUtil.showUtilization()
import sys
import os
subpath=os.path.dirname(os.path.abspath(__file__))
print(subpath)
sys.path.append(subpath)

def make_celery(app_name=__name__):
    backend = "redis://127.0.0.1:6379"
    broker  = "redis://127.0.0.1:6379"
    
    celery = Celery(__name__, backend=backend, broker=broker)

    return celery

celery_app = make_celery('high_model')
model_path = 'MLP-KTLim/llama-3-Korean-Bllossom-8B-Q4_K_M.gguf',
@shared_task(name='high_model_task')
def high_model_task(query_text ):
    s0 = time()
    print("******high_model_task:",query_text )

    result = None
    s1 = time()
    print(f'\n****************** elapsed time: {s1-s0} ******************\n')
    return result
def main():
    celery_app.worker_main(argv=['worker', '--loglevel=info'])

if __name__ == '__main__':
    main()
