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
subpath=os.path.dirname(os.path.abspath(__file__))#+'\ControlLog'
print(subpath)
sys.path.append(subpath)

def make_celery(app_name=__name__):
    backend = "redis://127.0.0.1:6379"
    broker  = "redis://127.0.0.1:6379"
    
    celery = Celery(__name__, backend=backend, broker=broker)

    return celery
celery_app = make_celery('vllm')
model_path = 'llava model here', #to chage model to llava
@shared_task(name='generate_text_task')
def generate_text_task(query_text ):
    s0 = time()
    print("******generate_text_task:",query_text )


    result = None
    s1 = time()
    print(f'\n****************** elapsed time: {s1-s0} ******************\n')
    return result
def main():
    celery_app.worker_main(argv=['worker', '--loglevel=info'])

if __name__ == '__main__':
    main()
