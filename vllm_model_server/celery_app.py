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
subpath=os.path.dirname(os.path.abspath(__file__))+'\ControlLog'
print(subpath)
sys.path.append(subpath)
from ControlLog.extractKeywords import extractKeywords
from ControlLog.extractKeywords import *
def make_celery(app_name=__name__):
    backend = "redis://127.0.0.1:6379"
    broker  = "redis://127.0.0.1:6379"
    
    celery = Celery(__name__, backend=backend, broker=broker)

    return celery
celery_app = make_celery('nlp')
# keywords_ext = KeywordsExtractor()
model_path = './MLP-KTLim/llama-3-Korean-Bllossom-8B-Q4_K_M.gguf',
# eks = extractKeywords( model_path )
@shared_task(name='generate_text_task')
def generate_text_task(query_text ):
    s0 = time()
    print("******generate_text_task:",query_text )
    # keywords_ext = KeywordsExtractor()

    # response = keywords_ext.extract_keywords_from_response(seach_text)
    # result = eks.extract(1, event1_query[3], camera_data )
    # result = eks.extract(2, face2_query[1], camera_data)
    # result = eks.extract(3, plate3_query[7], camera_data)
    # result = eks.extract(4, object4_query[4], camera_data)
    # result = eks.extract(5, object4_query[1], camera_data )
    result = {
        "search_type": 1,
        "start_tm": "2024-09-23T00:00:00.000000",
        "end_tm": "2024-09-26T23:59:59.000000",
        "camera_id": [
            "방배역1",
            "효령빌딩",
            "포항제철소",
            "강남역",
            "서울대학교",
            "판문점"
        ],
        "result": {
            "event_type": [
                30,
                35,
                26
            ],
            "object_type": 0,
            "object_attr": 0
        }
    }
    print( result )    
    s1 = time()
    print(f'\n****************** elapsed time: {s1-s0} ******************\n')
    return result
def main():
    celery_app.worker_main(argv=['worker', '--loglevel=info'])

if __name__ == '__main__':
    main()
