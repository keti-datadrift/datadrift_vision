transformers==4.39.1

pip install fastapi celery redis "uvicorn[standard]"
pip install python-multipart
https://dgkim5360.tistory.com/entry/install-redis-for-linux-or-windows
https://github.com/microsoftarchive/redis/releases

redis-cli

#데모 테스트 실행방법
(제외해도 되도록 수정했음) 1.redis-server --port 6380 --slaveof 127.0.0.1 6379
(redis-server redis.windows.conf)
2.python publisher.py
3.python client.py


------------
pip install pyqt5
pip install pyqt5-tools
----------------- 제외 -----------------
celery -A celery_app worker --pool=solo -l info
celery -A celery_app worker --loglevel=DEBUG
