pip install fastapi celery redis "uvicorn[standard]"
pip install python-multipart
https://dgkim5360.tistory.com/entry/install-redis-for-linux-or-windows
https://github.com/microsoftarchive/redis/releases

#테스트 실행방법
1.redis-server --port 6380 --slaveof 127.0.0.1 6379
2.python publisher.py
3.python pyqt_aimemo_listcontrol_v11.py
4.python client

start "redis-server" "C:\Program Files\Redis\redis-server.exe" --port 6380 --slaveof 127.0.0.1 6379"
