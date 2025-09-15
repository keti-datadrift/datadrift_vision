from datetime import datetime, timedelta
import time

import logging
import pandas as pd
import psycopg2
import glob
# from sentence_transformers import SentenceTransformer
import pgvector
import traceback
import yaml
import os

base_abspath = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))
with open(base_abspath+'/config.yaml') as f:
    config = yaml.full_load(f)
    AIMEMO_TABLE_NAME = config['aimemo_table_name']

    BATCH_SIZE_GPU = config['BATCH_SIZE_GPU']
    BATCH_SIZE_SQL = config['BATCH_SIZE_SQL']
    # SLEEP = config['SLEEP']
    # MAX_RUN_TIME = config['MAX_RUN_TIME'] 
    # TEST = config['TEST']  
    # START_TIME = config['START_TIME']
    # END_TIME  = config['END_TIME']

    ef_search = config['ef_search']
    print('ef_search =',ef_search)

def configure_logging(process_id):
    log_file = f"D:/text_similarity/process_{process_id}.log"
    logging.basicConfig(
        filename=log_file,
        # filemode='a+',
        filemode='w',
        format='%(asctime)s [Process %(process)d] %(levelname)s: %(message)s',
        level=logging.INFO,
        force=True 
    )
    logging.info(f"Logging started for process {process_id}")
    pass

def local_connect_db():
    with open(base_abspath+'/config.yaml') as f:
        config = yaml.full_load(f)

    conn = None
    cur = None
    try:        
        host = config['local_host']
        port = config['local_port']
        dbname = config['local_dbname']
        user = config['local_user']
        password = config['local_password']     
        # PostgreSQL 서버에 연결
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password
        )
        conn.autocommit = True
        cur = conn.cursor()
    except Exception as e:
        log_msg = f'Exception: {traceback.format_exc()}'
        print(log_msg)
        logging.info(log_msg)

    return conn, cur

def ams_connect_db():
    try:
        ams_conn = psycopg2.connect(
            host=config['ams_host'],
            port=config['ams_port'],
            dbname=config['ams_dbname'],
            user=config['ams_user'],
            password=config['ams_password']
        )
        ams_cur = ams_conn.cursor()
        return ams_conn,ams_cur
    except Exception as e:
        logging.info("Error occurred:", e)
        return None,None

    
def insert_db(conn,cur,sentence, embedding,TABLE_NAME):
    query = f"""
    INSERT INTO {TABLE_NAME} (sentence, embedding)
    VALUES (%s, %s)
    """
    cur.execute(query, (sentence, embedding))

def db_connect_close(conn,cur,process_name):
    log_msg = f'Info: ({process_name}), db_connect_close()'
    print(log_msg)
    logging.info(log_msg)
    if cur:
        cur.close()
        cur = None
    if conn:
        conn.close()
        conn = None

def ams_close_and_reconnect_db(conn,cur,process_name):
    log_msg = f'Exception: ({process_name}), {traceback.format_exc()}'
    logging.info(log_msg)
    if cur:
        cur.close()
        cur = None
    if conn:
        conn.close()
        conn = None
    while True:
        log_msg = f'Exception: {process_name}, cur.close(), conn.close(), conn = local_connect_db(), re-connect'
        print(log_msg)
        logging.info(log_msg)
        # conn,cur = local_connect_db()
        conn,cur = ams_connect_db()
        # cur = conn.cursor()
        if conn and cur:
            break
        else:
            log_msg = f'Exception: {process_name}, re-connect try again, {SLEEP} seconds......'
            print(log_msg)
            logging.info(log_msg)
            time.sleep(SLEEP) 
    
    return conn,cur

def similarity_search(conn, cur, query_text, embedding,TABLE_NAME):
    # SET enable_seqscan = OFF;
    # EXPLAIN ANALYSE
    command_sql = f"""
                    SET enable_seqscan = OFF;


                    select sentence, 1-(embedding <=> %s::vector) as score\
                    from {TABLE_NAME}
                    order by embedding <=> %s::vector
                    limit {ef_search}"""
    print(command_sql)
    bt = datetime.now() 
    cur.execute(command_sql,(embedding,embedding))
    db_content = cur.fetchmany(ef_search)

    print('similarity search N=5, time: ',datetime.now()-bt)

    return db_content

def similarity_search_filter(conn, cur, query_text, embedding,TABLE_NAME,start_time,end_time):
    start_date = '2024-12-01 00:00:00.000000'
    end_date = '2025-12-31 00:00:00.000000'
    command_sql = f"""select sentence,address,LATITUDE, LONGITUDE, 1-(embedding <=> %s::vector) as score\
                    from {TABLE_NAME}
                    WHERE created_at >= %s and created_at <= %s
                    order by embedding <=> %s::vector                    
                    limit {ef_search}"""
    print(command_sql)
    bt = datetime.now() 
    cur.execute(command_sql,(embedding,start_time,end_time,embedding))
    db_content = cur.fetchmany(ef_search)

    print('similarity search N=5, time: ',datetime.now()-bt)

    return db_content

def similarity_search_filter2(conn, cur, query_text, embedding,TABLE_NAME,latitude,longitude):
    start_date = '2024-12-01 00:00:00.000000'
    end_date = '2025-12-31 00:00:00.000000'
    # Haversine formula in SQL to calculate distance within 1000 meters
    # command_sql = f"""
    #     SELECT 
    #         sentence, address,LATITUDE,LONGITUDE,
    #         1 - (embedding <=> %s::vector) AS score,
    #         (6371000 * acos( -- Earth's radius in meters
    #             cos(radians(%s)) * cos(radians(LATITUDE)) * 
    #             cos(radians(LONGITUDE) - radians(%s)) +
    #             sin(radians(%s)) * sin(radians(LATITUDE))
    #         )) AS distance
    #     FROM {TABLE_NAME}
    #     WHERE 
    #         created_at >= %s AND created_at <= %s
    #         AND (6371000 * acos(
    #             cos(radians(%s)) * cos(radians(LATITUDE)) * 
    #             cos(radians(LONGITUDE) - radians(%s)) +
    #             sin(radians(%s)) * sin(radians(LATITUDE))
    #         )) <= 1000000 -- Filter for rows within 10000 meters
    #     ORDER BY embedding <=> %s::vector
    #     LIMIT 100;
    # """
    # print(command_sql)
    # bt = datetime.now() 
    # cur.execute(
    #     command_sql, 
    #     (embedding, latitude, longitude, latitude, start_date, end_date, latitude, longitude, latitude, embedding)
    # )    
    # 경위도 차이 변환:
    # 위도 1도 차이는 약 111.19km (111,190m).
    # 경도 1도 차이는 위도에 따라 달라지며, 적도에서 약 111.32km (111,320m).

    command_sql = f"""
        SELECT 
            sentence,  
            abs(LATITUDE - %s) AS LATITUDE_DIFF,  
            abs(LONGITUDE - %s) AS LONGITUDE_DIFF, 
            address,
            1 - (embedding <=> %s::vector) AS score
        FROM {TABLE_NAME}
        WHERE 
            created_at >= %s AND created_at <= %s
            AND LATITUDE BETWEEN %s - 0.1 AND %s + 0.1
            AND LONGITUDE BETWEEN %s - 0.1 AND %s + 0.1
        ORDER BY embedding <=> %s::vector
        LIMIT {ef_search};
    """
    print("Generated SQL Query:")
    print(command_sql)

    bt = datetime.now()

    # Execute the query
    cur.execute(
        command_sql, 
        (
            latitude,        # LATITUDE difference calculation
            longitude,       # LONGITUDE difference calculation
            embedding,       # For embedding vector
            start_date,      # For created_at >= %s
            end_date,        # For created_at <= %s
            latitude,        # For LATITUDE BETWEEN %s - 0.01
            latitude,        # For LATITUDE BETWEEN %s + 0.01
            longitude,       # For LONGITUDE BETWEEN %s - 0.01
            longitude,       # For LONGITUDE BETWEEN %s + 0.01
            embedding        # For ORDER BY embedding <=> %s::vector
        )
    )


    
    db_content = cur.fetchmany(ef_search)

    print('similarity search N=5, time: ',datetime.now()-bt)

    return db_content

def get_next_start_time(start_time: str, process_name):
    # date_format = '%Y-%m-%d %H:%M:%S.%f'
    # start_time = datetime.strptime(start_time, date_format)    
    if 'day'==TEST:
        next_time = start_time + timedelta(days=1)
    elif 'hour'==TEST:
        next_time = start_time + timedelta(hours=1)
    else:
        log_msg = f'{process_name}, TEST value is wrong!!!'
        logging.info(log_msg)
        raise Exception(log_msg)
    return next_time