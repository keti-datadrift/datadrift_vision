from datetime import datetime
import pandas as pd
import psycopg2
import glob
# from sentence_transformers import SentenceTransformer
import pgvector
import traceback
import yaml
import os
base_abspath = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))
       

from pgvec_lib import *
def delete_vectordb(conn,cur,TABLE_NAME):
    print(f'+++++++ create_db() TABLE_NAME={TABLE_NAME} started +++++++')
    try:
        query =  f"""DROP TABLE IF EXISTS {TABLE_NAME};          
                    """
        cur.execute(query)

        conn.commit()

        print(f'{TABLE_NAME} is removed!!!')
    except Exception as e: 
        log_msg = f'Exception: {TABLE_NAME} create is failed, {traceback.format_exc()}'
        print(log_msg)

def create_aimemo_vectordb(conn,cur,TABLE_NAME):
    print(f'+++++++ create_db() TABLE_NAME={TABLE_NAME} started +++++++')

    try:

        query =  f"""CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                    id BIGSERIAL,
                    request_id BIGINT NOT NULL,
                    event_type INTEGER,
                    event_time  timestamp without time zone NOT NULL,
                    camera_name VARCHAR(20) NOT NULL,                    
                    description TEXT,
                    rtsp_url TEXT,
                    image_path TEXT,
                    embedding VECTOR(768),
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (id, created_at)
                    ) PARTITION BY RANGE (created_at);             
                    """
        cur.execute(query)
        sql_command = """SET max_parallel_workers = 8;
                            SET parallel_setup_cost = 0;
                            SET parallel_tuple_cost = 0;
                            SET maintenance_work_mem = '2GB';
                            SET max_parallel_workers_per_gather = 4;
                            """
        cur.execute(query)
        conn.commit()

        print(f'{TABLE_NAME} is created!!!')
    except Exception as e: 
        log_msg = f'Exception: {TABLE_NAME} create is failed, {traceback.format_exc()}'     
        print(log_msg)

def delete_all_vectordb():
    conn,cur = local_connect_db()
    with open(base_abspath+'/config.yaml') as f:
        config = yaml.full_load(f)
        AIMEMO_TABLE_NAME = config['aimemo_table_name']

        tables = [AIMEMO_TABLE_NAME]

        for TABLE_NAME in tables:
            delete_vectordb(conn,cur, TABLE_NAME)

def main_create_vectordb():
    conn,cur = local_connect_db()
    with open(base_abspath+'/config.yaml') as f:
        config = yaml.full_load(f)
        AIMEMO_TABLE_NAME = config['aimemo_table_name']

        create_aimemo_vectordb(conn,cur, AIMEMO_TABLE_NAME)

if __name__ == "__main__":
    # conn,cur = local_connect_db()
    # delete_all_vectordb()
    main_create_vectordb()
    pass

