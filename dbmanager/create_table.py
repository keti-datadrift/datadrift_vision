from datetime import datetime
import pandas as pd
import psycopg2
import glob
import logging
import traceback
import yaml
import os
import sys

base_abspath = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))
sys.path.append(base_abspath)      

# from dbmanager.pgvec_lib import *


def connect_db():
    with open(base_abspath+'/config.yaml',encoding='utf-8') as f:
        config = yaml.full_load(f)

    conn = None
    cur = None
    try:     
        # TABLE_NAME = config['table_name']   
        host = config['postgres']['host']
        port = config['postgres']['port']
        dbname = config['postgres']['dbname']
        user = config['postgres']['user']
        password = config['postgres']['password']     
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
def create_db(conn,cur,TABLE_NAME):
    print(f'+++++++ create_db() TABLE_NAME={TABLE_NAME} started +++++++')

    try:
        query = f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id           BIGSERIAL,
            frame_id     TEXT,                           -- response['frame_id']
            camera_name  VARCHAR(32) NOT NULL,           -- response['camera_id']
            class_name   VARCHAR(30),                    -- response['class']
            event_time   TIMESTAMP WITHOUT TIME ZONE NOT NULL,
            confidence   REAL,                           -- response['confidence']
            vlm_valid    VARCHAR(20),                    -- response['vlm_valid'] (검증 결과)

            created_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (id, created_at)
        ) PARTITION BY RANGE (created_at);
        """
        # created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        # request_id BIGINT NOT NULL,
        # confidence_score FLOAT,
        # image_path TEXT,
        cur.execute(query)
        sql_command = """SET max_parallel_workers = 8;
                        SET parallel_setup_cost = 0;
                        SET parallel_tuple_cost = 0;
                        SET maintenance_work_mem = '1900MB';
                        SET max_parallel_workers_per_gather = 4;
                    """

        cur.execute(sql_command)
        conn.commit()

        print(f'{TABLE_NAME} is created!!!')
    except Exception as e: 
        log_msg = f'Exception: {TABLE_NAME} create is failed, {traceback.format_exc()}'     
        print(log_msg)

def main_create_vectordb():
    conn,cur = connect_db()
    with open(base_abspath+'/config.yaml',encoding='utf-8') as f:
        config = yaml.full_load(f)
        DD_TABLE_NAME = config['datadrift_table']

        create_db(conn,cur, DD_TABLE_NAME)

if __name__ == "__main__":
    main_create_vectordb()
    pass

