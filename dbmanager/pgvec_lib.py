from datetime import datetime, timedelta
import time

import logging
import pandas as pd
import psycopg2
import glob
# from sentence_transformers import SentenceTransformer
# import pgvector
import traceback
import yaml
import os

base_abspath = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))
with open(base_abspath+'/config.yaml', encoding="utf-8") as f:
    config = yaml.full_load(f)
    AIMEMO_TABLE_NAME = config['datadrift_table']

def configure_logging(process_id):
    log_file = f"./process_{process_id}.log"
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
    with open(base_abspath+'/config.yaml', encoding="utf-8-sig") as f:
        config = yaml.full_load(f)

    conn = None
    cur = None
    try:        
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

