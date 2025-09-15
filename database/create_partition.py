import sys
import psycopg2
from pytz import timezone
from datetime import datetime, timedelta
import yaml
from pgvec_lib import local_connect_db

import os
base_abspath = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))
with open(base_abspath+'/config.yaml') as f:
    config = yaml.full_load(f)


def create_partition_if_not_exists(START_DATE,END_DATE,partition_name,TABLE_NAME):
    conn,cursor = local_connect_db()
    try:      

        sql_command=f"""CREATE TABLE IF NOT EXISTS {partition_name}
                    PARTITION OF {TABLE_NAME}
                    FOR VALUES FROM ('{START_DATE} 00:00:00') TO ('{END_DATE} 00:00:00');"""
    
        cursor.execute(sql_command)
        conn.commit()
        print(f"{partition_name}: Partition created successfully!")
    except Exception as e:
        print(f"Error: {e}\n{partition_name}: Partition creation failed!")
        

def do_create_partition():
    date_format = "%Y-%m-%d"    
    date_object = datetime.now(timezone('Asia/Seoul'))
    # END_DATE_ = START_DATE_ + timedelta(days=1)  
    REF_DATE = date_object.strftime(date_format)  
    # START_DATE = '2024-10-09'
    n = 0
    date_object = datetime.strptime(REF_DATE, date_format) + timedelta(days=0+n)
    START_DATE = date_object.strftime(date_format)

    date_object = datetime.strptime(REF_DATE, date_format) + timedelta(days=1+n)
    END_DATE = date_object.strftime(date_format)

    date_object = datetime.strptime(REF_DATE, date_format) + timedelta(days=2+n)
    END_DATE2 = date_object.strftime(date_format)
    # END_DATE = END_DATE_.strftime("%Y-%m-%d")
    print(f'************ {START_DATE} ~ {END_DATE} ***********')

    # CURR_DATE = START_DATE.replace('-','_')

    with open(base_abspath+'/config.yaml') as f:
        config = yaml.full_load(f)
        AIMEMO_TABLE_NAME = config['aimemo_table_name']

        tables = [AIMEMO_TABLE_NAME]
        for TABLE_NAME in tables: 
            partition_name = f'''{TABLE_NAME}_{START_DATE.replace('-','_')}'''
            create_partition_if_not_exists(START_DATE,END_DATE,partition_name,TABLE_NAME)  
            
            # 이튿날의 파티션 생성
            partition_name = f'''{TABLE_NAME}_{END_DATE.replace('-','_')}'''
            create_partition_if_not_exists(END_DATE,END_DATE2,partition_name,TABLE_NAME)              

if __name__ == "__main__":
    # Get the SQL command from the command-line argument
    # if len(sys.argv) != 2:
    #     print("Usage: python3 partition_creation.py '<SQL_COMMAND>'")
    #     sys.exit(1)

    # START_DATE = sys.argv[1]
    # START_DATE = datetime.now(timezone('Asia/Seoul'))# + timedelta(days=1)
    # START_DATE = START_DATE.strftime("%Y-%m-%d")
    do_create_partition()

