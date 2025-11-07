#!/usr/bin/env python3
"""
APScheduler를 이용해 drift_checker.py를 주기적으로 실행
"""

import time
import signal
import sys
import logging
import traceback
import debugpy
from apscheduler.schedulers.background import BackgroundScheduler
from drift_checker import main as drift_check_main, setup_logging, load_config
from dbmanager.create_table import do_create_table
from dbmanager.create_partition import do_create_partition

def scheduled_job():
    """드리프트 감지 작업 실행"""
    logging.info("Running scheduled drift check...")
    drift_check_main()


def main():
    """APScheduler 초기화 및 주기 설정"""
    try:
        config = load_config()
        setup_logging(
            log_file=config.get("logging", {}).get("log_file", "logs/drift_scheduler.log"),
            log_level=config.get("logging", {}).get("log_level", "INFO")
        )
        # at the beginning
        do_create_table()
        do_create_partition()
        drift_check_main()
        # Get partition creation schedule from config
        drift_config = config.get('drift_detection', {})
        partition_hour = drift_config.get('partition_create_hour', 23)
        partition_minute = drift_config.get('partition_create_minute', 0)

        # scheduling
        scheduler = BackgroundScheduler()
        scheduler.add_job(
            do_create_partition,
            trigger='cron',
            hour=partition_hour,
            minute=partition_minute,
            id='daily_partition_create',
            replace_existing=True
        )

        logging.info(f"📅 Partition creation scheduled daily at {partition_hour:02d}:{partition_minute:02d}")      
        scheduler.add_job(scheduled_job, "interval", hours=1, id="drift_check")
        # scheduler.add_job(scheduled_job, "interval", seconds=10, id="drift_check")

        scheduler.start()
        logging.info("🚀 APScheduler drift detection service started (interval=1h)")

        # 안전한 종료 처리
        def shutdown(signum, frame):
            import traceback
            # logging.info("🛑 Shutting down scheduler...")
            logging.info(f"Signal {signum} received. Traceback:")
            traceback.print_stack(frame)
            scheduler.shutdown(wait=False)
            sys.exit(0)

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

        while True:
            time.sleep(60)

    except Exception as e:
        logging.error(f"Unexpected error in scheduler: {e}", exc_info=True)
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
