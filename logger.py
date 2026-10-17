import csv
from datetime import datetime


def log_event(event):
    with open("fatigue_log.csv","a",newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now(), event])
