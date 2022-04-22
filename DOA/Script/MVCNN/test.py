
import logging
import sys

values = sys.argv
# print(values)

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logging.info('Admin logged in')