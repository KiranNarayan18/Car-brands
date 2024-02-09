import logging 
import os
from datetime import datetime
import sys


def error_message_detail(error, error_detail: sys):
    _, _, exe_tb = error_detail.exc_info()
    file_name = exe_tb.tb_frame.f_code.co_filename

    error_message = "Error Occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exe_tb.tb_lineno, str(error))

    return error_message


class CustomException(Exception):

    def __init__(self, error_message, error_detail: sys) -> None:
        super().__init__(error_message)
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message
    


LOG_DIR=f"{datetime.now().strftime('%m_%d_%Y')}"

logs_path=os.path.join(os.getcwd(),"logs")
os.makedirs(logs_path,exist_ok=True)

LOG_FILE_PATH=os.path.join(logs_path, f'{LOG_DIR}.txt')

logging.basicConfig(

    format="[ %(asctime)s ] - %(levelname)s - %(module)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("customLogger")


# class logger:   

#     def info(e, sys):
#         error_message = CustomException(e, sys)
#         customLogger.info(error_message)

#     def error(e, sys):
#         error_message = CustomException(e, sys)
#         customLogger.error(error_message)



if __name__ == "__main__":
    
    try:
        a = 1/0
    except Exception as e:
        logger.error(e, sys)
        