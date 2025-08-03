import sys
import traceback

def get_error_message(error):
    exc_type, exc_obj, exc_tb = sys.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    return f"Error in [{file_name}], line [{line_number}]: {error}"

class CustomException(Exception):
    def __init__(self, error):
        message = get_error_message(error)
        super().__init__(message)