import logging  ## Ye logger,info , batata ha ki ye point pe kya hoga aur kuya nahi
import os ## Ye toh normally path and new files ke liye hota ha 
from datetime import datetime ## Ye kyu kia , taaki hum log file mein bata sake ki konse m=time mein humne ye part complete kia tha

Log_File=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

## Bhai yaha strftime ye ek timestamp ha jo batata ha ki time konse way mein rahega

log_path=os.path.join(os.getcwd(),'logs',Log_File) ## Ye bata ha ki kaha meri log file banegi

os.makedirs(log_path,exist_ok=True) ## Kyu agar bani ha toh aacha ha , nahi toh banao 

Log_file_path=os.path.join(log_path,Log_File) 

logging.basicConfig(
    filename=Log_file_path,
    format='[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)


# if __name__ == "__main__":
#     logging.info("Logging Has Started")






