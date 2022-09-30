import os
import shutil


def delete_tb_logs():
    for folder in os.listdir('logs'):
        shutil.rmtree(os.path.join('logs', folder))


if __name__ == '__main__':
    delete_tb_logs()
