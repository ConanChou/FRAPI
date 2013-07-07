#!/usr/bin/env python

import os

def batch_upload(path):
    for dir_name, dir_names, file_names in os.walk(path):
        for subdir_name in dir_names:
            file_path = os.path.join(dir_name, subdir_name)
            for file_name in os.listdir(file_path):
                full_path = os.path.join(file_path, file_name)
                os.system("""curl -F file="@%s" -F name='%s' -X POST http://localhost:5000/face_recognizer/api/v1/faces/training""" % (full_path, subdir_name))

def main():
    batch_upload('../training_data')

if __name__ == '__main__':
    main()
