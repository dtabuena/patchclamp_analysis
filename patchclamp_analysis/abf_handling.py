def get_sub_files(rootdir):
    'Recursively search subfolders and return a list of all files'
    file_list =[]
    for rootdir, dirs, files in os.walk(rootdir):
            file_list.extend([os.path.join(rootdir,f) for f in files])
    return file_list
