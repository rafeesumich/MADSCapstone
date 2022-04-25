import os

def is_image(filename):

    data = open(filename,'rb').read(10)

    # check if file is JPG or JPEG
    if data[:3] == b'\xff\xd8\xff':
        return True
    return False


folder = 'all_images/'  # Enter directory name where images are held

i = 0  # Counter to keep track of files deleted
num_files_checked = 0

# Iterate through files and delete those that are not jpegs
for filename in os.listdir(folder):
    full_path = folder + filename
    num_files_checked += 1
    if num_files_checked%1000 == 0:
        print('files checked: ' + str(num_files_checked))  # Shows progress
    try:
        full_path = folder + filename
        if open(full_path,'r').read(1) == '':
            os.remove(full_path)
            i += 1
    except:
        if is_image(full_path) == False:
            i += 1
            os.remove(full_path)

print(str(i) + ' files deleted from ' + folder)
