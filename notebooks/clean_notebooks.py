# Clean notebooks from output and system-dependent metadata for pushing to git
import subprocess
import re


proc = subprocess.Popen(['git', 'ls-files', '-m', '*.ipynb'], stdout=subprocess.PIPE, universal_newlines=True)
filelist = proc.communicate()[0].split('\n')[:-1]

if len(filelist) == 0:
    print("Nothing to clean.")
else:
    for filename in filelist:
        print("Cleaning: {}... ".format(filename), end='')

        with open(filename, "r+") as f:
            text = f.read()

            n_outputs = text.count('"outputs": [\n')
            if n_outputs > 0:
                answer = input("Notebook contains cell execution output. Clean? (y/n) ")
                if answer == 'y':
                    text = re.sub('"execution_count": [0-9]+', '"execution_count": null', text)
                    text = re.sub('"outputs": \[[\n\r].+?"source": \[', '"outputs": [],\n   "source": [', text, flags=re.DOTALL)
                else:
                    print("- Skipped {}".format(filename))
                    continue

            text = re.sub('"display_name": ".*"', '"display_name": "Python 3"', text)
            text = re.sub('"language": ".*"', '"language": "python"', text)
            text = re.sub('"name": ".*"', '"name": "python3"', text)
            text = re.sub('"version": ".*"', '"version": ""', text)

            pos = f.seek(0)
            f.write(text)
            f.truncate()
            print("Done")
