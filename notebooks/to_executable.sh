#!/bin/bash

source=${1}
filename="${source%.*}"
target="${filename}.py"

if [ ! -f ${source} ]; then
	echo "${source} does not exist, terminating"
	exit 1
fi

if [ -f ${target} ]; then
	echo "${target} already exists, removing"
	rm ${target}
fi

echo "converting ${source} to ${target}"
jupyter nbconvert --to script ${source}

echo "post-processing.."
# remove potential lines that try to invoke ipython
sed -i '/get_ipython()/d' ${target}

# change matplotlib backend
sed -i '/^import matplotlib$/a matplotlib.use("Agg")' ${target}

echo "done"

