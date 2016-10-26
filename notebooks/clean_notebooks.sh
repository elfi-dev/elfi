#!/bin/bash
#
# Clean notebooks from system-dependent metadata for pushing to git
#

FILEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && readlink -f -- . )"
cd $FILEDIR

for i in $(find -maxdepth 1 -name "*.ipynb"); do
	n_outputs=`cat ${i} | grep '"outputs": \[$' | wc -l`
	if [[ ! "${n_outputs}" == "0" ]]; then
		echo "Notebook '${i}' contains cell execution output. Please remove before committing."
	fi
	sed -i 's/"display_name": ".*"/"display_name": "Python 3"/' ${i}
	sed -i 's/"language": ".*"/"language": "python"/' ${i}
	sed -i 's/"name": ".*"/"name": "python3"/' ${i}
	sed -i 's/"version": ".*"/"version": ""/g' ${i}
done
echo "done"

