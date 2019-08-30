path=`pwd`/$1
files=`ls ${path}`

for f in $files 
     do
        echo process ${f} in folder ${path}
        python /home/LAB/zhangzy/ProgrammingAlpha/test/db_test/gen_knownet_corpus/local_mode.py --folder $path --file ${f} &
    done
echo waiting for all tasks to finish
wait
echo *************
echo *************
echo finished processing all files