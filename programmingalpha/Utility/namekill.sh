progress=$1
echo "progress are $progress"
ps -ef | grep zhangzy | grep $progress

pids=`ps -ef | grep zhangzy |grep $progress | grep -v grep | grep -v namekill.sh | awk '{print $2}'`
for pid in $pids;do
echo "kill -9 ${pid}"
kill -9 $pid

done

echo "left for $progress"
ps -ef | grep zhangzy |grep $progress

