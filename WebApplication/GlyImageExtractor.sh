#!/bin/bash

SERVICE=GlyImageExtractor
FLASK_APP=./${SERVICE}.py

start() {
	PIPELINE=$1  # Capture the argument passed to the function
    echo -n "Starting ${SERVICE} with args: '${PIPELINE}': "
	export FLASK_APP
	export FLASK_ENV=development
        # Construct the command, including --p only if PIPELINE is provided
		echo "PIPELINE argument: $PIPELINE"
		if [ -z "$PIPELINE" ]; then
			nohup .venv2/bin/python ${FLASK_APP} > ${SERVICE}.log 2>&1 &
		else
			nohup .venv2/bin/python ${FLASK_APP} --p "${PIPELINE}" > ${SERVICE}.log 2>&1 &
		fi
	RETVAL=$?
	echo "done."
	return $RETVAL
}	

stop() {
	echo -n "Shutting down ${SERVICE}: "
	PIDS=`ps -ef | fgrep -w ${FLASK_APP} | fgrep -v grep | awk '{print $2}'`
	if [ "$PIDS" != "" ]; then
	  kill -9 `ps -ef | fgrep -w ${FLASK_APP} | fgrep -v grep | awk '{print $2}'`
	  RETVAL=$?
	else
	  RETVAL=0
	fi
        echo "done."
	return $RETVAL
}

status() {
	ps -ef | fgrep -w ${FLASK_APP} | fgrep -v grep
	RETVAL=$?
	return $RETVAL
}

case "$1" in
    start)
	start "${@:2}"
	RETVAL=$?
	;;
    stop)
	stop
	RETVAL=$?
	;;
    restart)
	stop || true
	start "${@:2}"
	RETVAL=$?
	;;
    status)
	status
	RETVAL=0
	;;
    *)
	echo "Usage: $SERVICE {start|stop|restart|status}"
	exit 1
	;;
esac
exit $RETVAL
