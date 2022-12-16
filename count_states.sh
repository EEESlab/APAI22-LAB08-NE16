#!/bin/sh
STATES="State"

if [ $# -eq 0 ];
then
    grep "$STATES" | sed 's/.*State \(.*$\)/\1/' | sort | uniq -c
elif [ $# -eq 1 ];
then
    if [ -f $1 ]; then
        grep "$STATES" $1 | sed 's/.*State \(.*$\)/\1/' | sort | uniq -c
    else
        echo "File doesn't exist."
        return 1
    fi
else
    echo "Unsupported number of arguments. Expected 0 or 1 arguments."
    return 1
fi

