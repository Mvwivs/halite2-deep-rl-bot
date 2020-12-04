#!/bin/bash

filename="bot_list.txt"

while IFS= read -r line
do
    read -r interpreter file <<< "$line"

    path=$(dirname "${file}")
    name=$(basename "${path}")

    # echo $name
    # echo $line
    python ./manager.py -A "$name" -p "$line"
    
done < "$filename"
