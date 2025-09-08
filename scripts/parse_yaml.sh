#!/bin/bash

# Function to parse YAML and export variables
# Adapted from https://stackoverflow.com/a/21189044
function parse_yaml {
   local prefix=$2
   local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
   sed -ne "s|^\($s\):|\1|" \
        -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  "$1" |
   awk -F$fs '{
      indent = length($1)/2;
      vname[indent] = $2;
      for (i in vname) {if (i > indent) {delete vname[i]}}
      if (length($3) > 0) {
         vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
         # Check if the value is a list (starts with "(" and ends with ")")
         if ($3 ~ /^\(.*\)$/) {
            array_value = substr($3, 2, length($3) - 2); # Remove parentheses
            gsub(/\"/, "", array_value); # Remove any quotes around values
            printf("%s%s%s=(%s)\n", "'$prefix'", vn, $2, array_value);
         } else {
            printf("%s%s%s=\"%s\"\n", "'$prefix'", vn, $2, $3);
         }
      }
   }'
}