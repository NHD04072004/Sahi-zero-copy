#!/bin/bash

##################################################################################
# Checks, if the imports of the supported models are working as expected.
# Not all frameworks are available on all python versions.
#
# It checks it for all python versions
# Also, it checks, if the command line tool works on different examples
##################################################################################

# This script should abort on error
set -e

# Commands to check (okay means: return code 0)
# Commands to check (okay means: return code 0)
COMMANDS=(
  "python3 -c 'import torch'"
  "python3 -c 'import ultralytics'"
  "pytest -x"
)

# Corresponding to the commands, the expected behaviour
CONTEXTS=(
  "torch"
  "ultralytics"
  "pytest"
)

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Initialize an array to store results
declare -A RESULTS

# Loop over each command
for cmd in "${COMMANDS[@]}"; do
    echo -n "Checking $cmd..."
    
    # Run the command
    # If the command is a shell script, run with bash to avoid permission issues
    if [[ "$cmd" == *.sh ]]; then
        if [ -f "$cmd" ]; then
            if bash "$cmd"; then
                RESULTS["$cmd"]="${GREEN}✅ okay${NC}"
            else
                RESULTS["$cmd"]="${RED}❌ not working${NC}"
            fi
        else
            RESULTS["$cmd"]="${RED}❌ missing file${NC}"
        fi
    else
        if eval "$cmd"; then
            RESULTS["$cmd"]="${GREEN}✅ okay${NC}"
        else
            RESULTS["$cmd"]="${RED}❌ not working${NC}"
        fi
    fi
    echo -e "${RESULTS["$cmd"]}"
done

# Display the results
echo -e "\nResults:"
for index in "${!COMMANDS[@]}"; do
  cmd="${COMMANDS[$index]}"
  context="${CONTEXTS[$index]}"
  echo -e "$context : ${RESULTS["$cmd"]}"
done
