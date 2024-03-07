
#!/bin/bash

PBSIM3_DIR="vendor/pbsim3"

if [ -d "$PBSIM3_DIR" ]; then
    cd $PBSIM3_DIR
    echo "Downloading PBSIM3 profile into $PBSIM3_DIR ..."
    wget "https://www.dropbox.com/scl/fo/kqmr2fjo5yaqrdycfxv8k/h?rlkey=sff6e5aqvngvxjk12xka3yjww&e=1&dl=0"
    mv 'h?rlkey=sff6e5aqvngvxjk12xka3yjww&e=1&dl=0' download.zip
    echo "Exctracting files ..."
    unzip download.zip
    rm download.zip
    echo "Successful!"
else
    echo "Directory $PBSIM3_DIR does not exist!"
    echo "Install PBSIM3 by running \"python install_tools.py\" first, or change the path to PBSIM3 inside this script."
    exit 1
fi

