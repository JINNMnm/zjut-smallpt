#!/bin/bash

EXEC=./smallpt_cpu
LOG=render_log.txt

echo "Rendering log from 4 to 100 samples (step 4)" > $LOG
echo "--------------------------------------------" >> $LOG

for ((spp=4; spp<=100; spp+=4))
do
    echo "Running $EXEC $spp"
    echo "[Spp: $spp]" >> $LOG
    $EXEC $spp >> $LOG 2>&1
    echo "" >> $LOG
done

echo "All tests completed. Log saved to $LOG."