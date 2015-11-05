#!/bin/bash

for userdir in ./yoho-enroll/*/
do
    udir=${userdir%*/}
    echo "$udir"

    # Move the recordings in the 1/2/3/4 sessions folders
    # into current user folder
    # Delete the empty 1/2/3/4 session folders
    for sessiondir in $udir/*/
    do
       mv $sessiondir/*.wav $udir
       rm $sessiondir -r
    done

    for wav_path in $udir/*.wav
    do
        echo $'\t'"$wav_path"
        w_decode -o short_01 -f "$wav_path" "${wav_path%.*}.sph"
        sox -t sph "${wav_path%.*}.sph" -b 16 -t wav "${wav_path%.*}.uncomp.wav"
    done
done
