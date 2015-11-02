for dir in ./enroll/*/
do
    dir=${dir%*/}
    for dir2 in $dir/*.sph
    do
        echo "$dir2"
        echo "${dir2%.*}.sph"
        # ./sphere/bin/w_decode -o short_01 -f "$dir2" "${dir2%.*}.sph"
        sox -t sph "${dir2%.*}.sph" -b 16 -t wav "${dir2%.*}.uncomp.wav"
        #echo $dir2
    done
done
