mkdir 
for dir in ./enroll/*/
do
    dir=${dir%*/}
    dir1="${dir}/1"
    dir2="${dir}/2"
    dir3="${dir}/3"
    dir4="${dir}/4"
    aggregate_dir="${dir}/accumulate"
    # mkdir $aggregate_dir
    # echo $aggregate_dir
    mv $aggregate_dir/* $dir/
    #cp $dir1/* $aggregate_dir
    #cp $dir2/* $aggregate_dir
    #cp $dir3/* $aggregate_dir
    #cp $dir4/* $aggregate_dir
    echo $dir1
done
