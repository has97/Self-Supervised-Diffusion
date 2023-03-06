path='../imagenet20_val'
# dest='../diffimagenet'
z=''
strength='0.2'
steps='400'
nsamples='3'
for folder in  $(ls -d $path/* | cut -f3 -d'/');do
    # mkdir "$dest/$folder"
    # for file in $(ls -p $path/$folder | grep -v /);do
    #     echo "$path/$folder/.ipynb_checkpoints"
    # echo "$folder"
    rm -rf "$path/$folder/.ipynb_checkpoints"
        # done        
    done